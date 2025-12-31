import pandas as pd
import numpy as np
import logging
import joblib
from typing import Tuple, List, Dict
from catboost import CatBoostRegressor, Pool, cv
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 定义数据字段常量
KEEP_COLS = [
    'note_id', 'title', 'desc', 'tags', 'cover_path', 'EVI_rate',
    # --- 商业上下文 (Baseline 特征) ---
    'mkctx__b_brand_tier', 'mkctx__b_brand_origin', 'mkctx__b_product_stage',
    'mkctx__b_primary_category', 'mkctx__b_campaign_goal',
    'mkctx__c_need_archetype', 'mkctx__c_efficacy_goal', 'mkctx__c_lifecycle',
    'mkctx__c_channel_behavior',
    'mkctx__scene_marketing_nodes', 'mkctx__scene_season', 'mkctx__scene_climate',
    # --- 原子特征 (Residual 特征 - Baseline 不使用!) ---
    'painpoint_word_cnt', 'audience_word_cnt', 'sentiment_intensity', 'emoji_ratio',
    'title_question_flag', 'ingredient_cnt', 'efficacy_cnt', 'solution_pattern_flag',
    'comparison_flag', 'info_density_score', 'readability_score', 'paragraph_cnt',
    'list_structure_flag', 'avg_sentence_len', 'title_keyword_cov',
    'content_keyword_density', 'tag_content_consistency', 'vertical_tag_ratio',
    'brightness_mean', 'saturation_mean', 'contrast_score', 'face_area_ratio',
    'text_overlay_ratio', 'sharpness_score'
]

class BaselineModelPipeline:
    """
    Baseline 预测模型管道
    
    目标: 预测特定商业上下文(Context)下的平均热度潜力(EVI_rate)。
    核心逻辑: 仅使用 mkctx__ 开头的特征进行训练，忽略内容质量特征。
    """
    
    def __init__(self, model_path='models/baseline/baseline_model.cbm'):
        self.model_path = model_path
        self.model = None
        
        # 自动筛选商业上下文特征 (以 mkctx__ 开头)
        self.context_features = [
            col for col in KEEP_COLS 
            if col.startswith('mkctx__') and col not in ['mkctx__note_id', 'mkctx__schema_version']
        ]
        self.target_col = 'EVI_rate'
        
        logger.info(f"Baseline 模型初始化完成。")
        logger.info(f"特征列表 ({len(self.context_features)}个): {self.context_features}")

    def load_and_validate_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        步骤 1: 数据校验与字段过滤
        """
        logger.info("步骤 1/5: 数据校验...")
        
        # 1. 检查必要字段是否存在
        missing_cols = [c for c in self.context_features + [self.target_col] if c not in df.columns]
        if missing_cols:
            raise ValueError(f"输入数据缺失必要字段: {missing_cols}")
            
        # 2. 仅保留 KEEP_COLS 中的有效列
        available_cols = [c for c in KEEP_COLS if c in df.columns]
        df_subset = df[available_cols].copy()
        
        logger.info(f"数据加载成功，样本数: {len(df_subset)}")
        return df_subset

    def preprocess_outliers(self, df: pd.DataFrame, remove_outliers=True) -> pd.DataFrame:
        """
        步骤 2: IQR 异常值处理
        
        原因: Baseline 代表"平均水平"，极端的爆款(EVI极高)或死贴(EVI极低)属于"噪音"，
        会拉偏模型对"常态"的拟合。
        """
        logger.info("步骤 2/5: 异常值处理 (IQR Cleaning)...")
        
        if not remove_outliers:
            return df
        
        Q1 = df[self.target_col].quantile(0.25)
        Q3 = df[self.target_col].quantile(0.75)
        IQR = Q3 - Q1
        
        # 定义上下界 (1.5倍 IQR 是标准统计学边界)
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        # 过滤数据
        df_clean = df[(df[self.target_col] >= lower_bound) & (df[self.target_col] <= upper_bound)].copy()
        
        drop_cnt = len(df) - len(df_clean)
        logger.info(f"剔除异常样本数: {drop_cnt}, 剩余样本数: {len(df_clean)}")
        logger.info(f"EVI_rate 范围锁定: [{lower_bound:.4f}, {upper_bound:.4f}]")
        
        return df_clean

    def preprocess_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        步骤 3: 缺失值填充与类型转换
        """
        logger.info("步骤 3/5: 特征预处理...")
        
        df_processed = df.copy()
        
        for col in self.context_features:
            # 1. 缺失值填充: 商业属性缺失通常意味着"未知"或"其他"
            # CatBoost 对 NaN 敏感，建议显式填充字符串
            df_processed[col] = df_processed[col].fillna('Unknown')
            
            # 2. 类型强制转换: 确保全是字符串
            # 注意: 某些 List 类型的列 (如 "['油皮', '干皮']") 会自动转为字符串形式
            # 这种组合字符串正好代表了一种特定的"细分赛道"，直接作为 Category 处理即可
            df_processed[col] = df_processed[col].astype(str)
            
        return df_processed

    def train(self, df: pd.DataFrame, perform_cv=True):
        """
        步骤 4: 模型训练 (包含交叉验证优化)
        """
        logger.info("步骤 4/5: 开始训练 Baseline 模型...")
        
        X = df[self.context_features]
        y = df[self.target_col]
        
        # CatBoost 需要指定哪些列是类别特征
        # 由于我们只选用了 mkctx__，它们全都是类别特征
        cat_features_indices = list(range(len(self.context_features)))
        
        # 初始化模型参数
        # loss_function='RMSE': 回归任务标准损失
        # learning_rate=0.05: 较小的学习率有助于防止过拟合
        # depth=6: 树深，6-8是 CatBoost 的黄金区间
        # l2_leaf_reg=3: L2正则化系数，抑制过拟合
        params = {
            'iterations': 1000,
            'learning_rate': 0.05,
            'depth': 6,
            'l2_leaf_reg': 3,
            'loss_function': 'RMSE',
            'eval_metric': 'RMSE',
            'cat_features': cat_features_indices,
            'verbose': 100,
            'random_seed': 42
        }
        
        train_pool = Pool(X, y, cat_features=cat_features_indices)
        
        if perform_cv:
            logger.info("正在执行 5-Fold 交叉验证以寻找最佳迭代次数...")
            cv_results = cv(
                pool=train_pool,
                params=params,
                fold_count=5,
                early_stopping_rounds=50,
                plot=False,
                verbose=False
            )
            best_iteration = cv_results['test-RMSE-mean'].idxmin()
            logger.info(f"CV 优化完成。最佳迭代次数: {best_iteration}, 最佳 RMSE: {cv_results['test-RMSE-mean'].min():.4f}")
            
            # 使用最佳轮数重新设置参数
            params['iterations'] = best_iteration
        
        # 使用全量数据进行最终训练
        self.model = CatBoostRegressor(**params)
        self.model.fit(X, y)
        
        # 保存模型
        self.model.save_model(self.model_path)
        logger.info(f"模型已保存至: {self.model_path}")
        
        # 简单评估 (In-sample)
        preds = self.model.predict(X)
        logger.info(f"训练集 MAE: {mean_absolute_error(y, preds):.4f}")
        logger.info(f"训练集 R2: {r2_score(y, preds):.4f}")

    def predict_baseline(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        步骤 5: 预测 (输出包含 Baseline 预测值的 DataFrame)
        """
        logger.info("步骤 5/5: 生成 Baseline 预测...")
        
        if self.model is None:
            try:
                self.model = CatBoostRegressor()
                self.model.load_model(self.model_path)
            except Exception as e:
                raise Exception("模型未加载且文件不存在，请先训练模型。")
        
        # 预处理 (即使是预测，也要做同样的特征转换)
        df_processed = self.preprocess_features(df)
        X = df_processed[self.context_features]
        
        # 生成预测值
        baseline_preds = self.model.predict(X)
        
        # 将预测值拼接到原 DataFrame
        result_df = df.copy()
        result_df['baseline_pred'] = baseline_preds
        
        # 计算 Residual (残差) = 真实表现 - 平均表现
        # 正残差表示表现优于大盘，负残差表示表现差于大盘
        result_df['residual_score'] = result_df[self.target_col] - result_df['baseline_pred']
        
        return result_df

# ==========================================
# 调用示例
# ==========================================
if __name__ == "__main__":
    # 1. 加载你的数据 (假设这里是你的 DataFrame)
    input_file = 'data/data_with_features.csv'    
    df = pd.read_csv(input_file)
    
    # 2. 初始化管道
    pipeline = BaselineModelPipeline()

    try:
        # 3. 执行流程
        # 3.1 校验
        df_valid = pipeline.load_and_validate_data(df)
        
        # 3.2 清洗 (IQR 过滤异常值)
        df_clean = pipeline.preprocess_outliers(df_valid, remove_outliers=True)
        
        # 3.3 预处理 (填充、类型转换)
        df_ready = pipeline.preprocess_features(df_clean)
        
        # 3.4 训练 (带 CV 优化)
        pipeline.train(df_ready, perform_cv=True)
        
        # 3.5 预测 (生成 baseline_pred 和 residual_score)
        df_result = pipeline.predict_baseline(df) # 注意：预测时可以对全量数据预测，包括异常值
        
        print("\n=== 预测结果预览 ===")
        print(df_result[['mkctx__b_primary_category', 'EVI_rate', 'baseline_pred', 'residual_score']].head())

        df_result.to_csv('data/data_with_features_baseline.csv', index=False, encoding='utf-8-sig')
        print(f"Baseline预测结果已保存至: data/data_with_features_baseline.csv")
        
    except Exception as e:
        logger.error(f"执行出错: {e}")
        import traceback
        traceback.print_exc()