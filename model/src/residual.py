import pandas as pd
import numpy as np
import lightgbm as lgb
from lightgbm import early_stopping, log_evaluation
import joblib
import logging
import shap  # 新增: 用于特征归因
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

# ==========================================
# 1. 彩色日志封装 (提升终端体验)
# ==========================================
class ColoredLogger:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'

    @staticmethod
    def info(msg):
        print(f"{ColoredLogger.GREEN}[INFO] {msg}{ColoredLogger.ENDC}")

    @staticmethod
    def step(msg):
        print(f"\n{ColoredLogger.CYAN}{ColoredLogger.BOLD}>>> {msg}{ColoredLogger.ENDC}")

    @staticmethod
    def warn(msg):
        print(f"{ColoredLogger.WARNING}[WARN] {msg}{ColoredLogger.ENDC}")

    @staticmethod
    def error(msg):
        print(f"{ColoredLogger.FAIL}[ERROR] {msg}{ColoredLogger.ENDC}")
    
    @staticmethod
    def result(msg):
        print(f"{ColoredLogger.BLUE}[RESULT] {msg}{ColoredLogger.ENDC}")

# ==========================================
# 2. 配置与特征列表
# ==========================================
ATOMIC_FEATURES = [
    # --- 共情力 ---
    'painpoint_word_cnt', 'audience_word_cnt', 'sentiment_intensity', 'emoji_ratio', 'title_question_flag',
    # --- 专业力 ---
    'ingredient_cnt', 'efficacy_cnt', 'solution_pattern_flag', 'comparison_flag', 'info_density_score',
    # --- 易读力 ---
    'readability_score', 'paragraph_cnt', 'list_structure_flag', 'avg_sentence_len',
    # --- 搜索力 ---
    'title_keyword_cov', 'content_keyword_density', 'tag_content_consistency', 'vertical_tag_ratio',
    # --- 吸睛力 ---
    'brightness_mean', 'saturation_mean', 'contrast_score', 'face_area_ratio', 'text_overlay_ratio', 'sharpness_score'
]

class ResidualModelPipeline:
    def __init__(self, model_path='models/residual/residual_model.pkl'):
        self.model_path = model_path
        self.model = None
        self.explainer = None
        
        # 特征工程
        self.feature_cols = ATOMIC_FEATURES + ['baseline_pred']
        self.target_col = 'log_residual_score'
        
        self.context_distributions = {}

    # ----------------------------------------------------------------
    # 数据准备
    # ----------------------------------------------------------------
    def prepare_data(self, df: pd.DataFrame, is_train=True):
        """清洗、计算对数Target"""
        if 'baseline_pred' not in df.columns:
            raise ValueError("缺失 'baseline_pred' 列")
        
        df_proc = df.copy()
        df_proc[ATOMIC_FEATURES] = df_proc[ATOMIC_FEATURES].fillna(0)
        
        # 1. 强制 Baseline 预测值非负 (防止 log 报错)
        df_proc['baseline_pred'] = df_proc['baseline_pred'].clip(lower=0.1)
        
        if is_train:
            if 'EVI_rate' not in df_proc.columns:
                raise ValueError("缺失真实标签")
            
            df_proc['EVI_rate'] = df_proc['EVI_rate'].clip(lower=0.1)
            
            # 计算真实值与baseline预测值的对数残差
            df_proc['log_evi'] = np.log1p(df_proc['EVI_rate'])
            df_proc['log_base'] = np.log1p(df_proc['baseline_pred'])
            df_proc['log_residual_score'] = df_proc['log_evi'] - df_proc['log_base']
            
            # 3. 异常值处理
            mean_res = df_proc['log_residual_score'].mean()
            std_res = df_proc['log_residual_score'].std()
            mask = (df_proc['log_residual_score'] >= mean_res - 3*std_res) & \
                   (df_proc['log_residual_score'] <= mean_res + 3*std_res)
            
            dropped = len(df_proc) - mask.sum()
            if dropped > 0:
                ColoredLogger.warn(f"Log空间清洗: 剔除 {dropped} 条异常倍率样本")
                df_proc = df_proc[mask].copy()
                
            # 构建分布库 (依然存原始 EVI 用于百分位计算)
            if 'mkctx__b_primary_category' in df_proc.columns:
                self.context_distributions = df_proc.groupby('mkctx__b_primary_category')['EVI_rate'].apply(list).to_dict()
                
        return df_proc

    # ----------------------------------------------------------------
    # 核心训练逻辑 (含优化参数)
    # ----------------------------------------------------------------
    def train(self, df: pd.DataFrame):
        ColoredLogger.step("Step 1: 数据划分")
        df_full = self.prepare_data(df, is_train=True)
        X = df_full[self.feature_cols]
        y = df_full[self.target_col]
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # 参数配置 (保持 Log-Space 优化配置)
        params = {
            'objective': 'regression_l1', # L1 损失 (MAE)
            'metric': 'mae',              # 评估指标
            'learning_rate': 0.03,
            'num_leaves': 31,
            'max_depth': 6,
            'min_child_samples': 30,
            'reg_alpha': 0.5,
            'reg_lambda': 0.5,
            'n_jobs': -1,
            'verbosity': -1,
            'seed': 42
        }
        
        # 使用 CV 寻找最佳迭代次数
        dtrain = lgb.Dataset(X_train, label=y_train)
        
        cv_results = lgb.cv(
            params,
            dtrain,
            num_boost_round=2000, # 增加上限，防止欠拟合
            nfold=5,
            stratified=False,
            seed=42,
            callbacks=[
                lgb.early_stopping(stopping_rounds=50),
                lgb.log_evaluation(period=50)
            ]
        )
        
        # cv_results 的 keys 通常是 'valid l1-mean' 或 'valid rmse-mean'
        eval_key = [k for k in cv_results.keys() if 'mean' in k][0]
        
        best_n = len(cv_results[eval_key])
        best_score = cv_results[eval_key][-1]
        
        ColoredLogger.result(f"CV 最佳轮数: {best_n}, 最佳得分 ({eval_key}): {best_score:.4f}")
        
        # 使用最佳参数在 Train集 上重新训练
        self.model = lgb.LGBMRegressor(n_estimators=best_n, **params)
        self.model.fit(X_train, y_train)
        
        # 评估 Test 集 (泛化能力)
        y_pred_test = self.model.predict(X_test)
        test_r2 = r2_score(y_test, y_pred_test)
        test_mae = mean_absolute_error(y_test, y_pred_test)
        
        ColoredLogger.result(f"测试集 (Log空间泛化) R2: {test_r2:.4f}, MAE: {test_mae:.4f}")
        
        # 保存
        self.explainer = shap.TreeExplainer(self.model)
        joblib.dump({'model': self.model, 'ctx_dist': self.context_distributions}, self.model_path)

    # ----------------------------------------------------------------
    # 百分位计算逻辑 (Context Percentile)
    # ----------------------------------------------------------------
    def calculate_percentile(self, score, category):
        """计算分数在同赛道历史数据中的百分位排名 (0~100)"""
        if category not in self.context_distributions:
            return 50.0 # 无参考数据，默认中位数
        
        ref_scores = self.context_distributions[category]
        # 使用 percentileofscore 计算排名
        pct = stats.percentileofscore(ref_scores, score, kind='rank')
        return pct

    # ----------------------------------------------------------------
    # 预测与解释
    # ----------------------------------------------------------------
    def predict_and_explain(self, df: pd.DataFrame):
        ColoredLogger.step("Step 3: 推理与归因分析")
        
        if self.model is None:
            data = joblib.load(self.model_path)
            self.model = data['model']
            self.context_distributions = data['ctx_dist']
            self.explainer = shap.TreeExplainer(self.model)

        df_pred = self.prepare_data(df, is_train=False)
        X = df_pred[self.feature_cols]
        
        # 1. 预测 (得到的是 Log 空间的残差)
        log_residual_preds = self.model.predict(X)
        
        # 2. 还原最终预测值
        # 公式: Final = exp( log(Base+1) + log_residual ) - 1
        baseline_log = np.log1p(df_pred['baseline_pred'])
        final_log = baseline_log + log_residual_preds
        final_preds = np.expm1(final_log) # 还原回 EVI 空间
        
        # 3. 计算 SHAP
        shap_values = self.explainer.shap_values(X)
        
        results = []
        for idx, (log_res, final) in enumerate(zip(log_residual_preds, final_preds)):
            row_data = df.iloc[idx]
            category = row_data.get('mkctx__b_primary_category', 'Unknown')
            
            percentile = self.calculate_percentile(final, category)
            
            # [关键] 解释 SHAP 值
            # 在 Log 空间，SHAP值 +0.1 意味着结果增加 e^0.1 ≈ 10% (乘性增益)
            sample_shap = shap_values[idx]
            feat_contribs = list(zip(self.feature_cols, sample_shap))
            feat_contribs.sort(key=lambda x: x[1], reverse=True)
            
            # 显示格式优化：显示带来的提升百分比
            def fmt_shap(val):
                pct = (np.expm1(abs(val))) * 100
                sign = "+" if val > 0 else "-"
                return f"{sign}{pct:.1f}%"

            top_pos = [f"{k}({fmt_shap(v)})" for k, v in feat_contribs if v > 0 and k != 'baseline_pred'][:3]
            top_neg = [f"{k}({fmt_shap(v)})" for k, v in feat_contribs if v < 0 and k != 'baseline_pred'][-3:]
            
            results.append({
                'note_id': row_data['note_id'],
                'category': category,
                'EVI_rate': row_data['EVI_rate'],
                'baseline': row_data['baseline_pred'],
                'log_residual': log_res,                 # 对数残差标签
                'multiplier': f"x{np.exp(log_res):.2f}", # 放大倍数
                'final_pred': final,
                'beat_competitors': f"{percentile:.1f}%",
                'positive_drivers': top_pos,
                'negative_drags': top_neg
            })
            
        return pd.DataFrame(results)

# ==========================================
# 模拟执行
# ==========================================
if __name__ == "__main__":
    # 模拟数据
    df = pd.read_csv('data/data_with_features_baseline.csv')
    
    pipeline = ResidualModelPipeline()
    
    # 1. 训练 (包含数据划分与泛化检测)
    pipeline.train(df)
    
    # 2. 预测与归因
    res_df = pipeline.predict_and_explain(df)
    
    ColoredLogger.result("最终预测与归因结果:")
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)
    print(res_df[['category', 'final_pred', 'beat_competitors', 'positive_drivers', 'negative_drags']])
    res_df.to_csv('data/data_with_features_residual.csv', index=False, encoding='utf-8-sig')
    print(f"最终预测与归因结果已保存至: data/data_with_features_residual.csv")