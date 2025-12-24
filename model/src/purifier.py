"""
数据提纯器：将原始数据提纯为模型可用的数据
"""

import io
import os
import re
import numpy as np
import pandas as pd

from typing import Optional, Dict
from datetime import datetime, timedelta

KEEP_COLUMNS = [
    'note_id', 'title', 'desc', 'tags', 'publish_time', 
    'cover_path', 'liked_count', 'hot_level', 'comments_sample'
]

class HotLevelLabeler:
    """
    爆款标签生成器
    
    功能：
    1. 消除时间积累效应，计算 EVI 速率（Engagement Velocity Index）
    2. 基于分组（账号/关键词）进行相对评级
    3. 生成 S/A/B/C/D 五级爆款标签
    
    使用方法：
    >>> labeler = HotLevelLabeler()
    >>> df_labeled = labeler.fit_transform(df)
    """
    
    def __init__(self, evi_weights=None,time_smoothing_hours=2.0,platform_default='xhs'):
        self.evi_weights = evi_weights or {
            'comment': 3.0, 
            'save': 5.0, 
            'share': 10.0, 
            'like': 1.0
        }
        self.time_smoothing_hours = time_smoothing_hours    # 时间平滑因子（小时），防止短时爆发数值过大，默认 2.0
        self.platform_default = platform_default            # 默认平台名称，当数据中缺失 platform 字段时使用，默认 'xhs'
        
        self.group_stats_ = None                            # 存储分组统计信息（供调试和分析使用）
    
    def _calculate_evi(self, df):                           
        """
        计算原始 EVI (Engagement Value Index)
        """

        metric_cols = ['comment_count', 'collected_count', 'share_count', 'liked_count']
        
        # 数据清洗
        for col in metric_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
        
        # 计算加权 EVI
        df['raw_EVI'] = (
            df['comment_count'] * self.evi_weights['comment'] +
            df['collected_count'] * self.evi_weights['save'] +
            df['share_count'] * self.evi_weights['share'] +
            df['liked_count'] * self.evi_weights['like']
        )
        
        return df
    
    def _calculate_velocity(self, df):
        """
        消除时间积累效应，计算 EVI 速率
        """
        
        # 解析时间字段
        df['publish_time'] = pd.to_datetime(df['publish_time'], errors='coerce')
        df['crawl_time'] = pd.to_datetime(df['crawl_time'], errors='coerce')
        
        # 确定基准时间（可选：使用当前时间或最小爬取时间）
        # 对于历史数据，使用最小爬取时间 + 1小时作为统一参考点
        reference_time = df['crawl_time'].min() + timedelta(hours=1)
        
        # 计算发布时长（小时）
        df['hours_diff'] = (reference_time - df['publish_time']).dt.total_seconds() / 3600
        
        # 处理异常时间（防止负数）
        df['hours_diff'] = df['hours_diff'].apply(lambda x: max(0, x))
        
        # 计算 EVI 速率 = EVI / (时长 + 平滑因子)
        df['EVI_velocity'] = df['raw_EVI'] / (df['hours_diff'] + self.time_smoothing_hours)
        
        return df
    
    def _generate_group_key(self, row):
        """
        生成分组键，用于分组归一化
        
        分组逻辑：
        - 账号来源：platform_user_用户ID
        - 关键词来源：platform_kw_搜索词
        """
        src = str(row.get('src', '')).lower()
        platform = str(row.get('platform', self.platform_default)).lower()
        
        if 'account' in src or '账号' in src:
            user_id = str(row.get('user_id', 'unknown')).lower()
            return f"{platform}_user_{user_id}"
        else:
            keyword = str(row.get('search_keyword', 'unknown')).lower()
            return f"{platform}_kw_{keyword}"
    
    def _calculate_group_thresholds(self, df):
        """
        基于分组计算分位数阈值
        """
        
        # 生成分组键
        df['group_key'] = df.apply(self._generate_group_key, axis=1)
        
        # 计算各组的分位数（50%, 80%, 90%, 95%）
        group_stats = df.groupby('group_key')['EVI_velocity'].quantile([0.5, 0.8, 0.90, 0.95]).unstack()
        group_stats.columns = ['t50', 't80', 't90', 't95']
        
        # 保存统计信息
        self.group_stats_ = group_stats
        
        # 合并阈值到原数据
        df = df.merge(group_stats, on='group_key', how='left')
        
        return df
    
    def _get_grade(self, row):
        """
        根据 EVI 速率和分组阈值进行评级
        
        评级规则：
        - S: > 95th percentile (Top 5%)
        - A: > 90th percentile (Top 10%)
        - B: > 80th percentile (Top 20%)
        - C: > 50th percentile (Top 50%)
        - D: <= 50th percentile (Bottom 50%)
        """
        score = row['EVI_velocity']
        
        if score > row['t95']: return 'S'
        elif score > row['t90']: return 'A'
        elif score > row['t80']: return 'B'
        elif score > row['t50']: return 'C'
        else: return 'D'
    
    def fit_transform(self, df):
        """
        主处理流程：对输入数据进行爆款标签生成
        
        参数:
            df (pd.DataFrame): 输入数据，需包含以下字段：
                - comment_count, collected_count, share_count, liked_count
                - publish_time, crawl_time
                - src, search_keyword (可选: platform, user_id)
        
        返回:
            pd.DataFrame: 带有爆款标签的数据，新增字段：
                - raw_EVI: 原始互动价值
                - hours_diff: 发布时长（小时）
                - EVI_velocity: EVI 速率
                - group_key: 分组键
                - t50, t80, t90, t95: 分组阈值
                - hot_level: 爆款等级 (S/A/B/C/D)
        """
        df = df.copy()
        
        # 1. 计算原始 EVI
        df = self._calculate_evi(df)
        
        # 2. 计算 EVI 速率（消除时间影响）
        df = self._calculate_velocity(df)
        
        # 3. 计算分组阈值
        df = self._calculate_group_thresholds(df)
        
        # 4. 生成爆款标签
        df['hot_level'] = df.apply(self._get_grade, axis=1)
        
        return df
    
    def get_group_stats(self):
        """
        获取分组统计信息（需先调用 fit_transform）
        """
        if self.group_stats_ is None:
            raise ValueError("请先调用 fit_transform() 方法生成标签")
        return self.group_stats_

class DataCleaner:
    """
    负责原始数据的清洗、格式化与特征预处理的基础类
    """

    def __init__(self, raw_data_str: str):
        self.raw_data = raw_data_str
        self.df: Optional[pd.DataFrame] = None

    def load_data(self):
        """加载数据"""
        self.df = pd.read_csv(self.raw_data)
        print(f"[INFO] 原始数据加载成功，共 {len(self.df)} 条样本。")

    def _clean_text_field(self, text: str) -> str:
        """清洗单个文本字段"""

        # 如果文本为空，返回空字符串
        if pd.isna(text):
            return ""

        # 1. 去除多余的换行和首尾空格
        text = str(text).strip()
        # 2. 将连续的空白字符替换为单个空格
        text = re.sub(r'\s+', ' ', text)
        return text

    def _parse_tags(self, tag_str: str) -> list:
        """将逗号分隔的字符串转换为列表"""

        # 如果文本为空，返回空列表
        if pd.isna(tag_str):
            return []

        # 处理中文逗号，按逗号分割为列表，去除空格
        return [t.strip() for t in tag_str.replace('，', ',').split(',') if t.strip()]

    def process(self):
        """执行提纯主流程"""

        # 基础文本清洗 (Title & Desc)
        print("[PROCESS] 正在清洗文本字段 (Title, Desc)...")
        self.df['title'] = self.df['title'].apply(self._clean_text_field)
        self.df['desc'] = self.df['desc'].apply(self._clean_text_field)
        for col in ["title", "desc", "comments_sample"]:
            if col in self.df.columns:
                self.df[col] = self.df[col].apply(lambda x: self._clean_text_field(x) if not pd.isna(x) else "")

        # 标签与评论列表化 (Tags & Comments)
        print("[PROCESS] 正在解析 Tags 和 Comments...")
        self.df["tags_list"] = self.df["tags"].apply(self._parse_tags)
        self.df['comments_list'] = self.df['comments_sample'].apply(
            lambda x: str(x).split('||') if not pd.isna(x) else []
        )

        # 时间结构化 (Temporal Structuring)
        print("[PROCESS] 正在处理时间字段...")
        self.df['publish_time'] = pd.to_datetime(self.df['publish_time'], errors='coerce')

        # 构建爆款标签 (Hot Level Label Construction)
        print("[PROCESS] 正在构建爆款等级 (Hot Level)...")
        self.df['hot_level'] = HotLevelLabeler().fit_transform(self.df)['hot_level']

        # 图像路径标准化
        self.df["cover_path"] = self.df["cover_path"].apply(lambda x: "" if pd.isna(x) or str(x).strip().lower() == "nan" else str(x).strip())
        self.df = self.df[self.df["cover_path"].str.strip() != ""].reset_index(drop=True)                      # 去除图片路径为空的数据
        self.df = self.df[self.df["cover_path"].apply(lambda p: os.path.exists(p))].reset_index(drop=True)     # 去除图片路径不存在的数据

        # 数据列筛选与重命名
        self.df['tags'] = self.df['tags_list']
        final_columns = [c for c in KEEP_COLUMNS if c in self.df.columns]

        self.df = self.df[final_columns]

        assert self.df["hot_level"].notna().all(), "hot_level 存在缺失"
        assert self.df["cover_path"].map(type).eq(str).all(), "cover_path 存在非字符串"
        
        print("[INFO] 数据清洗完成!")

    def get_cleaned_data(self) -> pd.DataFrame:
        """获取提纯后的数据"""

        return self.df

    def show_statistics(self):
        """打印数据分布，检查类别平衡性"""

        print("\n等级分布:")
        print(f"{'等级':<10} | {'数量':>8} | {'占比':>8}")
        print("-" * 30)
        for grade in ['S', 'A', 'B', 'C', 'D']:
            count = self.df['hot_level'].value_counts().get(grade, 0)
            ratio = count / len(self.df) * 100
            print(f"{grade:<10} | {count:>8} | {ratio:>7.2f}%")

# --- 执行入口 ---
if __name__ == "__main__":
    # 读取数据
    raw_csv_data = './data/clean_20251223_202051.csv'

    # 实例化并运行 pipeline
    cleaner = DataCleaner(raw_csv_data)          # 数据提纯器实例化
    cleaner.load_data()                          # 加载数据
    cleaner.process()                            # 数据提纯
    cleaner.show_statistics()                    # 打印数据分布，检查类别平衡性

    # 获取最终 DataFrame
    cleaned_df = cleaner.get_cleaned_data()     # 提纯后的数据

    # 保存数据
    output_file = './data/data_with_label'

    cleaned_df.to_csv(f'{output_file}.csv', index=False)
    cleaned_df.to_excel(f'{output_file}.xlsx', index=False)

    print(f"\n[INFO] 提纯数据已保存至: {output_file}.csv|xlsx")
