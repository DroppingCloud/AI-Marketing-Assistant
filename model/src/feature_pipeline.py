"""
原子特征提取流水线 V2

Visual: brightness_mean, saturation_mean, contrast_score, face_area_ratio, text_overlay_ratio, sharpness_score
Empathy: painpoint_word_cnt, audience_word_cnt, sentiment_intensity, emoji_ratio, title_question_flag
Authority: ingredient_cnt, efficacy_cnt, solution_pattern_flag, comparison_flag, info_density_score
Readability: readability_score, paragraph_cnt, list_structure_flag, avg_sentence_len
Searchability: title_keyword_cov, content_keyword_density, tag_content_consistency, vertical_tag_ratio
"""
import os
import re
import json
import sys
import logging
from pathlib import Path
from typing import Dict, List, Set, Any, Optional

import jieba
import numpy as np
import pandas as pd
from tqdm import tqdm
from snownlp import SnowNLP

import cv2

# ===========================
# 保留字段配置
# ===========================
KEEP_COLS = [
    'note_id',
    'title',
    'desc',
    'tags',
    'cover_path',
    'EVI_rate',

    'mkctx__b_brand_tier',
    'mkctx__b_brand_origin',
    'mkctx__b_product_stage',
    'mkctx__b_primary_category',
    'mkctx__b_campaign_goal',

    'mkctx__c_need_archetype',
    'mkctx__c_efficacy_goal',
    'mkctx__c_lifecycle',
    'mkctx__c_channel_behavior',

    'mkctx__scene_marketing_nodes',
    'mkctx__scene_season',
    'mkctx__scene_climate',

    'painpoint_word_cnt',
    'audience_word_cnt',
    'sentiment_intensity',
    'emoji_ratio',
    'title_question_flag',

    'ingredient_cnt',
    'efficacy_cnt',
    'solution_pattern_flag',
    'comparison_flag',
    'info_density_score',

    'readability_score',
    'paragraph_cnt',
    'list_structure_flag',
    'avg_sentence_len',

    'title_keyword_cov',
    'content_keyword_density',
    'tag_content_consistency',
    'vertical_tag_ratio',

    'brightness_mean',
    'saturation_mean',
    'contrast_score',
    'face_area_ratio',
    'text_overlay_ratio',
    'sharpness_score'
]

# ===========================
# 日志配置
# ===========================
class ColorFormatter(logging.Formatter):
    """彩色日志格式化器"""
    RESET = "\033[0m"
    COLORS = {
        'DEBUG': '\033[36m',    # 青色
        'INFO': '\033[32m',     # 绿色
        'WARNING': '\033[33m',  # 黄色
        'ERROR': '\033[31m',    # 红色
    }

    def format(self, record: logging.LogRecord) -> str:
        levelname = record.levelname
        message = super().format(record)
        return f"{self.COLORS.get(levelname, self.RESET)}{message}{self.RESET}"


def setup_logger(name: str = __name__, level: int = logging.INFO) -> logging.Logger:
    """配置彩色日志输出"""
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setLevel(level)
        formatter = ColorFormatter(
            fmt='%(asctime)s [%(levelname)s] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    
    return logger


logger = setup_logger()


# ===========================
# 词库注册中心
# ===========================
class LexiconRegistry:
    """
    新词库注册中心
    加载统一格式的词库文件 (audience, efficacy, ingredients, pain_points, vertical_tags)
    """
    def __init__(self, base_dir: str = 'lexicons'):
        self.base_dir = Path(base_dir)
        
        # 词库集合
        self.audience: Set[str] = set()        # 人群词库
        self.efficacy: Set[str] = set()        # 功效词库
        self.ingredients: Set[str] = set()     # 成分词库
        self.pain_points: Set[str] = set()     # 痛点词库
        self.vertical_tags: Set[str] = set()   # 垂直标签词库
        
        # 词库元数据
        self.metadata: Dict[str, Dict] = {}
        
        # 加载所有词库
        self._load_all_lexicons()
        logger.info(f"✓ 词库加载完成 | 人群:{len(self.audience)} | 功效:{len(self.efficacy)} | "
                   f"成分:{len(self.ingredients)} | 痛点:{len(self.pain_points)} | 垂直标签:{len(self.vertical_tags)}")

    def _load_all_lexicons(self):
        """加载所有新词库"""
        lexicon_configs = [
            ('audience.json', self.audience),
            ('efficacy.json', self.efficacy),
            ('ingredients.json', self.ingredients),
            ('pain_points.json', self.pain_points),
            ('vertical_tags.json', self.vertical_tags)
        ]
        
        for filename, target_set in lexicon_configs:
            self._load_lexicon(filename, target_set)
    
    def _load_lexicon(self, filename: str, target_set: Set[str]):
        """加载单个词库文件"""
        file_path = self.base_dir / filename
        if not file_path.exists():
            logger.warning(f"词库文件不存在: {file_path}")
            return
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # 保存元数据
            lexicon_id = data.get('lexicon_id', filename.replace('.json', ''))
            self.metadata[lexicon_id] = {
                'title': data.get('title', ''),
                'term_count': data.get('term_count', 0),
                'generated_at': data.get('generated_at', '')
            }
            
            # 提取词条
            terms = data.get('terms', [])
            for term_obj in terms:
                # 本词
                term = term_obj.get('term', '').strip()
                if term:
                    target_set.add(term)
                    
                # 同义词
                synonyms = term_obj.get('synonyms', [])
                for syn in synonyms:
                    if syn and syn.strip():
                        target_set.add(syn.strip())
            
            logger.debug(f"[{filename}] 加载 {len(target_set)} 个词条")
            
        except Exception as e:
            logger.error(f"加载词库失败 [{filename}]: {e}")

# ===========================
# 结构模式（正则）
# ===========================
class PatternRegistry:
    def __init__(self, base_dir: str = 'lexicons'):
        self.base_dir = Path(base_dir)
        self.usage_patterns: List[re.Pattern] = []
        self.comparison_patterns: List[re.Pattern] = []
        self._load_or_default()

    def _load_or_default(self):
        usage = [
            r"(怎么用|如何用|用法|步骤|教程|流程|顺序)",
            r"(先.*再.*|第[一二三四五六七八九十\d]+步)",
            r"(建议|推荐).*(方法|步骤|搭配)",
        ]
        comparison = [
            r"(对比|测评|横评|PK|VS|哪个更|谁更|区别|优缺点|避雷)",
            r"(更适合|不适合|胜出|翻车)",
        ]
        self.usage_patterns = [re.compile(p) for p in usage]
        self.comparison_patterns = [re.compile(p, flags=re.IGNORECASE) for p in comparison]

# ===========================
# 视觉特征提取器（V1 视觉特征回归 + OCR lazy）
# ===========================
class VisualFeatureExtractor:
    def __init__(
        self,
        enable_ocr: bool = True,
        ocr_langs: tuple = ('ch_sim', 'en'),
        ocr_gpu: bool = False,
        haarcascade_path: Optional[str] = None
    ):
        self.enable_ocr = enable_ocr
        self.ocr_langs = ocr_langs
        self.ocr_gpu = ocr_gpu
        self._ocr_reader = None  # 真正 lazy

        # if haarcascade_path is None:
        #     haarcascade_path = "/opt/anaconda3/envs/ML/lib/python3.12/site-packages/cv2/data/haarcascade_frontalface_default.xml"
        # self.face_cascade = cv2.CascadeClassifier(haarcascade_path)
        if haarcascade_path is None:
            haarcascade_path = os.path.join(cv2.data.haarcascades, "haarcascade_frontalface_default.xml")

        self.face_cascade = cv2.CascadeClassifier(haarcascade_path)

        if self.face_cascade.empty():
            logger.warning(f"⚠️ Haarcascade 加载失败: {haarcascade_path}（将跳过人脸检测）")


    def _get_ocr_reader(self):
        if not self.enable_ocr:
            return None
        if self._ocr_reader is None:
            try:
                import easyocr
                logger.info("首次使用 OCR：初始化 EasyOCR Reader...")
                self._ocr_reader = easyocr.Reader(list(self.ocr_langs), gpu=self.ocr_gpu, verbose=False)
                logger.info(f"✓ OCR Reader 初始化完成 | langs={self.ocr_langs} gpu={self.ocr_gpu}")
            except Exception as e:
                logger.error(f"OCR Reader 初始化失败: {e}")
                self._ocr_reader = None
        return self._ocr_reader

    def extract_from_path(self, image_path: str, use_ocr: Optional[bool] = None) -> Dict[str, Any]:
        feats = {
            'brightness_mean': np.nan,
            'saturation_mean': np.nan,
            'contrast_score': np.nan,
            'face_area_ratio': 0.0,
            'text_overlay_ratio': 0.0,
            'sharpness_score': np.nan,
        }

        if not image_path or (isinstance(image_path, float) and pd.isna(image_path)):
            return feats
        if not os.path.exists(image_path):
            return feats

        img = cv2.imread(image_path)
        if img is None:
            return feats

        try:
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            h, w = gray.shape
            total_pixels = max(h * w, 1)

            feats['brightness_mean'] = float(hsv[:, :, 2].mean())
            feats['saturation_mean'] = float(hsv[:, :, 1].mean())
            feats['contrast_score'] = float(gray.std())
            feats['sharpness_score'] = float(cv2.Laplacian(gray, cv2.CV_64F).var())

            # 人脸占比（Haar）
            faces = self.face_cascade.detectMultiScale(gray, 1.1, 4)
            face_area = 0
            for (x, y, fw, fh) in faces:
                face_area += fw * fh
            feats['face_area_ratio'] = float(min(face_area / total_pixels, 1.0))

            # OCR 文字覆盖率
            should_use_ocr = self.enable_ocr if use_ocr is None else use_ocr
            if should_use_ocr:
                reader = self._get_ocr_reader()
                if reader is not None:
                    result = reader.readtext(img)  # 传 ndarray 更快
                    text_area = 0.0
                    for item in result:
                        coords = item[0]  # [[x1,y1],...]
                        xs = [p[0] for p in coords]
                        ys = [p[1] for p in coords]
                        bw = max(xs) - min(xs)
                        bh = max(ys) - min(ys)
                        if bw > 0 and bh > 0:
                            text_area += bw * bh
                    feats['text_overlay_ratio'] = float(min(text_area / total_pixels, 1.0))

        except Exception as e:
            logger.error(f"视觉特征提取失败 [{image_path}]: {e}")

        return feats

# ===========================
# 文本特征提取器
# ===========================
class TextFeatureExtractor:
    def __init__(self, registry, patterns):
        self.reg = registry                   # 词库注册中心
        self.pat = patterns                   # 结构模式（正则）

        self.emoji_pattern = re.compile(r'[\U00010000-\U0010ffff]', flags=re.UNICODE)  # 表情符号模式
        self.list_pattern = re.compile(r'^(\d+\.|-|•|Step)', re.MULTILINE)             # 列表模式

    def process(self, row: pd.Series) -> Dict[str, Any]:
        # 读取字段
        def _safe_str(x) -> str:
            return "" if (x is None or (isinstance(x, float) and pd.isna(x)) or pd.isna(x)) else str(x)
        title = _safe_str(row.get('title', ''))
        desc = _safe_str(row.get('desc', ''))
        search_kw = _safe_str(row.get('search_keyword', ''))

        raw_tags = row.get('tags', '')
        if raw_tags is None or pd.isna(raw_tags):
            tags = []
        elif isinstance(raw_tags, list):
            tags = [str(t).strip() for t in raw_tags if str(t).strip()]
        else:
            tags = [t.strip() for t in str(raw_tags).split(",") if t.strip()]


        # 拼接文本
        full_text = f"{title} {desc}".strip()
        # 分词
        words = list(jieba.cut(full_text))
        word_set = set(words)
        feats: Dict[str, Any] = {}

        # --- 共情力 ---
        feats['painpoint_word_cnt'] = self._count_hit(word_set, self.reg.pain_points)
        feats['audience_word_cnt'] = self._count_hit(word_set, self.reg.audience)
        feats['emoji_ratio'] = len(self.emoji_pattern.findall(full_text)) / (len(full_text) + 1)
        feats['title_question_flag'] = 1 if re.search(r'[?？吗呢怎么如何]', title) else 0
        feats['sentiment_intensity'] = self._calc_sentiment_intensity(full_text)

        # --- 专业力 ---
        feats['ingredient_cnt'] = self._count_hit(word_set, self.reg.ingredients)
        feats['efficacy_cnt'] = self._count_hit(word_set, self.reg.efficacy)
        feats['solution_pattern_flag'] = 1 if self._check_regex_any(desc, self.pat.usage_patterns) else 0
        feats['comparison_flag'] = 1 if self._check_regex_any(desc, self.pat.comparison_patterns) else 0

        # 信息密度： (成分+功效+垂直标签) 的命中词长总和 / 正文长度
        info_keywords = (self.reg.ingredients | self.reg.efficacy | self.reg.vertical_tags)
        keyword_len = sum(len(w) for w in word_set if w in info_keywords)
        feats['info_density_score'] = keyword_len / (len(desc) + 1)

        # --- 易读力 ---
        feats['paragraph_cnt'] = desc.count('\n') + 1 if desc else 0
        feats['list_structure_flag'] = 1 if self.list_pattern.search(desc) else 0

        sentences = re.split(r'[。！？.!?\n]', desc)
        valid = [s.strip() for s in sentences if len(s.strip()) > 1]
        feats['avg_sentence_len'] = float(np.mean([len(s) for s in valid])) if valid else 0.0
        feats['readability_score'] = 100.0 / (feats['avg_sentence_len'] + 1.0)

        # --- 搜索力 ---
        st = search_kw.strip()
        feats['title_keyword_cov'] = 1.0 if st and st in title else 0.0
        feats['content_keyword_density'] = (full_text.count(st) / (len(words) + 1)) if st else 0.0

        # 标签一致性 & 垂直标签占比
        if not tags:
            feats['vertical_tag_ratio'] = 0.0
            feats['tag_content_consistency'] = 0.0
        else:
            vertical_count = 0
            for tag in tags:
                if tag in self.reg.vertical_tags:
                    vertical_count += 1
                elif any(v in tag for v in self.reg.vertical_tags):
                    vertical_count += 1
            feats['vertical_tag_ratio'] = vertical_count / len(tags)

            hit_tags = sum(1 for t in tags if t and t in full_text)
            feats['tag_content_consistency'] = hit_tags / len(tags)

        return feats

    def _count_hit(self, text_word_set: Set[str], lex: Set[str]) -> int:
        """计算文本命中词库的次数"""
        return len(text_word_set & lex)

    def _check_regex_any(self, text: str, pattern_list: List[re.Pattern]) -> bool:
        """检查文本是否匹配任一正则模式"""
        return any(p.search(text) for p in pattern_list)

    def _calc_sentiment_intensity(self, text: str) -> float:
        """SnowNLP 计算情绪强度"""
        text = (text or "").strip()
        if not text:
            return 0.0

        p_pos = SnowNLP(text).sentiments      # ∈ [0,1]
        signed_score = 2.0 * p_pos - 1.0      # 映射到 [-1,1]
        return abs(signed_score)

# ===========================
# 主流水线
# ===========================
class AtomicFeaturePipeline:
    """原子特征提取主流水线"""
    def __init__(
        self,
        lexicon_dir: str = 'lexicons',
        checkpoint_file: str = 'data/.checkpoint_features.csv',
        checkpoint_interval: int = 50
    ):
        """初始化流水线"""
        self.registry = LexiconRegistry(base_dir=lexicon_dir)                             # 词库注册中心
        self.patterns = PatternRegistry(base_dir=lexicon_dir)                             # 结构模式（正则）
        self.text_extractor = TextFeatureExtractor(self.registry, self.patterns)          # 文本特征提取器
        self.visual_extractor = VisualFeatureExtractor(enable_ocr=True, ocr_gpu=True)                   # 视觉特征提取器
        self.checkpoint_file = checkpoint_file                                            # 断点文件路径
        self.checkpoint_interval = checkpoint_interval                                    # 断点保存间隔
        
        logger.info("✓ 特征提取流水线初始化完成")
    
    def run(self, df: pd.DataFrame, process_visual: bool = False) -> pd.DataFrame:
        """执行特征提取（支持断点续传）"""
        logger.info(f"开始处理 {len(df)} 条数据...")
        
        # 检查断点
        start_idx, results = self._load_checkpoint()
        
        # 处理数据
        for idx in tqdm(range(start_idx, len(df)), desc="提取特征", initial=start_idx, total=len(df)):
            row = df.iloc[idx]
            
            # 提取文本特征
            text_features = self.text_extractor.process(row)
            
            # 提取视觉特征
            visual_features = {}
            if process_visual:
                image_path = row.get('cover_path', '')
                visual_features = self.visual_extractor.extract_from_path(image_path)
            
            # 合并特征
            features = {**row.to_dict(), **text_features, **visual_features}
            results.append(features)
            
            # 保存断点
            if (idx + 1) % self.checkpoint_interval == 0:
                self._save_checkpoint(results)
                logger.info(f"✓ 已保存断点 ({idx + 1}/{len(df)})")
        
        # 删除断点文件
        if os.path.exists(self.checkpoint_file):
            os.remove(self.checkpoint_file)
            logger.info("✓ 断点文件已删除")
        
        logger.info(f"✓ 特征提取完成，共处理 {len(results)} 条数据")
        return pd.DataFrame(results)
    
    def _load_checkpoint(self) -> tuple:
        """加载断点"""
        if not os.path.exists(self.checkpoint_file):
            return 0, []
        
        try:
            checkpoint_df = pd.read_csv(self.checkpoint_file)
            results = checkpoint_df.to_dict('records')
            start_idx = len(results)
            logger.info(f"✓ 检测到断点文件，已处理 {start_idx} 条，从第 {start_idx + 1} 条继续...")
            return start_idx, results
        except Exception as e:
            logger.warning(f"断点文件加载失败: {e}，从头开始处理")
            return 0, []
    
    def _save_checkpoint(self, results: List[Dict]):
        """保存断点"""
        try:
            checkpoint_df = pd.DataFrame(results)
            os.makedirs(os.path.dirname(self.checkpoint_file), exist_ok=True)
            checkpoint_df.to_csv(self.checkpoint_file, index=False, encoding='utf-8-sig')
        except Exception as e:
            logger.error(f"保存断点失败: {e}")


# ===========================
# 主函数
# ===========================
def main():
    """主函数示例"""
    # 示例：加载数据
    df = pd.read_csv('data/output_annotated.csv')
    
    # 初始化流水线
    pipeline = AtomicFeaturePipeline(
        lexicon_dir='lexicons',
        checkpoint_file='data/.checkpoint_features.csv',
        checkpoint_interval=50
    )
    
    # 执行特征提取
    result_df = pipeline.run(df, process_visual=True)

    # 筛选字段与
    result_df = result_df[KEEP_COLS]
    
    # 保存结果
    output_path = 'data/data_with_features.csv'
    result_df.to_csv(output_path, index=False, encoding='utf-8-sig')
    logger.info(f"✓ 结果已保存到: {output_path}")

if __name__ == '__main__':
    main()
