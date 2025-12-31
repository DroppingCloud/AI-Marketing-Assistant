"""
旧版特征提取器 - 统一文本与视觉特征提取
====================================

功能概述:
---------
1. 文本特征提取: 从标题、正文、标签中提取45维文本特征
2. 视觉特征提取: 从图片中提取10维视觉特征
3. 批量处理: 支持从CSV读取数据，匹配图片，输出完整特征集
"""

import pandas as pd
import numpy as np
import cv2
import re
import os
import math
import json
import jieba
import jieba.posseg as pseg
from tqdm import tqdm

# ============================================================================
# 可选依赖检测
# ============================================================================

try:
    import mediapipe as mp
    HAS_MEDIAPIPE = True
except ImportError:
    HAS_MEDIAPIPE = False
    print("[INFO] MediaPipe not available, using OpenCV for face detection")


# ============================================================================
# 1. 词库注册模块
# ============================================================================

class LexiconRegistry:
    """
    词库加载与管理
    
    功能:
    -----
    - 加载美妆领域相关的各类词库（成分、功效、品类等）
    - 加载结构化模式（正则表达式）
    - 为jieba分词注册自定义词汇
    
    词库类型:
    ---------
    - 美妆知识: 成分、功效、品类、肤质
    - 内容风格: 口语、情绪、热词
    - 用户上下文: 人群、场景、痛点、预算
    - 结构模式: 号召语、用法、总结、对比等
    """
    
    def __init__(self, base_dir='lexicons'):
        """
        初始化词库
        
        参数:
        -----
        base_dir: str
            词库根目录路径
        """
        self.base_dir = base_dir
        
        # 集合类词库
        self.ingredients = set()
        self.efficacy = set()
        self.product_categories = set()
        self.skin_types = set()
        self.colloquial = set()
        self.emotions = set()
        self.hot_words = set()
        self.audiences = set()
        self.pain_points = set()
        self.scenarios = set()
        self.budget_sensitive = set()
        self.search_keywords_global = set()
        
        # 正则模式列表
        self.imperative_patterns = []
        self.budget_patterns = []
        self.usage_patterns = []
        self.summary_patterns = []
        self.comparison_patterns = []
        self.pain_solution_patterns = []
        
        # 执行加载
        self._load_all()
        
    def _load_all(self):
        """加载所有词库和模式"""
        print(f">>> 正在加载词库 (base_dir={self.base_dir})...")
        
        try:
            # 加载术语词库
            self._load_terms('beauty_knowledge/ingredients.json', self.ingredients)
            self._load_terms('beauty_knowledge/efficacy.json', self.efficacy)
            self._load_terms('beauty_knowledge/product_category.json', self.product_categories)
            self._load_terms('beauty_knowledge/skin_type.json', self.skin_types)
            self._load_terms('content_style/colloquial.json', self.colloquial)
            self._load_terms('content_style/emotion.json', self.emotions)
            self._load_terms('content_style/hotwords.json', self.hot_words)
            self._load_terms('user_context/audience.json', self.audiences)
            self._load_terms('user_context/painpoint.json', self.pain_points)
            self._load_terms('user_context/scenario.json', self.scenarios)
            self._load_terms('user_context/budget.json', self.budget_sensitive)
            
            # 加载正则模式
            self._load_simple_patterns('patterns/imperative_patterns.json', self.imperative_patterns)
            self._load_simple_patterns('user_context/budget.json', self.budget_patterns, key='price_patterns')
            self._load_structure_patterns('patterns/structure_patterns.json')
            self._load_search_keywords('search/search_keywords.json')
            
            # 注册到jieba分词器
            all_words = self.ingredients | self.efficacy | self.pain_points | self.hot_words
            for word in all_words:
                jieba.add_word(word)
                
            print(f"[✓] 词库加载完成: 成分={len(self.ingredients)}, 功效={len(self.efficacy)}")
            
        except Exception as e:
            print(f"[✗] 词库加载异常: {e}")
            import traceback
            traceback.print_exc()
    
    def _load_terms(self, rel_path, target_set):
        """加载标准格式的术语词库"""
        path = os.path.join(self.base_dir, rel_path)
        if not os.path.exists(path):
            print(f"[WARN] 文件不存在: {path}")
            return
        
        try:
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                if 'terms' in data and isinstance(data['terms'], list):
                    for item in data['terms']:
                        if 'term' in item:
                            target_set.add(item['term'])
                        if 'synonyms' in item:
                            target_set.update(item['synonyms'])
        except Exception as e:
            print(f"[WARN] 加载失败 {rel_path}: {e}")
    
    def _load_simple_patterns(self, rel_path, target_list, key='patterns'):
        """加载简单的正则模式列表"""
        path = os.path.join(self.base_dir, rel_path)
        if not os.path.exists(path):
            return
        
        try:
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                patterns = data.get(key, [])
                for p in patterns:
                    if isinstance(p, dict) and 'regex' in p:
                        target_list.append(p['regex'])
                    elif isinstance(p, str):
                        target_list.append(p)
        except Exception as e:
            print(f"[WARN] 加载模式失败 {rel_path}: {e}")
    
    def _load_structure_patterns(self, rel_path):
        """加载结构化模式（复杂结构）"""
        path = os.path.join(self.base_dir, rel_path)
        if not os.path.exists(path):
            return
        
        try:
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                root = data.get('patterns', {})
                
                def extract_regex(group_name):
                    result = []
                    items = root.get(group_name, [])
                    for item in items:
                        if 'regex' in item:
                            result.append(item['regex'])
                    return result
                
                self.usage_patterns = extract_regex('usage_method')
                self.summary_patterns = extract_regex('summary')
                self.comparison_patterns = extract_regex('comparison')
                self.pain_solution_patterns = extract_regex('painpoint_solution_effect')
                
        except Exception as e:
            print(f"[WARN] 加载结构模式失败: {e}")
    
    def _load_search_keywords(self, rel_path):
        """加载检索关键词"""
        path = os.path.join(self.base_dir, rel_path)
        if not os.path.exists(path):
            return
        
        try:
            with open(path, 'r', encoding='utf-8') as f:
                content = json.load(f)
                groups = content.get('groups', {})
                for group_val in groups.values():
                    if 'terms' in group_val:
                        for t in group_val['terms']:
                            self.search_keywords_global.add(t['term'])
                            if 'synonyms' in t:
                                self.search_keywords_global.update(t['synonyms'])
        except Exception as e:
            print(f"[WARN] 加载检索词失败: {e}")


# ============================================================================
# 2. 文本特征提取模块
# ============================================================================

class TextFeatureExtractor:
    """
    文本特征提取器
    
    功能:
    -----
    从笔记的标题、正文、标签中提取45维文本特征，包括：
    - 标题特征 (7维): 长度、数字、疑问句、关键词覆盖等
    - 正文结构 (9维): 长度、句数、段落、列表、总结等
    - 语义特征 (24维): 热词、口语、成分、功效、人群等
    - 标签特征 (5维): 数量、一致性、垂直标签等
    """
    
    def __init__(self, lexicons):
        """
        初始化提取器
        
        参数:
        -----
        lexicons: LexiconRegistry
            已加载的词库实例
        """
        self.lex = lexicons
    
    # ------------------------------------------------------------------------
    # 工具方法
    # ------------------------------------------------------------------------
    
    def _count_hits(self, text, lexicon):
        """计算文本命中词库的次数（含重复）"""
        return sum(1 for word in lexicon if word in text) if text else 0
    
    def _count_unique_hits(self, text, lexicon):
        """计算文本命中词库的不重复词数"""
        return len(set(word for word in lexicon if word in text)) if text else 0
    
    def _check_regex(self, text, patterns):
        """检查文本是否匹配任一正则模式"""
        if not text:
            return 0
        for pat in patterns:
            try:
                if re.search(pat, text, re.IGNORECASE):
                    return 1
            except re.error:
                continue
        return 0
    
    # ------------------------------------------------------------------------
    # 特征提取模块
    # ------------------------------------------------------------------------
    
    def extract_title_features(self, row):
        """
        提取标题特征 (7维)
        
        特征列表:
        ---------
        0. title_len: 标题长度
        1. title_number_flag: 是否包含数字
        2. title_question_flag: 是否为疑问句
        3. title_keyword_cov: 核心关键词覆盖率
        4. title_keyword_cnt: 关键词命中数量
        5. title_keyword_pos_score: 关键词位置得分
        6. title_readability_score: 可读性得分
        """
        title = str(row.get('title', ''))
        search_kw = str(row.get('search_keyword', ''))
        
        feats = {}
        feats['title_len'] = len(title)
        feats['title_number_flag'] = 1 if re.search(r'\d|[一二三四五六七八九十]', title) else 0
        feats['title_question_flag'] = 1 if re.search(r'[?？]|怎么|如何|好用吗|什么|避雷吗', title) else 0
        feats['title_keyword_cov'] = 1 if (search_kw and search_kw in title) else 0
        feats['title_keyword_cnt'] = self._count_hits(title, self.lex.search_keywords_global)
        
        # 关键词位置得分
        if search_kw and search_kw in title:
            pos = title.find(search_kw)
            feats['title_keyword_pos_score'] = 1 - (pos / len(title))
        else:
            feats['title_keyword_pos_score'] = 0
        
        # 可读性: 符号占比越低越好
        symbol_cnt = len(re.findall(r'[^\w\s]', title))
        feats['title_readability_score'] = 1 - (symbol_cnt / (len(title) + 1))
        
        return feats
    
    def extract_content_features(self, row):
        """
        提取正文结构特征 (9维)
        
        特征列表:
        ---------
        7. content_len: 正文长度
        8. sentence_cnt: 句子数
        9. avg_sentence_len: 平均句长
        10. paragraph_cnt: 段落数
        11. list_structure_flag: 是否有列表结构
        12. summary_flag: 是否有总结段落
        13. info_density_score: 信息密度
        14. readability_score: 可读性
        15. solution_pattern_flag: 痛点-方案-效果结构
        """
        desc = str(row.get('desc', ''))
        feats = {}
        
        feats['content_len'] = len(desc)
        
        # 句子切分
        sentences = [s for s in re.split(r'[。！？.!?\n]', desc) if len(s.strip()) > 1]
        feats['sentence_cnt'] = len(sentences)
        feats['avg_sentence_len'] = np.mean([len(s) for s in sentences]) if sentences else 0
        
        # 段落和结构
        feats['paragraph_cnt'] = desc.count('\n') + 1
        list_matches = re.findall(r'(\d\.|[abcd]\.|•|✔|✅|👉|①|②)', desc)
        feats['list_structure_flag'] = 1 if len(list_matches) >= 3 else 0
        
        # 总结段落
        if self.lex.summary_patterns:
            feats['summary_flag'] = self._check_regex(desc, self.lex.summary_patterns)
        else:
            feats['summary_flag'] = self._check_regex(desc, [r'(总结|综上|结论|最后|总的来说)'])
        
        # 信息密度（实词比例）
        try:
            words = list(pseg.cut(desc))
            content_words = [w for w, flag in words if flag.startswith(('n', 'v', 'a'))]
            feats['info_density_score'] = len(content_words) / (len(words) + 1)
        except:
            feats['info_density_score'] = 0
        
        # 可读性
        avg_len = feats['avg_sentence_len']
        feats['readability_score'] = max(0, min(1, 1 - (avg_len - 5) / 45))
        
        # 痛点-方案-效果链
        pat_flag = self._check_regex(desc, self.lex.pain_solution_patterns)
        if pat_flag:
            feats['solution_pattern_flag'] = 1
        else:
            has_pain = 1 if self._count_hits(desc, self.lex.pain_points) > 0 else 0
            has_eff = 1 if self._count_hits(desc, self.lex.efficacy) > 0 else 0
            feats['solution_pattern_flag'] = 1 if (has_pain and has_eff) else 0
        
        return feats
    
    def extract_semantic_features(self, row):
        """
        提取语义特征 (24维)
        
        特征列表:
        ---------
        16-19: 热词、口语、Emoji、标点
        20-21: 感叹号、问号占比
        22-24: 第二人称、祈使句、情绪强度
        25-28: 成分词、功效词（数量+多样性）
        29-30: 肤质词、品类词
        31-32: 用法/步骤、对比信息
        33-35: 人群、场景、痛点词
        36: 价格敏感度
        37-39: 检索词覆盖、密度
        """
        full_text = str(row.get('title', '')) + " " + str(row.get('desc', ''))
        feats = {}
        
        # 内容风格
        feats['hotword_hit_rate'] = self._count_hits(full_text, self.lex.hot_words) / (len(full_text) / 100 + 1)
        feats['colloquial_ratio'] = self._count_hits(full_text, self.lex.colloquial) / (len(full_text) / 100 + 1)
        emoji_cnt = len(re.findall(r'\[.*?\]', full_text))
        feats['emoji_ratio'] = emoji_cnt / (len(full_text) + 1)
        
        # 标点符号
        punct_cnt = len(re.findall(r'[，。！？、,!?]', full_text))
        feats['punctuation_density'] = punct_cnt / (len(full_text) + 1)
        feats['exclamation_ratio'] = (full_text.count('!') + full_text.count('！')) / (len(full_text) + 1)
        feats['question_ratio'] = (full_text.count('?') + full_text.count('？')) / (len(full_text) + 1)
        
        # 互动风格
        sec_person_words = ['你', '你们', '姐妹', '宝宝', '大家', '集美']
        feats['second_person_ratio'] = sum(full_text.count(w) for w in sec_person_words) / (len(full_text) / 100 + 1)
        feats['imperative_ratio'] = self._check_regex(full_text, self.lex.imperative_patterns)
        feats['sentiment_intensity'] = feats['exclamation_ratio'] * 100 + self._count_hits(full_text, self.lex.emotions)
        
        # 美妆专业知识
        feats['ingredient_cnt'] = self._count_hits(full_text, self.lex.ingredients)
        feats['ingredient_diversity'] = self._count_unique_hits(full_text, self.lex.ingredients)
        feats['efficacy_cnt'] = self._count_hits(full_text, self.lex.efficacy)
        feats['efficacy_diversity'] = self._count_unique_hits(full_text, self.lex.efficacy)
        feats['skin_type_cnt'] = self._count_hits(full_text, self.lex.skin_types)
        feats['product_category_cnt'] = self._count_hits(full_text, self.lex.product_categories)
        
        # 内容结构
        feats['usage_method_flag'] = self._check_regex(full_text, self.lex.usage_patterns)
        if self.lex.comparison_patterns:
            feats['comparison_flag'] = self._check_regex(full_text, self.lex.comparison_patterns)
        else:
            feats['comparison_flag'] = 1 if re.search(r'(对比|区别|PK|pk|胜出|前后|变化)', full_text) else 0
        
        # 用户定位
        feats['audience_word_cnt'] = self._count_hits(full_text, self.lex.audiences)
        feats['scenario_word_cnt'] = self._count_hits(full_text, self.lex.scenarios)
        feats['painpoint_word_cnt'] = self._count_hits(full_text, self.lex.pain_points)
        
        # 价格敏感
        feats['budget_sensitivity_flag'] = 1 if (
            self._count_hits(full_text, self.lex.budget_sensitive) > 0 or
            self._check_regex(full_text, self.lex.budget_patterns)
        ) else 0
        
        # 检索词相关
        feats['search_keyword_cov'] = 1 if str(row.get('search_keyword')) in full_text else 0
        feats['search_keyword_cnt'] = self._count_hits(full_text, self.lex.search_keywords_global)
        feats['keyword_density'] = feats['search_keyword_cnt'] / (len(full_text) + 1)
        
        return feats
    
    def extract_tag_features(self, row):
        """
        提取标签特征 (5维)
        
        特征列表:
        ---------
        40. tag_content_consistency: 标签与正文一致性
        41. tag_cnt: 标签数量
        42. vertical_tag_ratio: 垂直标签占比
        43. generic_tag_ratio: 泛标签占比
        44. tag_keyword_hit_cnt: 标签命中关键词数
        """
        tags_str = str(row.get('tags', ''))
        tags = [t.strip().replace('#', '').replace('[话题]', '') 
                for t in re.split(r'[,，\s]', tags_str) if t.strip()]
        
        feats = {}
        
        # 标签与正文一致性
        desc = str(row.get('desc', ''))
        overlap = sum(1 for t in tags if t in desc)
        feats['tag_content_consistency'] = overlap / len(tags) if tags else 0
        
        # 标签统计
        feats['tag_cnt'] = len(tags)
        feats['vertical_tag_ratio'] = 0  # 词库缺失，保留接口
        feats['generic_tag_ratio'] = 0   # 词库缺失，保留接口
        feats['tag_keyword_hit_cnt'] = sum(1 for t in tags if t in self.lex.search_keywords_global)
        
        return feats
    
    def extract(self, row):
        """
        提取单行的所有文本特征
        
        参数:
        -----
        row: pd.Series
            包含 title, desc, tags, search_keyword 的数据行
        
        返回:
        -----
        dict: 包含45维文本特征的字典
        """
        features = {}
        features.update(self.extract_title_features(row))
        features.update(self.extract_content_features(row))
        features.update(self.extract_semantic_features(row))
        features.update(self.extract_tag_features(row))
        return features


# ============================================================================
# 3. 视觉特征提取模块
# ============================================================================

class VisualFeatureExtractor:
    """
    视觉特征提取器
    
    功能:
    -----
    从笔记封面图片中提取10维视觉特征，包括：
    - 光影色彩 (4维): 亮度、饱和度、对比度、色彩丰富度
    - 质量风格 (3维): 清晰度、熵、视觉复杂度
    - 人像主体 (3维): 是否有人、人脸占比、人脸数量
    
    技术方案:
    ---------
    - 人脸检测: MediaPipe (优先) > OpenCV Haar Cascade (备选)
    - 色彩分析: Hasler & Süsstrunk 算法
    - 清晰度: Laplacian 方差
    """
    
    def __init__(self, use_mediapipe=True):
        """
        初始化提取器
        
        参数:
        -----
        use_mediapipe: bool
            是否尝试使用MediaPipe（更准确但可能不稳定）
        """
        self.mp_face_detection = None
        self.face_cascade = None
        
        # 尝试初始化 MediaPipe
        if HAS_MEDIAPIPE and use_mediapipe:
            try:
                self.mp_face_detection = mp.solutions.face_detection.FaceDetection(
                    model_selection=1, min_detection_confidence=0.5
                )
            except Exception as e:
                print(f"[WARN] MediaPipe初始化失败: {e}")
                self.mp_face_detection = None
        
        # 备选: OpenCV Haar Cascade
        if not self.mp_face_detection:
            cascade_path = "/opt/anaconda3/envs/ML/share/opencv4/haarcascades/haarcascade_frontalface_default.xml"
            if not os.path.exists(cascade_path):
                cascade_path = os.path.join(cv2.data.haarcascades, 'haarcascade_frontalface_default.xml')
            
            if os.path.exists(cascade_path):
                self.face_cascade = cv2.CascadeClassifier(cascade_path)
            else:
                print(f"[WARN] 人脸检测模型未找到: {cascade_path}")
    
    def extract(self, image_path):
        """
        提取单张图片的视觉特征
        
        参数:
        -----
        image_path: str
            图片文件路径
        
        返回:
        -----
        dict: 包含10维视觉特征的字典
        """
        # 特征默认值
        features = {
            'brightness_mean': 0.0,
            'saturation_mean': 0.0,
            'contrast_score': 0.0,
            'colorfulness_score': 0.0,
            'sharpness_score': 0.0,
            'entropy_score': 0.0,
            'visual_complexity': 0.0,
            'human_present': 0,
            'face_area_ratio': 0.0,
            'face_count': 0
        }
        
        # 检查文件存在性
        if not os.path.exists(image_path):
            return features
        
        # 读取图片
        img = cv2.imread(image_path)
        if img is None:
            return features
        
        try:
            height, width = img.shape[:2]
            total_pixels = height * width
            
            # 预处理
            gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            h, s, v = cv2.split(hsv_img)
            
            # ----------------------------------------------------------------
            # 模块A: 光影色彩特征
            # ----------------------------------------------------------------
            
            features['brightness_mean'] = np.mean(v) / 255.0
            features['saturation_mean'] = np.mean(s) / 255.0
            features['contrast_score'] = np.std(gray_img) / 128.0
            
            # 色彩丰富度 (Hasler & Süsstrunk 算法)
            B, G, R = cv2.split(img.astype("float"))
            rg = np.absolute(R - G)
            yb = np.absolute(0.5 * (R + G) - B)
            std_root = np.sqrt((np.std(rg) ** 2) + (np.std(yb) ** 2))
            mean_root = np.sqrt((np.mean(rg) ** 2) + (np.mean(yb) ** 2))
            features['colorfulness_score'] = std_root + (0.3 * mean_root)
            
            # ----------------------------------------------------------------
            # 模块B: 质量与复杂度特征
            # ----------------------------------------------------------------
            
            # 清晰度 (Laplacian方差)
            laplacian_var = cv2.Laplacian(gray_img, cv2.CV_64F).var()
            features['sharpness_score'] = math.log(laplacian_var + 1)
            
            # 视觉复杂度 (边缘密度)
            edges = cv2.Canny(gray_img, 100, 200)
            features['visual_complexity'] = np.count_nonzero(edges) / total_pixels
            
            # 图像熵
            hist = cv2.calcHist([gray_img], [0], None, [256], [0, 256])
            hist_norm = hist.ravel() / hist.sum()
            logs = np.log2(hist_norm + 1e-7)
            features['entropy_score'] = -1 * (hist_norm * logs).sum()
            
            # ----------------------------------------------------------------
            # 模块C: 人像主体特征
            # ----------------------------------------------------------------
            
            face_area = 0.0
            face_count = 0
            
            if HAS_MEDIAPIPE and self.mp_face_detection:
                # MediaPipe 检测
                rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                results = self.mp_face_detection.process(rgb_img)
                
                if results.detections:
                    face_count = len(results.detections)
                    for detection in results.detections:
                        bboxC = detection.location_data.relative_bounding_box
                        face_area += (bboxC.width * bboxC.height)
            
            elif self.face_cascade:
                # OpenCV Haar Cascade 检测
                faces = self.face_cascade.detectMultiScale(
                    gray_img, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
                )
                face_count = len(faces)
                for (x, y, w, h) in faces:
                    face_area += (w * h) / total_pixels
            
            features['human_present'] = 1 if face_count > 0 else 0
            features['face_count'] = face_count
            features['face_area_ratio'] = min(face_area, 1.0)
        
        except Exception as e:
            print(f"[WARN] 图片处理失败 {image_path}: {e}")
        
        return features


# ============================================================================
# 4. 批量处理协调器
# ============================================================================

class BatchProcessor:
    """
    批量特征提取协调器
    
    功能:
    -----
    1. 读取CSV数据
    2. 加载词库
    3. 批量提取文本特征
    4. 查找对应图片并提取视觉特征
    5. 合并所有特征并保存
    """
    
    def __init__(self, 
                 input_csv='data/data_with_label.csv',
                 output_csv='data/data_with_full_features.csv',
                 image_dir='image',
                 lexicon_dir='lexicons',
                 use_mediapipe=False):
        """
        初始化批处理器
        
        参数:
        -----
        input_csv: str
            输入CSV路径
        output_csv: str
            输出CSV路径
        image_dir: str
            图片目录路径
        lexicon_dir: str
            词库目录路径
        use_mediapipe: bool
            视觉提取是否使用MediaPipe
        """
        self.input_csv = input_csv
        self.output_csv = output_csv
        self.image_dir = image_dir
        self.lexicon_dir = lexicon_dir
        self.use_mediapipe = use_mediapipe
        
        # 兼容从src目录运行
        self._adjust_paths()
        
        # 提取器实例（延迟初始化）
        self.lexicons = None
        self.text_extractor = None
        self.visual_extractor = None
        self.image_map = {}
    
    def _adjust_paths(self):
        """自动调整路径（兼容从src/目录运行）"""
        if not os.path.exists(self.input_csv):
            self.input_csv = os.path.join('..', self.input_csv)
        if not os.path.exists(self.image_dir):
            self.image_dir = os.path.join('..', self.image_dir)
        if not os.path.exists(self.lexicon_dir):
            self.lexicon_dir = os.path.join('..', self.lexicon_dir)
    
    def _build_image_index(self):
        """构建 note_id -> image_path 的映射"""
        print(f"\n>>> 扫描图片目录: {self.image_dir}")
        
        for root, _, files in os.walk(self.image_dir):
            for file in files:
                if file.lower().endswith(('.jpg', '.jpeg', '.png', '.webp')):
                    note_id = os.path.splitext(file)[0]
                    self.image_map[note_id] = os.path.join(root, file)
        
        print(f"[✓] 找到 {len(self.image_map)} 张图片")
    
    def run(self):
        """执行完整的批量提取流程"""
        print("=" * 70)
        print("批量特征提取器 - 文本+视觉")
        print("=" * 70)
        
        # ====================================================================
        # Step 1: 数据加载
        # ====================================================================
        print(f"\n[1/5] 读取数据: {self.input_csv}")
        if not os.path.exists(self.input_csv):
            print(f"[✗] 文件不存在: {self.input_csv}")
            return
        
        df = pd.read_csv(self.input_csv)
        print(f"[✓] 数据行数: {len(df)}")
        
        # ====================================================================
        # Step 2: 初始化词库和提取器
        # ====================================================================
        print(f"\n[2/5] 初始化词库和提取器")
        self.lexicons = LexiconRegistry(base_dir=self.lexicon_dir)
        self.text_extractor = TextFeatureExtractor(self.lexicons)
        self.visual_extractor = VisualFeatureExtractor(use_mediapipe=self.use_mediapipe)
        print("[✓] 提取器就绪")
        
        # ====================================================================
        # Step 3: 文本特征提取
        # ====================================================================
        print(f"\n[3/5] 提取文本特征 (45维)")
        text_features = []
        
        for idx, row in tqdm(df.iterrows(), total=len(df), desc="文本特征"):
            feats = self.text_extractor.extract(row)
            text_features.append(feats)
        
        text_feat_df = pd.DataFrame(text_features)
        print(f"[✓] 文本特征提取完成，维度: {text_feat_df.shape[1]}")
        
        # ====================================================================
        # Step 4: 视觉特征提取
        # ====================================================================
        print(f"\n[4/5] 提取视觉特征 (10维)")
        self._build_image_index()
        
        visual_features = []
        success_count = 0
        missing_count = 0
        
        for idx, row in tqdm(df.iterrows(), total=len(df), desc="视觉特征"):
            note_id = str(row['note_id'])
            
            if note_id in self.image_map:
                img_path = self.image_map[note_id]
                try:
                    feats = self.visual_extractor.extract(img_path)
                    visual_features.append(feats)
                    success_count += 1
                except Exception as e:
                    # 出错时使用默认值
                    visual_features.append(self.visual_extractor.extract(""))
            else:
                # 未找到图片时使用默认值
                visual_features.append(self.visual_extractor.extract(""))
                missing_count += 1
        
        visual_feat_df = pd.DataFrame(visual_features)
        print(f"[✓] 视觉特征提取完成")
        print(f"    成功: {success_count} 张 | 缺失: {missing_count} 张")
        
        # ====================================================================
        # Step 5: 合并并保存
        # ====================================================================
        print(f"\n[5/5] 合并特征并保存")
        
        # 保留元信息列
        meta_cols = ['note_id', 'title', 'hot_level', 'search_keyword']
        meta_df = df[meta_cols]
        
        # 合并所有特征
        result_df = pd.concat([meta_df, text_feat_df, visual_feat_df], axis=1)
        
        # 保存
        result_df.to_csv(self.output_csv, index=False, encoding='utf-8-sig')
        
        print(f"[✓] 保存完成: {self.output_csv}")
        print(f"[✓] 最终维度: {result_df.shape} (行×列)")
        print(f"[✓] 特征列数: {len(text_feat_df.columns) + len(visual_feat_df.columns)}")
        
        print("\n" + "=" * 70)
        print("✨ 批量提取完成！")
        print("=" * 70)


# ============================================================================
# 5. 主函数
# ============================================================================

def main():
    """主入口函数"""
    processor = BatchProcessor(
        input_csv='data/data_with_label.csv',
        output_csv='data/data_with_full_features.csv',
        image_dir='image',
        lexicon_dir='lexicons',
        use_mediapipe=False  # 建议关闭以提高稳定性
    )
    processor.run()


if __name__ == "__main__":
    main()

