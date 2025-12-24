"""
文本特征提取器：从原始数据中提取出模型可用的文本特征
"""

import pandas as pd
import numpy as np
import re
import jieba
import jieba.posseg as pseg
import json
import os

# ================= 词库注册 (LexiconRegistry) =================
class LexiconRegistry:
    """
    负责加载 README.md 中描述的 JSON 格式词库
    """
    def __init__(self, base_dir='lexicons'):
        self.base_dir = base_dir
        
        # --- 集合类词库 (Set) ---
        self.ingredients = set()       # 成分
        self.efficacy = set()          # 功效
        self.product_categories = set()# 品类
        self.skin_types = set()        # 肤质
        self.colloquial = set()        # 口语
        self.emotions = set()          # 情绪
        self.hot_words = set()         # 热词
        self.audiences = set()         # 人群
        self.pain_points = set()       # 痛点
        self.scenarios = set()         # 场景
        self.budget_sensitive = set()  # 预算敏感
        self.vertical_tags = set()     # 垂直标签 (缺文件)
        self.generic_tags = set()      # 泛标签 (缺文件)
        self.search_keywords_global = set() # 全局检索词库
        
        # --- 正则模式类 (List) ---
        self.imperative_patterns = []  # 号召
        self.budget_patterns = []      # 价格正则
        
        # --- 结构化模式 (来自 structure_patterns.json) ---
        self.usage_patterns = []       # 用法/步骤
        self.summary_patterns = []     # 总结
        self.comparison_patterns = []  # 对比
        self.pain_solution_patterns = [] # 痛点方案链
        
        # --- 加载过程 (需确保文件存在) ---
        print(f">>> 正在加载词库 (base_dir={self.base_dir})...")
        try:
            # 1. 加载各类 Term (Standard JSON)
            self._load_terms('beauty_knowledge/ingredients.json', self.ingredients)
            self._load_terms('beauty_knowledge/efficacy.json', self.efficacy)
            self._load_terms('beauty_knowledge/product_category.json', self.product_categories)
            self._load_terms('beauty_knowledge/skin_type.json', self.skin_types)
            self._load_terms('content_style/colloquial.json', self.colloquial)
            self._load_terms('content_style/emotion.json', self.emotions) # 注意文件名是 emotion.json
            self._load_terms('content_style/hotwords.json', self.hot_words)
            self._load_terms('user_context/audience.json', self.audiences)
            self._load_terms('user_context/painpoint.json', self.pain_points)
            self._load_terms('user_context/scenario.json', self.scenarios)
            self._load_terms('user_context/budget.json', self.budget_sensitive)
            
            # 2. 加载正则 Patterns
            self._load_simple_patterns('patterns/imperative_patterns.json', self.imperative_patterns)
            self._load_simple_patterns('user_context/budget.json', self.budget_patterns, key='price_patterns')
            
            # 3. 加载复杂的结构化 Patterns
            self._load_structure_patterns('patterns/structure_patterns.json')

            # 4. 加载全局检索词 (search_keywords.json 包含 groups)
            self._load_search_keywords('search/search_keywords.json')
            
            # 5. Jieba 初始化
            all_words = self.ingredients | self.efficacy | self.pain_points | self.hot_words
            for w in all_words:
                jieba.add_word(w)
                
            print(f"词库加载完毕。成分词: {len(self.ingredients)}, 功效词: {len(self.efficacy)}")
            
        except Exception as e:
            print(f"Error loading lexicons: {e}")
            import traceback
            traceback.print_exc()

    def _load_terms(self, rel_path, target_set):
        path = os.path.join(self.base_dir, rel_path)
        if not os.path.exists(path): 
            print(f"Warning: File not found {path}")
            return
        try:
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                # 假设标准格式: { "terms": [ {"term":...}, ... ] }
                if 'terms' in data and isinstance(data['terms'], list):
                    for item in data['terms']:
                        if 'term' in item: target_set.add(item['term'])
                        if 'synonyms' in item: target_set.update(item['synonyms'])
        except Exception as e:
            print(f"Failed to load terms from {rel_path}: {e}")

    def _load_simple_patterns(self, rel_path, target_list, key='patterns'):
        """加载简单的正则列表，如 imperative_patterns"""
        path = os.path.join(self.base_dir, rel_path)
        if not os.path.exists(path): return
        try:
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                # 可能是 {"patterns": [...]} 或 {"price_patterns": [...]}
                patterns = data.get(key, [])
                for p in patterns:
                    if isinstance(p, dict) and 'regex' in p:
                        target_list.append(p['regex'])
                    elif isinstance(p, str): # 兼容纯字符串列表
                        target_list.append(p)
        except Exception as e:
            print(f"Failed to load patterns from {rel_path}: {e}")

    def _load_structure_patterns(self, rel_path):
        """专门加载 structure_patterns.json 的复杂结构"""
        path = os.path.join(self.base_dir, rel_path)
        if not os.path.exists(path): return
        try:
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                # data['patterns'] 是一个字典，包含 usage_method, summary, comparison 等
                root = data.get('patterns', {})
                
                # Helper to extract regex list from a group
                def extract(group_name):
                    res = []
                    items = root.get(group_name, [])
                    for item in items:
                        if 'regex' in item: res.append(item['regex'])
                    return res

                self.usage_patterns = extract('usage_method')
                self.summary_patterns = extract('summary')
                self.comparison_patterns = extract('comparison')
                self.pain_solution_patterns = extract('painpoint_solution_effect')
                
        except Exception as e:
            print(f"Failed to load structure patterns: {e}")

    def _load_search_keywords(self, rel_path):
        """加载 search_keywords.json (带 groups 结构)"""
        path = os.path.join(self.base_dir, rel_path)
        if not os.path.exists(path): return
        try:
            with open(path, 'r', encoding='utf-8') as f:
                content = json.load(f)
                groups = content.get('groups', {})
                for group_key, group_val in groups.items():
                    if 'terms' in group_val:
                        for t in group_val['terms']:
                            self.search_keywords_global.add(t['term'])
                            if 'synonyms' in t: self.search_keywords_global.update(t['synonyms'])
        except Exception as e:
            print(f"Failed to load search keywords: {e}")

# ================= 特征提取器 (FeatureExtractor) =================
class FeatureExtractor:
    def __init__(self, lexicons):
        self.lex = lexicons

    # --- 通用工具函数 ---
    def _count_hits(self, text, lexicon):
        return sum(1 for word in lexicon if word in text) if text else 0

    def _count_unique_hits(self, text, lexicon):
        return len(set(word for word in lexicon if word in text)) if text else 0

    def _check_regex(self, text, patterns):
        if not text: return 0
        for pat in patterns:
            try:
                if re.search(pat, text, re.IGNORECASE): return 1
            except re.error:
                continue
        return 0

    # ================= 模块 A: 标题特征 (Row 0-6) =================
    def extract_title_features(self, row):
        title = str(row.get('title', ''))
        search_kw = str(row.get('search_keyword', ''))
        
        feats = {}
        # [Def] Row 0: 标题长度（字符数）
        feats['title_len'] = len(title)
        
        # [Def] Row 1: 标题是否包含数字
        feats['title_number_flag'] = 1 if re.search(r'\d|[一二三四五六七八九十]', title) else 0
        
        # [Def] Row 2: 标题是否为疑问句
        feats['title_question_flag'] = 1 if re.search(r'[?？]|怎么|如何|好用吗|什么|避雷吗', title) else 0
        
        # [Def] Row 3: 标题命中核心检索词覆盖率 (0 或 1)
        feats['title_keyword_cov'] = 1 if (search_kw and search_kw in title) else 0
        
        # [Def] Row 4: 标题命中核心检索词数量 (全库范围)
        feats['title_keyword_cnt'] = self._count_hits(title, self.lex.search_keywords_global)
        
        # [Def] Row 5: 核心检索词在标题中的靠前程度 (位置归一化: 1最前, 0最后)
        if search_kw and search_kw in title:
            pos = title.find(search_kw)
            feats['title_keyword_pos_score'] = 1 - (pos / len(title))
        else:
            feats['title_keyword_pos_score'] = 0
            
        # [Def] Row 6: 标题可读性 (简化：符号占比越低，可读性越高)
        symbol_cnt = len(re.findall(r'[^\w\s]', title))
        feats['title_readability_score'] = 1 - (symbol_cnt / (len(title) + 1))
        
        return feats

    # ================= 模块 B: 正文结构特征 (Row 7-15) =================
    def extract_content_features(self, row):
        desc = str(row.get('desc', ''))
        feats = {}
        
        # [Def] Row 7: 正文长度
        feats['content_len'] = len(desc)
        
        # [Def] Row 8: 正文句子数 (按标点切分)
        sentences = re.split(r'[。！？.!?\n]', desc)
        sentences = [s for s in sentences if len(s.strip()) > 1] # 过滤空句
        feats['sentence_cnt'] = len(sentences)
        
        # [Def] Row 9: 平均句长
        feats['avg_sentence_len'] = np.mean([len(s) for s in sentences]) if sentences else 0
        
        # [Def] Row 10: 正文段落数 (按换行切分)
        feats['paragraph_cnt'] = desc.count('\n') + 1
        
        # [Def] Row 11: 是否呈现列表结构 (正则检测 1. 2. 或 emoji列表)
        list_pat = r'(\d\.|[abcd]\.|•|✔|✅|👉|①|②)'
        matches = re.findall(list_pat, desc)
        feats['list_structure_flag'] = 1 if len(matches) >= 3 else 0
        
        # [Def] Row 12: 是否包含总结段落 (使用 structure_patterns)
        # 如果 patterns 为空，回退到硬编码
        if self.lex.summary_patterns:
            feats['summary_flag'] = self._check_regex(desc, self.lex.summary_patterns)
        else:
            feats['summary_flag'] = self._check_regex(desc, [r'(总结|综上|结论|最后|总的来说)'])
        
        # [Def] Row 13: 信息密度 (实词/总词数)
        try:
            words = list(pseg.cut(desc))
            content_words = [w for w, flag in words if flag.startswith(('n', 'v', 'a'))]
            feats['info_density_score'] = len(content_words) / (len(words) + 1)
        except:
            feats['info_density_score'] = 0
            
        # [Def] Row 14: 正文可读性
        avg_len = feats['avg_sentence_len']
        feats['readability_score'] = max(0, min(1, 1 - (avg_len - 5) / 45))
        
        # [Def] Row 15: 痛点→方案→效果结构
        # 优先使用 patterns，如果没命中再用词库共现兜底
        pat_flag = self._check_regex(desc, self.lex.pain_solution_patterns)
        if pat_flag:
            feats['solution_pattern_flag'] = 1
        else:
            has_pain = 1 if self._count_hits(desc, self.lex.pain_points) > 0 else 0
            has_eff = 1 if self._count_hits(desc, self.lex.efficacy) > 0 else 0
            feats['solution_pattern_flag'] = 1 if (has_pain and has_eff) else 0
        
        return feats

    # ================= 模块 C: 语义特征 (Row 16-39) =================
    def extract_semantic_features(self, row):
        full_text = str(row.get('title', '')) + " " + str(row.get('desc', ''))
        feats = {}
        
        # [Def] Row 16: 热词命中率
        hits = self._count_hits(full_text, self.lex.hot_words)
        feats['hotword_hit_rate'] = hits / (len(full_text) / 100 + 1)
        
        # [Def] Row 17: 口语化比例
        feats['colloquial_ratio'] = self._count_hits(full_text, self.lex.colloquial) / (len(full_text)/100 + 1)
        
        # [Def] Row 18: Emoji 占比
        emoji_cnt = len(re.findall(r'\[.*?\]', full_text)) 
        feats['emoji_ratio'] = emoji_cnt / (len(full_text) + 1)
        
        # [Def] Row 19: 标点密度
        punct_cnt = len(re.findall(r'[，。！？、,!?]', full_text))
        feats['punctuation_density'] = punct_cnt / (len(full_text) + 1)
        
        # [Def] Row 20: 感叹号占比
        feats['exclamation_ratio'] = (full_text.count('!') + full_text.count('！')) / (len(full_text) + 1)
        
        # [Def] Row 21: 问号占比
        feats['question_ratio'] = (full_text.count('?') + full_text.count('？')) / (len(full_text) + 1)
        
        # [Def] Row 22: 第二人称占比
        sec_person_words = ['你', '你们', '姐妹', '宝宝', '大家', '集美']
        feats['second_person_ratio'] = sum(full_text.count(w) for w in sec_person_words) / (len(full_text)/100 + 1)
        
        # [Def] Row 23: 祈使/号召表达
        feats['imperative_ratio'] = self._check_regex(full_text, self.lex.imperative_patterns)
        
        # [Def] Row 24: 情绪强度
        feats['sentiment_intensity'] = feats['exclamation_ratio'] * 100 + self._count_hits(full_text, self.lex.emotions)
        
        # [Def] Row 25-26: 成分词
        feats['ingredient_cnt'] = self._count_hits(full_text, self.lex.ingredients)
        feats['ingredient_diversity'] = self._count_unique_hits(full_text, self.lex.ingredients)
        
        # [Def] Row 27-28: 功效词
        feats['efficacy_cnt'] = self._count_hits(full_text, self.lex.efficacy)
        feats['efficacy_diversity'] = self._count_unique_hits(full_text, self.lex.efficacy)
        
        # [Def] Row 29: 肤质词
        feats['skin_type_cnt'] = self._count_hits(full_text, self.lex.skin_types)
        
        # [Def] Row 30: 品类词
        feats['product_category_cnt'] = self._count_hits(full_text, self.lex.product_categories)
        
        # [Def] Row 31: 用法/步骤信息 (使用结构化 Patterns)
        feats['usage_method_flag'] = self._check_regex(full_text, self.lex.usage_patterns)
        
        # [Def] Row 32: 对比/前后效果 (使用结构化 Patterns)
        if self.lex.comparison_patterns:
            feats['comparison_flag'] = self._check_regex(full_text, self.lex.comparison_patterns)
        else:
            feats['comparison_flag'] = 1 if re.search(r'(对比|区别|PK|pk|胜出|前后|变化)', full_text) else 0
        
        # [Def] Row 33-35: 人群/场景/痛点
        feats['audience_word_cnt'] = self._count_hits(full_text, self.lex.audiences)
        feats['scenario_word_cnt'] = self._count_hits(full_text, self.lex.scenarios)
        feats['painpoint_word_cnt'] = self._count_hits(full_text, self.lex.pain_points)
        
        # [Def] Row 36: 价格敏感
        feats['budget_sensitivity_flag'] = 1 if (
            self._count_hits(full_text, self.lex.budget_sensitive) > 0 or 
            self._check_regex(full_text, self.lex.budget_patterns)
        ) else 0
        
        # [Def] Row 37-39: 全文检索词特征
        feats['search_keyword_cov'] = 1 if str(row.get('search_keyword')) in full_text else 0
        feats['search_keyword_cnt'] = self._count_hits(full_text, self.lex.search_keywords_global)
        feats['keyword_density'] = feats['search_keyword_cnt'] / (len(full_text) + 1)
        
        return feats

    # ================= 模块 D: 标签特征 (Row 40-44) =================
    def extract_tag_features(self, row):
        tags_str = str(row.get('tags', ''))
        tags = [t.strip().replace('#', '').replace('[话题]', '') for t in re.split(r'[,，\s]', tags_str) if t.strip()]
        
        feats = {}
        
        # [Def] Row 40: 标签与正文一致性
        desc = str(row.get('desc', ''))
        overlap = sum(1 for t in tags if t in desc)
        feats['tag_content_consistency'] = overlap / len(tags) if tags else 0
        
        # [Def] Row 41: 话题标签数量
        feats['tag_cnt'] = len(tags)
        
        # [Def] Row 42: 垂直标签占比 (缺失词库，暂为0)
        v_hits = sum(1 for t in tags if t in self.lex.vertical_tags)
        feats['vertical_tag_ratio'] = v_hits / len(tags) if tags else 0
        
        # [Def] Row 43: 泛标签占比 (缺失词库，暂为0)
        g_hits = sum(1 for t in tags if t in self.lex.generic_tags)
        feats['generic_tag_ratio'] = g_hits / len(tags) if tags else 0
        
        # [Def] Row 44: 标签中核心检索词命中数
        k_hits = sum(1 for t in tags if t in self.lex.search_keywords_global)
        feats['tag_keyword_hit_cnt'] = k_hits
        
        return feats

    # ================= 主流程 =================
    def process(self, df):
        print(">>> 开始特征提取 (Rows 0-44)...")
        results = []
        for idx, row in df.iterrows():
            f_row = {}
            f_row.update(self.extract_title_features(row))
            f_row.update(self.extract_content_features(row))
            f_row.update(self.extract_semantic_features(row))
            f_row.update(self.extract_tag_features(row))
            results.append(f_row)
            
        feat_df = pd.DataFrame(results)
        meta_cols = ['note_id', 'title', 'hot_level', 'search_keyword']
        final = pd.concat([df[meta_cols], feat_df], axis=1)
        return final

# ================= 3. 运行脚本 =================
if __name__ == "__main__":
    # 1. 准备数据
    try:
        # 假设运行在根目录下
        input_file = 'data/data_with_label.csv'
        if not os.path.exists(input_file):
            # 兼容 src 目录运行
            input_file = '../data/data_with_label.csv'
            
        df = pd.read_csv(input_file)
        print(f"载入数据: {len(df)} 条")
        
        # 2. 初始化词库
        # 自动探测词库目录
        lex_dir = 'lexicons'
        if not os.path.exists(lex_dir):
            lex_dir = '../lexicons'
            
        registry = LexiconRegistry(base_dir=lex_dir)
        
        # 3. 提取特征
        extractor = FeatureExtractor(registry)
        df_features = extractor.process(df)
        
        # 4. 结果检查与保存
        print("\n特征提取完成。结果预览:")
        print(df_features[['title_len', 'info_density_score', 'ingredient_cnt', 'hot_level']].head())
        
        output_file = 'data/data_with_text_features.csv'
        df_features.to_csv(output_file, index=False, encoding='utf-8-sig')
        print(f"\n文件已保存至: {output_file}")
        
    except FileNotFoundError as e:
        print(f"文件缺失: {e}")
    except Exception as e:
        print(f"运行时错误: {e}")
