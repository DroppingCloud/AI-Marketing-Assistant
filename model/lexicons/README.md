# Lexicons 词库目录

## beauty_knowledge/

### efficacy.json
功效词

```json
{ "term": "保湿", "canonical": "保湿", "match": "contains" }
{ "term": "修护", "canonical": "修护", "synonyms": ["屏障修护", "维稳"], "match": "contains" }
```

### ingredients.json
成分词

```json
{ "term": "烟酰胺", "canonical": "niacinamide", "synonyms": ["Niacinamide"], "match": "contains" }
{ "term": "玻尿酸", "canonical": "hyaluronic_acid", "synonyms": ["透明质酸", "HA", "Hyaluronic Acid"], "match": "contains" }
```

### product_category.json
产品分类词

```json
{ "term": "卸妆膏", "canonical": "卸妆膏", "match": "contains" }
{ "term": "洗面奶", "canonical": "洗面奶", "synonyms": ["洁面"], "match": "contains" }
```

### skin_type.json
肤质词

```json
{ "term": "混油", "canonical": "混油", "synonyms": ["混合偏油", "混油皮"], "match": "contains" }
{ "term": "敏感肌", "canonical": "敏感肌", "synonyms": ["敏皮", "屏障薄"], "match": "contains" }
```

## content_style/

### colloquial.json
口语词

```json
{ "term": "真的", "canonical": "真的", "match": "contains" }
{ "term": "太", "canonical": "太", "synonyms": ["巨", "超级"], "match": "contains" }
```

### emotion.json
情感词

```json
{ "term": "闭口消失", "canonical": "有效反馈", "weight": 1.4, "match": "contains" }
{ "term": "天塌了", "canonical": "强负面", "weight": 1.8, "match": "contains" }
```

### hotwords.json
平台热词

```json
{ "term": "无限回购", "canonical": "无限回购", "synonyms": ["回购", "N次回购"], "weight": 1.6, "match": "contains" }
{ "term": "避雷", "canonical": "避雷", "synonyms": ["别踩雷"], "weight": 1.4, "match": "contains" }
```

## patterns/

### imperative_patterns.json
号召词

```json
{ "name": "cta", "regex": "(快冲|闭眼入|一定要|别买|建议避雷|直接抄|照着买|求你们去试)" }
```

### structure_patterns.json
结构词

```json
{ "name": "step_words", "regex": "(步骤|第[一二三四五六七八九十0-9]+步|先.*再.*|然后|最后|干手干脸|先打湿脸)" }
{ "name": "conclusion", "regex": "(总结|总的来说|一句话|结论|最后一句|最终建议)" }
```

## risk/

### risk_signals.json
风险信号

```json
{ "term": "苏丹红", "canonical": "安全事件", "level": "high", "weight": 2.0, "match": "contains" }
{ "name": "medical_claim", "regex": "(治愈|药妆|医疗级|消炎|激素脸治疗|处方|医生推荐)" }
```

## search/

### search_keywords.json
核心检索词库：用于 title_keyword_*、search_keyword_*、keyword_density。按品类/需求组织，面向小红书/内容平台搜索语料

```json
"cleansing_makeup_removal": {
  "terms": [
    { "term": "卸妆膏", "canonical": "卸妆膏", "synonyms": ["卸妆霜"], "match": "contains" },
    { "term": "乳化", "canonical": "乳化", "synonyms": ["乳化失败", "乳化不起来"], "match": "contains" }
  ]
}
"anti_aging_repair": {
  "terms": [
    { "term": "抗老", "canonical": "抗老", "synonyms": ["抗衰", "抗初老", "淡纹"], "match": "contains" },
    { "term": "A醇", "canonical": "视黄醇", "synonyms": ["视黄醇", "Retinol"], "match": "contains" }
  ]
}
```

## user_context/

### audience.json
受众词

```json
{ "term": "学生党", "canonical": "学生党", "weight": 1.2, "match": "contains" }
{ "term": "成分党", "canonical": "圈层-成分党", "match": "contains" }
```

### budget.json
预算词

```json
{ "term": "平价", "canonical": "平价", "weight": 1.2, "match": "contains" }
{ "name": "ceiling", "regex": "(\\d+)\\s*(元)?\\s*(以内|以下|内)" }
```

### painpoint.json
痛点词

```json
{ "term": "闷闭口", "canonical": "闷闭口", "synonyms": ["闷痘", "长闭口"], "weight": 1.4, "match": "contains" }
{ "term": "糊眼", "canonical": "糊眼", "synonyms": ["辣眼", "眼睛睁不开"], "weight": 1.4, "match": "contains" }
```

### scenario.json
场景词

```json
{ "term": "冬季", "canonical": "冬季", "match": "contains" }
{ "term": "室温10", "canonical": "低温场景", "synonyms": ["10℃", "十度"], "match": "contains" }
```

