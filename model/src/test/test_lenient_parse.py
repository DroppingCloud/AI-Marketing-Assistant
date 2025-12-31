#!/usr/bin/env python3
"""
测试宽容解析功能
"""

import json
from notation import lenient_parse, SCHEMA

# 测试案例：包含无效枚举值的JSON
test_obj = {
    "note_id": "test123",
    "schema_version": "mkctx_v2.0",
    "language": "中文",
    "b_brand_tier": "大众",
    "b_brand_origin": "国货",
    "b_product_stage": "常青款",
    "b_primary_category": "护肤",
    "b_campaign_goal": ["提升互动/种草"],
    "b_benchmarking": {
        "has_benchmark": True,
        "benchmark_type": "品牌"  # ❌ 错误：应该是"明确品牌对标"
    },
    "c_need_archetype": "皮肤问题解决型",
    "c_efficacy_goal": ["修护", "保湿"],
    "c_lifecycle": ["未知"],
    "c_skin_type": ["干皮"],
    "c_skin_concerns": ["干燥", "敏感", "换季不适"],  # ❌ "换季不适"不在枚举中
    "c_budget_band": "0~100",
    "c_region_climate": "北方干冷",
    "c_channel_behavior": ["搜索导向"],
    "scene_marketing_nodes": ["未知"],
    "scene_season": "秋",
    "scene_climate": ["干燥"],
    "confidence": {
        "overall": 0.85,
        "fields_low_confidence": []
    },
    "evidence": {
        "text_snippets": ["test"],
        "used_fields": ["title"]
    }
}

print("=" * 80)
print("测试宽容解析功能")
print("=" * 80)

print("\n[输入] 包含错误的JSON:")
print(json.dumps(test_obj, ensure_ascii=False, indent=2))

print("\n[处理] 执行宽容解析...")
cleaned_obj, errors = lenient_parse(test_obj)

print("\n[输出] 清洗后的JSON:")
print(json.dumps(cleaned_obj, ensure_ascii=False, indent=2))

print("\n[错误列表]")
if errors:
    for i, err in enumerate(errors, 1):
        print(f"  {i}. {err}")
else:
    print("  无错误")

print("\n[验证] 检查是否通过Schema验证...")
try:
    from jsonschema import validate
    validate(instance=cleaned_obj, schema=SCHEMA)
    print("  ✓ Schema验证通过！")
except Exception as e:
    print(f"  ✗ Schema验证失败: {e}")

print("\n" + "=" * 80)
print("测试完成")
print("=" * 80)

