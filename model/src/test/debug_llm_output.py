#!/usr/bin/env python3
"""
调试LLM输出的脚本
用于查看LLM实际返回的JSON结构，帮助诊断Schema验证问题
"""

import json
import os
import sys
import pandas as pd
from openai import OpenAI

# 添加父目录到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# 加载环境变量
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# 导入配置
from notation import (
    SYSTEM_PROMPT,
    LLM_CONFIG,
    FILE_CONFIG,
    build_user_prompt,
    SCHEMA
)

def test_single_record():
    """测试单条记录的LLM输出"""
    
    print("=" * 80)
    print("LLM输出调试工具")
    print("=" * 80)
    
    # 读取第一条数据
    input_csv = FILE_CONFIG["input_csv"]
    df = pd.read_csv(input_csv)
    
    if len(df) == 0:
        print("错误：输入CSV为空")
        return
    
    row = df.iloc[0]
    note_id = row.get("note_id", "unknown")
    
    print(f"\n[1] 测试数据")
    print(f"  note_id: {note_id}")
    print(f"  title: {row.get('title', '')[:50]}...")
    
    # 构建提示词
    user_prompt = build_user_prompt(row)
    
    print(f"\n[2] 调用LLM...")
    print(f"  模型: {LLM_CONFIG['model']}")
    
    # 调用LLM
    client = OpenAI(
        api_key=os.getenv("DASHSCOPE_API_KEY") or LLM_CONFIG["api_key"],
        base_url=os.getenv("DASHSCOPE_API_BASE", "https://dashscope.aliyuncs.com/compatible-mode/v1"),
    )
    
    try:
        resp = client.chat.completions.create(
            model=LLM_CONFIG["model"],
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=LLM_CONFIG.get("temperature", 0.1),
            max_tokens=LLM_CONFIG.get("max_tokens", 2000),
        )
        
        result_text = resp.choices[0].message.content or ""
        
        print(f"  ✓ LLM调用成功")
        print(f"  Token使用: {resp.usage.total_tokens if resp.usage else 'N/A'}")
        
    except Exception as e:
        print(f"  ✗ LLM调用失败: {e}")
        return
    
    # 解析JSON
    print(f"\n[3] LLM原始输出")
    print("-" * 80)
    print(result_text)
    print("-" * 80)
    
    # 尝试解析JSON
    try:
        obj = json.loads(result_text.strip())
        print(f"\n[4] JSON解析成功")
        print(f"\n返回的字段（共{len(obj)}个）：")
        for key in sorted(obj.keys()):
            value = obj[key]
            if isinstance(value, (dict, list)):
                print(f"  - {key}: {type(value).__name__}")
            else:
                value_str = str(value)[:50]
                print(f"  - {key}: {value_str}")
        
    except json.JSONDecodeError as e:
        print(f"\n[4] JSON解析失败: {e}")
        return
    
    # 检查Schema
    print(f"\n[5] Schema验证")
    
    # 列出Schema要求的字段
    required_fields = SCHEMA.get("required", [])
    print(f"\nSchema要求的字段（共{len(required_fields)}个）：")
    for field in required_fields:
        status = "✓" if field in obj else "✗"
        print(f"  {status} {field}")
    
    # 检查额外字段
    schema_properties = set(SCHEMA.get("properties", {}).keys())
    returned_fields = set(obj.keys())
    extra_fields = returned_fields - schema_properties
    
    if extra_fields:
        print(f"\n⚠ LLM返回了{len(extra_fields)}个额外字段（Schema中未定义）：")
        for field in sorted(extra_fields):
            print(f"  - {field}")
        print("\n这些字段会导致Schema验证失败（additionalProperties: false）")
    
    missing_fields = schema_properties - returned_fields
    if missing_fields:
        print(f"\n⚠ LLM缺少{len(missing_fields)}个字段（Schema中定义但未返回）：")
        for field in sorted(missing_fields):
            print(f"  - {field}")
    
    if not extra_fields and not missing_fields:
        print(f"\n✓ 字段完全匹配！")
    
    # 尝试Schema验证
    print(f"\n[6] 执行Schema验证...")
    try:
        from jsonschema import validate
        validate(instance=obj, schema=SCHEMA)
        print("  ✓ Schema验证通过！")
    except Exception as e:
        print(f"  ✗ Schema验证失败:")
        print(f"    {str(e)[:200]}")
    
    # 保存调试输出
    debug_file = "data/debug_llm_output.json"
    with open(debug_file, "w", encoding="utf-8") as f:
        json.dump({
            "note_id": note_id,
            "llm_output": obj,
            "extra_fields": list(extra_fields),
            "missing_fields": list(missing_fields)
        }, f, ensure_ascii=False, indent=2)
    
    print(f"\n调试信息已保存到: {debug_file}")
    
    # 提供修复建议
    if extra_fields:
        print("\n" + "=" * 80)
        print("💡 修复建议")
        print("=" * 80)
        print("\n方案1: 优化Prompt（推荐）")
        print("  - 在SYSTEM_PROMPT中强调只输出Schema中定义的字段")
        print("  - 在user_prompt中列出所有必需字段")
        
        print("\n方案2: 调整Schema（如果额外字段有用）")
        print("  - 将这些字段添加到Schema的properties中")
        print("  - 或者设置 additionalProperties: true 允许额外字段")

if __name__ == "__main__":
    test_single_record()

