#!/usr/bin/env python3
"""
API配置测试脚本
用于验证通义千问API密钥是否正确配置
"""

import os
from openai import OpenAI

# 加载环境变量
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    print("⚠ 未安装 python-dotenv，将直接读取环境变量")

def test_dashscope_api():
    """测试通义千问API连接"""
    
    print("=" * 60)
    print("DashScope API 配置测试")
    print("=" * 60)
    
    # 检查.env文件位置
    env_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), '.env')
    print(f"\n检查 .env 文件位置: {env_path}")
    if os.path.exists(env_path):
        print("  ✓ .env 文件存在")
    else:
        print("  ✗ .env 文件不存在")
    
    # 检查API密钥（优先DASHSCOPE_API_KEY，其次LLM_API_KEY）
    api_key = os.getenv("DASHSCOPE_API_KEY") or os.getenv("LLM_API_KEY")
    
    if not api_key:
        print("\n✗ 错误：未找到 DASHSCOPE_API_KEY 环境变量")
        print("\n解决方案：")
        print("  1. 在项目根目录创建 .env 文件")
        print(f"     位置: {env_path}")
        print("  2. 添加一行：DASHSCOPE_API_KEY=你的真实密钥")
        print("  3. 确保密钥格式正确（如: sk-xxxxx）")
        print("  4. 重新运行此脚本")
        print("\n提示：密钥获取地址 https://dashscope.console.aliyun.com/")
        return False
    
    # 隐藏部分密钥显示
    masked_key = f"{api_key[:8]}...{api_key[-4:]}" if len(api_key) > 12 else "***"
    print(f"\n✓ 找到API密钥: {masked_key}")
    
    # 测试API调用
    print("\n正在测试API连接...")
    
    try:
        client = OpenAI(
            api_key=api_key,
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        )
        
        response = client.chat.completions.create(
            model="qwen-plus",  # 使用qwen-plus进行测试
            messages=[
                {"role": "system", "content": "你是一个测试助手"},
                {"role": "user", "content": "请回复：测试成功"}
            ],
            max_tokens=20
        )
        
        result = response.choices[0].message.content
        print(f"✓ API调用成功！")
        print(f"  模型响应: {result}")
        print(f"\n🎉 配置正确，可以开始标注任务！")
        return True
        
    except Exception as e:
        print(f"\n✗ API调用失败")
        print(f"  错误信息: {str(e)}")
        
        # 提供详细的错误诊断
        error_str = str(e)
        
        if "401" in error_str or "Unauthorized" in error_str:
            print("\n📋 错误原因: API密钥无效")
            print("  解决方案:")
            print("    1. 检查密钥是否完整复制（包括sk-前缀）")
            print("    2. 确认密钥未过期")
            print("    3. 访问 https://dashscope.console.aliyun.com/ 重新生成")
            
        elif "404" in error_str:
            print("\n📋 错误原因: API地址或模型不存在")
            print("  解决方案:")
            print("    1. 检查base_url是否正确")
            print("    2. 确认使用的模型名称是否可用")
            
        elif "429" in error_str:
            print("\n📋 错误原因: 请求频率超限")
            print("  解决方案:")
            print("    1. 稍等片刻后重试")
            print("    2. 增大 RUN_CONFIG['sleep_between'] 值")
            
        elif "timeout" in error_str.lower():
            print("\n📋 错误原因: 网络超时")
            print("  解决方案:")
            print("    1. 检查网络连接")
            print("    2. 如使用代理，请正确配置")
            
        else:
            print("\n📋 其他错误，请检查:")
            print("    1. 网络连接是否正常")
            print("    2. API服务是否可用")
            print("    3. 阿里云账户余额是否充足")
        
        return False

if __name__ == "__main__":
    success = test_dashscope_api()
    exit(0 if success else 1)

