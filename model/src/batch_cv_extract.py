"""
批量视觉特征提取脚本
从 data_with_features.csv 中读取 note_id，在 image/ 目录下查找对应图片，
提取视觉特征后追加到原 CSV 文件中。
"""
import pandas as pd
import os
from tqdm import tqdm
from cvFeature_distill import VisualFeatureExtractor

def find_images(base_dir):
    """
    遍历 image 目录，建立 note_id -> image_path 的映射
    """
    print(f"正在扫描图片目录: {base_dir} ...")
    image_map = {}
    
    # 递归查找所有图片文件
    for root, dirs, files in os.walk(base_dir):
        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg', '.png', '.webp')):
                # 文件名即为 note_id
                note_id = os.path.splitext(file)[0]
                full_path = os.path.join(root, file)
                image_map[note_id] = full_path
    
    print(f"找到 {len(image_map)} 张图片。")
    return image_map

def main():
    # ========== 1. 路径配置 ==========
    csv_path = 'data/data_with_text_features.csv'
    image_base_dir = 'image'
    
    # 兼容从 src/ 目录运行的情况
    if not os.path.exists(csv_path):
        csv_path = '../data/data_with_text_features.csv'
        image_base_dir = '../image'
    
    if not os.path.exists(csv_path):
        print(f"❌ Error: CSV文件不存在 {csv_path}")
        return
    
    if not os.path.exists(image_base_dir):
        print(f"❌ Error: 图片目录不存在 {image_base_dir}")
        return

    # ========== 2. 加载数据 ==========
    print(f"📖 读取数据: {csv_path}")
    df = pd.read_csv(csv_path)
    print(f"   数据行数: {len(df)}")
    
    # ========== 3. 建立图片索引 ==========
    image_map = find_images(image_base_dir)
    
    # ========== 4. 初始化提取器 ==========
    print("🚀 初始化视觉特征提取器...")
    # 为了稳定性，禁用 MediaPipe，只使用 OpenCV
    extractor = VisualFeatureExtractor(use_mediapipe=False)
    
    # ========== 5. 批量提取 ==========
    print("⚙️  开始批量提取视觉特征...")
    
    # 获取特征列名模板
    sample_feats = extractor.extract("non_existent_path")
    feature_keys = list(sample_feats.keys())
    
    print(f"   将提取以下 {len(feature_keys)} 个特征:")
    print(f"   {', '.join(feature_keys)}")
    
    # 初始化新列（如果不存在）
    for col in feature_keys:
        if col not in df.columns:
            df[col] = 0.0
    
    success_count = 0
    missing_count = 0
    error_count = 0
    
    # 使用 tqdm 显示进度
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="提取进度"):
        note_id = str(row['note_id'])
        
        if note_id in image_map:
            img_path = image_map[note_id]
            try:
                feats = extractor.extract(img_path)
                
                # 更新特征
                for k in feature_keys:
                    df.at[idx, k] = feats.get(k, 0.0)
                
                success_count += 1
            except Exception as e:
                # 单张图片处理失败，不影响整体
                error_count += 1
                if error_count <= 5:  # 只打印前5个错误
                    print(f"\n⚠️  处理失败 {note_id}: {e}")
        else:
            missing_count += 1
    
    # ========== 6. 结果统计 ==========
    print(f"\n{'='*50}")
    print(f"✅ 提取完成！")
    print(f"   成功处理: {success_count} 张")
    print(f"   未找到图片: {missing_count} 张 (特征置0)")
    print(f"   处理失败: {error_count} 张 (特征置0)")
    print(f"{'='*50}")
    
    # ========== 7. 保存结果 ==========
    output_file = 'data/data_with_full_features.csv'
    print(f"💾 保存结果至: {output_file}")
    df.to_csv(output_file, index=False, encoding='utf-8-sig')
    print(f"✨ 完成！")

if __name__ == "__main__":
    main()

