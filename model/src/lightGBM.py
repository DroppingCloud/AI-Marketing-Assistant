import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, f1_score, confusion_matrix
from lightgbm import LGBMClassifier

def run_final_fusion_model(filepath):
    print(">>> [Final Step] Feature Fusion: Statistics + Text Semantics...")
    
    # 1. 读取数据
    df = pd.read_csv(filepath)
    df['title'] = df['title'].fillna('无')
    
    # 2. 文本特征工程 (TF-IDF)
    # max_features=50: 只取出现频率最高的50个关键词，防止维度爆炸
    print("    Generating TF-IDF features from Title...")
    tfidf = TfidfVectorizer(max_features=50) 
    title_tfidf = tfidf.fit_transform(df['title']).toarray()
    
    # 将TF-IDF结果转为DataFrame，列名为关键词
    tfidf_cols = [f'txt_{word}' for word in tfidf.get_feature_names_out()]
    df_tfidf = pd.DataFrame(title_tfidf, columns=tfidf_cols)
    
    # 3. 合并特征
    # 原始统计特征（剔除ID、文本和Label）
    drop_cols = ['note_id', 'title', 'search_keyword', 'hot_level']
    X_stats = df.drop(columns=[c for c in drop_cols if c in df.columns])
    
    # 横向拼接：统计特征 + 文本特征
    X = pd.concat([X_stats, df_tfidf], axis=1)
    print(f"    Final Feature Shape: {X.shape} (Stats + 50 Text Features)")
    
    # 4. 目标变量 (Binary: High vs Rest)
    y = df['hot_level'].apply(lambda x: 1 if x in ['S', 'A'] else 0)
    
    # 5. 划分数据集
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # 6. 训练最终模型
    model = LGBMClassifier(
        objective='binary',
        class_weight='balanced',
        n_estimators=300,        # 稍微增加树的数量以适应更多特征
        learning_rate=0.03,
        random_state=42,
        verbose=-1
    )
    model.fit(X_train, y_train)
    
    # 7. 预测 (使用之前的经验阈值 0.06，或者直接用默认/调优后的)
    # 为了展示特征融合的效果，我们先看概率分布的变化
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # 简单使用一个稍微严格一点的阈值，看看Precision是否提升
    threshold = 0.1 # 稍微提高一点门槛，看看文本特征是否给了模型底气
    y_pred = (y_pred_proba >= threshold).astype(int)
    
    # 8. 最终评估
    print(f"\n--- Final Evaluation (Threshold = {threshold}) ---")
    print(classification_report(y_test, y_pred, target_names=['Rest', 'High']))
    
    # 9. 混淆矩阵
    cm = confusion_matrix(y_test, y_pred)
    print("Confusion Matrix:")
    print(cm)
    
    # 10. 查看现在的 Top 特征是哪些？(看看文本特征是否上榜)
    importance = pd.DataFrame({
        'Feature': X.columns,
        'Importance': model.feature_importances_
    }).sort_values(by='Importance', ascending=False).head(15)
    
    print("\n>>> Top 15 Features (Is Content King now?):")
    print(importance.to_string(index=False))

if __name__ == "__main__":
    run_final_fusion_model('./data/data_with_full_features.csv')