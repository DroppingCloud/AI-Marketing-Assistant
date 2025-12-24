import os
import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedShuffleSplit

if __name__ == "__main__":
    # --------------------- 加载数据 ---------------------
    input_file = './data/data_with_label.csv'
    df = pd.read_csv(input_file)
    print(f"载入数据: {len(df)} 条")

    # --------------------- 划分数据集 ---------------------
    X = df.drop(columns=['hot_level'])
    y = df['hot_level']

    # 第一次切分：分出 Test (10%) 和 Temp (90%)
    split1 = StratifiedShuffleSplit(n_splits=1, test_size=0.1, random_state=42)
    for train_idx, test_idx in split1.split(X, y):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    # 第二次切分：从 Temp 中分出 Train (80% of total) 和 Val (10% of total)
    split2 = StratifiedShuffleSplit(n_splits=1, test_size=0.1111, random_state=42)
    for train_idx, val_idx in split2.split(X_train, y_train):
        X_train, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
        y_train, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]

    # # 验证分布
    # print(f"Train 分布: {y_train.value_counts(normalize=True)}")
    # print(f"Val 分布: {y_val.value_counts(normalize=True)}")
    # print(f"Test 分布: {y_test.value_counts(normalize=True)}")

    level_map = {'S': 0, 'A': 1, 'B': 2, 'C': 3, 'D': 4}
    df['label_idx'] = df['hot_level'].map(level_map)

    split3 = StratifiedShuffleSplit(n_splits=1, test_size=0.1, random_state=42)
    for train_idx, test_idx in split1.split(df, df['label_idx']):
        temp_df = df.iloc[train_idx]
        test_df = df.iloc[test_idx]
    split4 = StratifiedShuffleSplit(n_splits=1, test_size=0.1111, random_state=42)
    for train_idx, val_idx in split2.split(temp_df, temp_df['label_idx']):
        train_df = temp_df.iloc[train_idx]
        val_df = temp_df.iloc[val_idx]

    train_df.to_csv('data/train.csv', index=False)
    val_df.to_csv('data/val.csv', index=False)
    test_df.to_csv('data/test.csv', index=False)

    print(f"Train 数据集: {len(train_df)} 条")
    print(f"Val 数据集: {len(val_df)} 条")
    print(f"Test 数据集: {len(test_df)} 条")

    

    