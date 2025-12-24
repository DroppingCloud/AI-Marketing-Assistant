"""
CSV 转 Excel 转换器：将 CSV 文件转换为 Excel 文件
"""

import pandas as pd

def csv_to_excel(csv_path, excel_path):
    """
    将 CSV 文件转换为 Excel (.xlsx)

    :param csv_path: 输入 CSV 文件路径
    :param excel_path: 输出 Excel 文件路径
    """
    # 读取 CSV
    df = pd.read_csv(csv_path)

    # 保存为 Excel 文件
    df.to_excel(excel_path, index=False)

    print(f"转换完成！已生成文件：{excel_path}")


# ================= 示例调用 =================
if __name__ == "__main__":
    filename = "data_with_label"
    input_file = f"./data/{filename}.csv"
    output_file = f"./data/{filename}.xlsx"
    csv_to_excel(input_file, output_file)
