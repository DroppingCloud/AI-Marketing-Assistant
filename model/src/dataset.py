import os
import pandas as pd

import torch
from PIL import Image                                   # 图像处理库
from torchvision import transforms                      # 图像预处理库 
from transformers import BertTokenizer                  # 文本处理库
from torch.utils.data import Dataset, DataLoader        # 数据加载库

# --- 配置参数 ---
class ModelConfig:
    # 标签映射表
    LABEL_MAP = {'S': 0, 'A': 1, 'B': 2, 'C': 3, 'D': 4}
    
    # 图像参数 (ResNet 标准输入)
    IMG_SIZE = 224
    IMG_MEAN = [0.485, 0.456, 0.406]
    IMG_STD = [0.229, 0.224, 0.225]
    
    # 文本参数 (BERT)
    BERT_PATH = 'bert-base-chinese' # 首次运行会自动下载，也可以指定本地路径
    MAX_LEN = 128                            # 文本最大长度      

class MultiModalDataset(Dataset):
    def __init__(self, df, img_root_dir, tokenizer, transform=None, mode='train'):
        """
        Args:
            df (DataFrame): 包含数据的 pandas DataFrame
            img_root_dir (str): 图片存放的根目录 (如果 csv 里是相对路径)
            tokenizer: BERT 分词器
            transform: 图像预处理函数
            mode (str): 'train' 或 'val'/'test'。train 模式下可能会做数据增强。
        """
        self.df = df.reset_index(drop=True)                     # 重置索引(保证索引连续)
        self.img_root_dir = img_root_dir                        # 图片根目录
        self.tokenizer = tokenizer                              # BERT 分词器
        self.transform = transform                              # 图像预处理函数
        self.mode = mode                                        # 模式

        self._debug_bad_printed = 0
        self._debug_max_prints = 20

    def __len__(self):
        """ 返回数据集长度 """
        return len(self.df)

    def __getitem__(self, idx):
        """ 获取单个样本 """
        row = self.df.iloc[idx]                                         # 获取单个样本
        
        # ---------------- 1. 处理 Label ----------------
        label_str = row['hot_level']                                    # 获取标签
        label = ModelConfig.LABEL_MAP.get(label_str, 4)                 # 获取标签映射(默认为 4)
        
        # ---------------- 2. 处理 Image ----------------
        img_path = os.path.join(self.img_root_dir, row['cover_path'])   # 获取图片路径
        
        try:
            image = Image.open(img_path).convert('RGB')                 # 打开图片并转换为 RGB 模式
        except Exception as e:
            # 工程容错：如果图片损坏或找不到，创建一个全黑图片防止训练中断
            print(f"Warning: Image not found {img_path}, using black image.")
            image = Image.new('RGB', (ModelConfig.IMG_SIZE, ModelConfig.IMG_SIZE), (0, 0, 0))

        if self.transform:
            image = self.transform(image)                               # 应用图像预处理函数
            
        # ---------------- 3. 处理 Text ----------------
        # 拼接标题和正文
        title = row.get('title', '')
        desc = row.get('desc', '')

        if pd.isna(title):
            title = ''
        if pd.isna(desc):
            desc = ''

        # 标题在前，描述在后，中间空格分隔
        text_content = f"{str(title)} {str(desc)}".strip()
        
        encoding = self.tokenizer.encode_plus(
            text_content,
            add_special_tokens=True,        # 添加 [CLS] 和 [SEP]
            max_length=ModelConfig.MAX_LEN,
            padding='max_length',           # 填充到固定长度
            truncation=True,                # 截断过长文本
            return_attention_mask=True,
            return_tensors='pt',            # 返回 PyTorch Tensor
        )

        return {
            'image': image,                               # [3, 224, 224]
            'input_ids': encoding['input_ids'].flatten(), # [128]
            'attention_mask': encoding['attention_mask'].flatten(), # [128]
            'label': torch.tensor(label, dtype=torch.long) # Scalar
        }

# --- 获取 Transform ---
def get_transforms(mode='train'):
    """
    mode='train': 常见增强，提高泛化
    mode!='train': 验证/测试不做随机增强，保证评估稳定
    """
    
    if mode == 'train':
        return transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomCrop((ModelConfig.IMG_SIZE, ModelConfig.IMG_SIZE)), # 随机裁剪增加鲁棒性
            transforms.RandomHorizontalFlip(), # 随机水平翻转
            transforms.ToTensor(),
            transforms.Normalize(mean=ModelConfig.IMG_MEAN, std=ModelConfig.IMG_STD)
        ])
    else:
        return transforms.Compose([
            transforms.Resize((ModelConfig.IMG_SIZE, ModelConfig.IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=ModelConfig.IMG_MEAN, std=ModelConfig.IMG_STD)
        ])

# ================= 单元测试 (Unit Test) =================
if __name__ == "__main__":
    # 模拟环境
    print("正在初始化 Tokenizer...")
    tokenizer = BertTokenizer.from_pretrained(ModelConfig.BERT_PATH)
    
    # 假设你之前保存的 train.csv
    try:
        input_file = './data/data_with_label.csv'
        train_df = pd.read_csv(input_file)

        # 图片根目录
        img_root = "" 
        
        dataset = MultiModalDataset(train_df.head(5), img_root, tokenizer, transform=get_transforms('train'))
        dataloader = DataLoader(dataset, batch_size=2, shuffle=True)
        
        print("\n正在测试 DataLoader 输出形状...")
        for batch in dataloader:
            print("Image Shape:", batch['image'].shape)       # 应为 [2, 3, 224, 224]
            print("Text IDs Shape:", batch['input_ids'].shape) # 应为 [2, 128]
            print("Label Shape:", batch['label'].shape)       # 应为 [2]
            print("Label Example:", batch['label'])
            break
            
        print("\n✅ Dataset 测试通过！数据流管道正常。")
        print("[Dataset] total:", len(dataset.df))
        print("[Dataset] cover_path null:", dataset.df["cover_path"].isna().sum())
        print("[Dataset] cover_path type counts:\n", dataset.df["cover_path"].map(type).value_counts().head())
        
    except FileNotFoundError:
        print("未找到 train.csv 或 模型文件，跳过测试。请确保文件路径正确。")