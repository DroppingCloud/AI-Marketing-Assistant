import torch
import torch.nn as nn
from transformers import BertModel
from torchvision import models

# 工业界惯例：固定随机种子，保证模型初始化的参数一致，方便复现
torch.manual_seed(42)

class ViralFlowModel(nn.Module):
    def __init__(self, num_classes=5, freeze_backbone=True):
        """
        Args:
            num_classes (int): 分类数量 (S/A/B/C/D -> 5类)
            freeze_backbone (bool): 是否冻结 BERT 和 ResNet 的参数。
                                    对于小数据集(5000条)，建议设为 True，
                                    只训练最后的分类层，防止过拟合。
        """
        super(ViralFlowModel, self).__init__()
        
        # ==================== 1. Text Tower (左脑: BERT) ====================
        # 加载预训练的中文 BERT
        print("正在加载 BERT 预训练权重...")
        self.bert = BertModel.from_pretrained('bert-base-chinese')
        self.text_dim = 768  # BERT base 的输出维度固定为 768
        
        # ==================== 2. Image Tower (右脑: ResNet) ====================
        # 加载预训练的 ResNet50
        print("正在加载 ResNet50 预训练权重...")
        resnet = models.resnet50(pretrained=True)
        
        # 我们只需要 ResNet 提取特征，不需要它原本的 1000类分类头
        # 去掉最后的全连接层 (fc)，保留之前的层作为特征提取器
        # ResNet50 最后一层全连接前的输出维度是 2048
        self.vision_encoder = nn.Sequential(*list(resnet.children())[:-1]) 
        self.img_dim = 2048

        # ==================== 3. Fusion Layer (决策层) ====================
        # 将 [文本特征 768] 和 [视觉特征 2048] 拼接 -> 2816 维
        fusion_input_dim = self.text_dim + self.img_dim
        
        # 定义分类器结构：
        # Linear -> BatchNorm -> ReLU -> Dropout -> Linear
        self.classifier = nn.Sequential(
            nn.Linear(fusion_input_dim, 512),
            nn.BatchNorm1d(512), # 加速收敛，防止梯度消失
            nn.ReLU(),
            nn.Dropout(0.3),     # 随机丢弃 30% 的神经元，防止过拟合 (这是小样本的关键)
            nn.Linear(512, num_classes)
        )

        # ==================== 4. 参数冻结策略 ====================
        if freeze_backbone:
            self._freeze_params()

    def _freeze_params(self):
        """冻结 BERT 和 ResNet 的参数，不让它们在训练中更新"""
        print("[INFO] 冻结骨干网络参数，仅训练分类头...")
        for param in self.bert.parameters():
            param.requires_grad = False
        for param in self.vision_encoder.parameters():
            param.requires_grad = False

    def forward(self, input_ids, attention_mask, image):
        """
        前向传播逻辑
        """
        # --- 1. 文本流 ---
        # BERT 输出是一个对象，其中 pooler_output 是 [CLS] 标记的向量，代表整句话的语义
        # Shape: [Batch_Size, 768]
        bert_output = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        text_features = bert_output.pooler_output 

        # --- 2. 视觉流 ---
        # ResNet 输出 Shape: [Batch_Size, 2048, 1, 1]
        vision_features = self.vision_encoder(image)
        # 展平为 [Batch_Size, 2048]
        vision_features = vision_features.view(vision_features.size(0), -1)

        # --- 3. 特征融合 ---
        # 在第1维度(列)进行拼接
        # combined_features Shape: [Batch_Size, 768 + 2048]
        combined_features = torch.cat((text_features, vision_features), dim=1)

        # --- 4. 分类预测 ---
        # logits Shape: [Batch_Size, num_classes]
        logits = self.classifier(combined_features)
        
        return logits

if __name__ == "__main__":
    print("=== 模型单元测试开始 ===")
    
    # 1. 实例化模型
    # 第一次运行会下载 BERT (约400MB) 和 ResNet (约100MB) 权重
    model = ViralFlowModel(num_classes=5, freeze_backbone=True)
    
    # 2. 模拟一个 Batch 的数据 (Batch Size = 2)
    batch_size = 2
    
    # 模拟文本输入: [Batch, Sequence Length]
    dummy_input_ids = torch.randint(0, 1000, (batch_size, 128)) 
    dummy_mask = torch.ones((batch_size, 128))
    
    # 模拟图片输入: [Batch, Channel, Height, Width]
    dummy_image = torch.randn((batch_size, 3, 224, 224))
    
    # 3. 前向传播
    print("\n正在执行前向传播...")
    outputs = model(dummy_input_ids, dummy_mask, dummy_image)
    
    # 4. 检查输出
    print(f"\n输入 Batch Size: {batch_size}")
    print(f"模型输出 Shape: {outputs.shape}") # 应该输出 [2, 5]
    
    # 检查数值是否正常 (不应出现 NaN)
    if torch.isnan(outputs).any():
        print("❌ 警告：输出包含 NaN (数值溢出)！")
    else:
        print("✅ 测试通过：输出数值正常。")
        print("输出示例 (Logits):\n", outputs.detach().numpy())