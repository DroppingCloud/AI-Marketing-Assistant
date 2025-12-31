import torch
import torch.nn as nn
from transformers import BertModel
from torchvision import models
from dataset import ModelConfig

# 工业界惯例：固定随机种子，保证模型初始化的参数一致，方便复现
torch.manual_seed(42)

class CrossModalAttentionFusion(nn.Module):
    """
    跨模态注意力融合模块
    通过注意力机制让文本和视觉特征相互增强
    """
    def __init__(self, text_dim, img_dim, hidden_dim=512):
        super(CrossModalAttentionFusion, self).__init__()
        
        # 将两个模态投影到相同维度
        self.text_proj = nn.Linear(text_dim, hidden_dim)
        self.img_proj = nn.Linear(img_dim, hidden_dim)
        
        # 多头注意力机制
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=8,
            dropout=0.2,  # 增加dropout
            batch_first=True
        )
        
        # 门控机制：学习如何融合两种模态
        self.gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Sigmoid()
        )
        
        # Layer Normalization
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.ln2 = nn.LayerNorm(hidden_dim)
        
    def forward(self, text_features, vision_features):
        """
        Args:
            text_features: [batch_size, text_dim]
            vision_features: [batch_size, img_dim]
        Returns:
            fused_features: [batch_size, hidden_dim * 2]
        """
        # 投影到相同维度
        text_proj = self.text_proj(text_features)  # [B, hidden_dim]
        img_proj = self.img_proj(vision_features)   # [B, hidden_dim]
        
        # 增加序列维度以使用MultiheadAttention
        # [B, 1, hidden_dim]
        text_seq = text_proj.unsqueeze(1)
        img_seq = img_proj.unsqueeze(1)
        
        # 交叉注意力：让文本关注图像特征
        text_attended, _ = self.cross_attention(
            query=text_seq,
            key=img_seq,
            value=img_seq
        )
        text_attended = text_attended.squeeze(1)  # [B, hidden_dim]
        text_enhanced = self.ln1(text_proj + text_attended)  # 残差连接
        
        # 交叉注意力：让图像关注文本特征
        img_attended, _ = self.cross_attention(
            query=img_seq,
            key=text_seq,
            value=text_seq
        )
        img_attended = img_attended.squeeze(1)  # [B, hidden_dim]
        img_enhanced = self.ln2(img_proj + img_attended)  # 残差连接
        
        # 门控融合
        gate_input = torch.cat([text_enhanced, img_enhanced], dim=1)
        gate_weights = self.gate(gate_input)  # [B, hidden_dim]
        
        # 加权融合：动态平衡两种模态的贡献
        gated_text = gate_weights * text_enhanced
        gated_img = (1 - gate_weights) * img_enhanced
        
        # 拼接作为最终输出
        fused = torch.cat([gated_text, gated_img], dim=1)  # [B, hidden_dim * 2]
        
        return fused

class ViralFlowModel(nn.Module):
    def __init__(self, num_classes=5, freeze_backbone=True, use_attention_fusion=True):
        """
        Args:
            num_classes (int): 分类数量 (S/A/B/C/D -> 5类)
            freeze_backbone (bool): 是否冻结 BERT 和 ResNet 的参数。
                                    对于小数据集(5000条)，建议设为 True，
                                    只训练最后的分类层，防止过拟合。
            use_attention_fusion (bool): 是否使用注意力融合机制（推荐！）
                                        True=注意力融合，False=简单拼接
        """
        super(ViralFlowModel, self).__init__()
        self.use_attention_fusion = use_attention_fusion
        
        # ==================== 1. Text Tower (左脑: BERT) ====================
        # 加载预训练的中文 BERT
        print("正在加载 BERT 预训练权重...")
        self.bert = BertModel.from_pretrained(ModelConfig.BERT_PATH, local_files_only=True)
        self.text_dim = 768  # BERT base 的输出维度固定为 768
        
        # ==================== 2. Image Tower (右脑: ResNet) ====================
        # 加载预训练的 ResNet50
        print("正在加载 ResNet50 预训练权重...")
        resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        
        # 我们只需要 ResNet 提取特征，不需要它原本的 1000类分类头
        # 去掉最后的全连接层 (fc)，保留之前的层作为特征提取器
        # ResNet50 最后一层全连接前的输出维度是 2048
        self.vision_encoder = nn.Sequential(*list(resnet.children())[:-1]) 
        self.img_dim = 2048

        # ==================== 3. Fusion Layer (决策层) ====================
        if self.use_attention_fusion:
            # 使用注意力融合
            print("[INFO] 使用跨模态注意力融合机制")
            self.fusion = CrossModalAttentionFusion(
                text_dim=self.text_dim,
                img_dim=self.img_dim,
                hidden_dim=512
            )
            fusion_output_dim = 512 * 2  # 融合后输出 1024 维
        else:
            # 使用简单拼接
            print("[INFO] 使用简单特征拼接")
            self.fusion = None
            fusion_output_dim = self.text_dim + self.img_dim  # 768 + 2048 = 2816
        
        # 定义分类器结构：
        # Linear -> BatchNorm -> ReLU -> Dropout -> Linear
        self.classifier = nn.Sequential(
            nn.Linear(fusion_output_dim, 512),
            nn.BatchNorm1d(512), # 加速收敛，防止梯度消失
            nn.ReLU(),
            nn.Dropout(0.5),     # 增加到50%的dropout，增强正则化，防止过拟合
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
    
    def unfreeze_progressively(self, stage=1):
        """
        渐进式解冻策略：逐步解冻骨干网络的高层参数
        
        Args:
            stage (int): 解冻阶段
                - 0: 全部冻结（默认状态）
                - 1: 解冻BERT最后1层 + ResNet最后1个block
                - 2: 解冻BERT最后2层 + ResNet最后2个block
                - 3: 解冻BERT最后4层 + ResNet最后3个block（几乎全解冻）
        
        Returns:
            int: 解冻的参数数量
        """
        unfrozen_count = 0
        
        if stage == 0:
            # 保持全部冻结
            return unfrozen_count
        
        # === 解冻 BERT ===
        if stage >= 1:
            # BERT有12层encoder，从后往前解冻
            layers_to_unfreeze = min(stage, 4)  # 最多解冻4层
            print(f"[INFO] 解冻 BERT 最后 {layers_to_unfreeze} 层...")
            
            # 解冻pooler（用于分类的池化层）
            for param in self.bert.pooler.parameters():
                param.requires_grad = True
                unfrozen_count += 1
            
            # 解冻encoder的最后几层
            total_layers = len(self.bert.encoder.layer)
            for i in range(total_layers - layers_to_unfreeze, total_layers):
                for param in self.bert.encoder.layer[i].parameters():
                    param.requires_grad = True
                    unfrozen_count += 1
        
        # === 解冻 ResNet ===
        # ResNet50结构: conv1, bn1, relu, maxpool, layer1, layer2, layer3, layer4
        # layer4是最高层特征，layer3次之
        if stage >= 1:
            blocks_to_unfreeze = min(stage, 3)  # 最多解冻3个block
            print(f"[INFO] 解冻 ResNet 最后 {blocks_to_unfreeze} 个block...")
            
            # vision_encoder是Sequential，我们需要找到layer3和layer4
            # ResNet的children顺序: conv1, bn1, relu, maxpool, layer1, layer2, layer3, layer4
            resnet_children = list(self.vision_encoder.children())
            
            # 最后几个block（倒序）
            if blocks_to_unfreeze >= 1 and len(resnet_children) >= 8:
                # 解冻layer4（索引7）
                for param in resnet_children[7].parameters():
                    param.requires_grad = True
                    unfrozen_count += 1
            
            if blocks_to_unfreeze >= 2 and len(resnet_children) >= 7:
                # 解冻layer3（索引6）
                for param in resnet_children[6].parameters():
                    param.requires_grad = True
                    unfrozen_count += 1
            
            if blocks_to_unfreeze >= 3 and len(resnet_children) >= 6:
                # 解冻layer2（索引5）
                for param in resnet_children[5].parameters():
                    param.requires_grad = True
                    unfrozen_count += 1
        
        print(f"[INFO] 渐进式解冻 Stage {stage}: 共解冻 {unfrozen_count} 个参数组")
        return unfrozen_count

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
        if self.use_attention_fusion:
            # 使用注意力融合
            combined_features = self.fusion(text_features, vision_features)
        else:
            # 简单拼接
            # combined_features Shape: [Batch_Size, 768 + 2048]
            combined_features = torch.cat((text_features, vision_features), dim=1)

        # --- 4. 分类预测 ---
        # logits Shape: [Batch_Size, num_classes]
        logits = self.classifier(combined_features)
        
        return logits

if __name__ == "__main__":
    print("=== 模型单元测试开始 ===")
    
    # 1. 实例化模型（测试两种融合方式）
    print("\n--- 测试1: 注意力融合模式 ---")
    model_attention = ViralFlowModel(num_classes=5, freeze_backbone=True, use_attention_fusion=True)
    
    print("\n--- 测试2: 简单拼接模式 ---")
    model_concat = ViralFlowModel(num_classes=5, freeze_backbone=True, use_attention_fusion=False)
    
    # 使用注意力融合模型进行测试
    model = model_attention
    
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