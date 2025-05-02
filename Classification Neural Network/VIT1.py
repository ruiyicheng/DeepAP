import torch
import torch.nn as nn
import torch.nn.functional as F


# 1. Patch Embedding: 将输入图像切分为多个patch，并将每个patch转换为一个embedding向量。
class PatchEmbedding(nn.Module):
    def __init__(self, in_channels, patch_size, embed_dim):
        super(PatchEmbedding, self).__init__()
        self.patch_size = patch_size  # 每个patch的大小
        self.embed_dim = embed_dim  # 每个patch的embedding维度
        # 使用卷积操作将patch大小映射到embedding维度
        # kernel_size=patch_size, stride=patch_size，表示每次提取一个patch
        self.conv = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        # x: (batch_size, channels, height, width) 输入图像的尺寸
        x = self.conv(x)  # 对输入x进行卷积操作，输出的形状为(batch_size, embed_dim, height/patch_size, width/patch_size)
        x = x.flatten(
            2)  # 将输出展平，变为(batch_size, embed_dim, num_patches)，num_patches = (height/patch_size) * (width/patch_size)
        x = x.transpose(1, 2)  # 转置为(batch_size, num_patches, embed_dim)，以便后续Transformer处理
        return x


# 2. Transformer Block: Vision Transformer中的核心模块，包含Multi-head Attention 和 Feed Forward网络
class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_hid_dim, dropout=0.1):
        super(TransformerBlock, self).__init__()
        # 多头自注意力机制
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout)
        # 前馈神经网络（Feed-forward network）
        self.ff = nn.Sequential(
            nn.Linear(embed_dim, ff_hid_dim),  # 从embed_dim到ff_hid_dim的线性变换
            nn.GELU(),  # 激活函数，GELU在Transformer中更常用
            nn.Linear(ff_hid_dim, embed_dim)  # 从ff_hid_dim到embed_dim的线性变换
        )
        # 归一化层：在每个子模块后进行残差连接和归一化
        self.layer_norm1 = nn.LayerNorm(embed_dim)
        self.layer_norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: (batch_size, num_patches, embed_dim) 输入数据
        # 转置为 (num_patches, batch_size, embed_dim) 以匹配 MultiheadAttention 的输入要求
        x_transposed = x.transpose(0, 1)  # (num_patches, batch_size, embed_dim)

        # 自注意力机制
        attn_output, _ = self.attn(x_transposed, x_transposed, x_transposed)  # (num_patches, batch_size, embed_dim)
        attn_output = self.dropout(attn_output)

        # 残差连接和LayerNorm
        x = x_transposed + attn_output  # (num_patches, batch_size, embed_dim)
        x = self.layer_norm1(x.transpose(0, 1))  # 转置回 (batch_size, num_patches, embed_dim) 后归一化

        # 前馈神经网络
        ff_output = self.ff(x)  # (batch_size, num_patches, embed_dim)
        ff_output = self.dropout(ff_output)

        # 残差连接和LayerNorm
        x = x + ff_output  # (batch_size, num_patches, embed_dim)
        x = self.layer_norm2(x)
        return x


# 3. Vision Transformer (ViT) 主模型
class VisionTransformer(nn.Module):
    def __init__(self, num_classes=2, embed_dim=256, num_heads=8, num_blocks=12, ff_hid_dim=512, patch_size=16,
                 img_size=128, in_channels=1, dropout=0.1):
        super(VisionTransformer, self).__init__()
        # 1. Patch Embedding层，将输入图像划分为patch并转换为embedding向量
        self.patch_embed = PatchEmbedding(in_channels, patch_size, embed_dim)  # 1: 单通道图像（例如灰度图像）

        # 计算patch数量
        num_patches = (img_size // patch_size) ** 2

        # 2. 初始化[CLS] token，通常是一个学习的参数，用于分类任务
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))  # 初始化[CLS] token，大小为[1, 1, embed_dim]

        # 3. 初始化位置编码（Positional Embedding），用于给patches加上位置信息
        self.pos_embed = nn.Parameter(torch.randn(1, num_patches + 1, embed_dim))  # +1 是因为有 [CLS] token

        # 4. Transformer层，多个TransformerBlock堆叠
        self.blocks = nn.ModuleList(
            [TransformerBlock(embed_dim, num_heads, ff_hid_dim, dropout) for _ in range(num_blocks)])

        # 5. 归一化层
        self.norm = nn.LayerNorm(embed_dim)

        # 6. 最后的分类层：使用[CLS] token的输出进行分类
        self.fc = nn.Linear(embed_dim, num_classes)  # 输出类别数为num_classes

        # 7. 初始化权重
        self._init_weights()

    def _init_weights(self):
        # 初始化权重
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.xavier_uniform_(self.fc.weight)
        if self.fc.bias is not None:
            nn.init.zeros_(self.fc.bias)

    def forward(self, x):
        # Step 1: Patch embedding
        patches = self.patch_embed(x)  # 输入图像转换为patches的embedding表示

        # Step 2: Add [CLS] token and positional encoding
        batch_size = patches.size(0)  # 获取batch_size
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)  # 扩展[CLS] token，使其与batch_size匹配
        x = torch.cat((cls_tokens, patches), dim=1)  # 将[CLS] token与patches拼接起来
        x = x + self.pos_embed  # 加上位置编码，给每个patch添加位置信息

        # Step 3: Pass through Transformer blocks
        for block in self.blocks:
            x = block(x)  # 通过所有TransformerBlock进行前向传播

        # Step 4: 归一化
        x = self.norm(x)

        # Step 5: Use the [CLS] token for classification
        cls_token = x[:, 0]  # 取出[CLS] token的输出（位置0的元素），用于分类
        out = self.fc(cls_token)  # 将[CLS] token通过全连接层进行分类
        return out


# 4. 创建ViT模型并进行测试
if __name__ == "__main__":
    # 创建一个Vision Transformer模型，设定参数
    model = VisionTransformer(
        num_classes=2,
        embed_dim=512,
        num_heads=8,  # 根据 embed_dim 调整 num_heads，例如 embed_dim=512 时，num_heads=8 是合理的（512/8=64）
        num_blocks=6,  # 减少层数以适应硬件资源
        ff_hid_dim=1024,
        patch_size=32,
        img_size=128,
        in_channels=1,
        dropout=0.1
    )

    # 打印模型结构（可选）
    print(model)

    # 生成一个随机输入tensor，表示一张128x128的灰度图像
    x = torch.randn(1, 1, 128, 128)  # Batch size = 1, Image size = 128x128, Channels = 1 (灰度图像)

    # 进行前向传播，得到输出
    output = model(x)
    print("Output shape:", output.shape)  # 输出shape应该是(1, num_classes)，即(1, 2)

    # 确认输出
    assert output.shape == (1, 2), f"Expected output shape (1, 2), but got {output.shape}"
    print("模型前向传播测试通过！")
