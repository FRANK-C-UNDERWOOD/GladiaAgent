"""
🧠🧠 Tensor Network Compression Module for Triple Embeddings
Inspired by the paper: "Compressing Neural Networks Using Tensor Networks with Exponentially Fewer Variational Parameters"
Author: DOCTOR + 歌蕾蒂娅 (2025)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class TripleCompressor(nn.Module):
    """三元组压缩器 - 统一使用384维嵌入"""
    
    def __init__(self, embed_dim=384, num_entities=10000, num_relations=100, device='cpu'):
        super(TripleCompressor, self).__init__()
        
        self.embed_dim = embed_dim  # 嵌入维度设为384
        self.num_entities = num_entities
        self.num_relations = num_relations
        self.device = device
        
        # 实体和关系嵌入层 - 都使用384维
        self.entity_embeddings = nn.Embedding(num_entities, embed_dim)
        self.relation_embeddings = nn.Embedding(num_relations, embed_dim)
        
        # 初始化嵌入权重
        nn.init.xavier_uniform_(self.entity_embeddings.weight)
        nn.init.xavier_uniform_(self.relation_embeddings.weight)
        
        # 三元组融合层
        self.triplet_fusion = nn.Sequential(
            nn.Linear(embed_dim * 3, embed_dim * 2),  # 三个384维向量融合
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(embed_dim * 2, embed_dim),      # 压缩到384维
            nn.LayerNorm(embed_dim)
        )
        
        # 注意力机制
        self.attention = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=8,
            dropout=0.1,
            batch_first=True
        )
        
        # 最终压缩层
        self.final_compression = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(embed_dim, embed_dim)  # 确保输出为384维
        )
        
        # 移到指定设备
        self.to(device)
    
    def forward(self, triplets):
        """
        前向传播
        Args:
            triplets: 三元组张量 [batch_size, 3] - [head, relation, tail]
        Returns:
            compressed: 压缩后的向量 [batch_size, 384]
        """
        batch_size = triplets.size(0)
        
        # 分离头实体、关系、尾实体
        heads = triplets[:, 0]      # [batch_size]
        relations = triplets[:, 1]  # [batch_size]
        tails = triplets[:, 2]      # [batch_size]
        
        # 获取嵌入向量 - 每个都是384维
        head_embeds = self.entity_embeddings(heads)      # [batch_size, 384]
        relation_embeds = self.relation_embeddings(relations)  # [batch_size, 384]
        tail_embeds = self.entity_embeddings(tails)      # [batch_size, 384]
        
        # 拼接三元组嵌入
        triplet_concat = torch.cat([head_embeds, relation_embeds, tail_embeds], dim=1)  # [batch_size, 1152]
        
        # 三元组融合
        fused = self.triplet_fusion(triplet_concat)  # [batch_size, 384]
        
        # 应用注意力机制
        # 重塑为序列形式用于注意力计算
        fused_seq = fused.unsqueeze(1)  # [batch_size, 1, 384]
        attended, _ = self.attention(fused_seq, fused_seq, fused_seq)  # [batch_size, 1, 384]
        attended = attended.squeeze(1)  # [batch_size, 384]
        
        # 残差连接
        attended = attended + fused
        
        # 最终压缩
        compressed = self.final_compression(attended)  # [batch_size, 384]
        
        # 验证输出维度
        assert compressed.size(1) == self.embed_dim, f"输出维度错误: {compressed.size(1)} != {self.embed_dim}"
        
        return compressed
    
    def compress_triplet(self, triplet_tensor):
        """
        兼容性方法 - 压缩单个三元组
        Args:
            triplet_tensor: 三元组张量
        Returns:
            compressed: 384维压缩向量
        """
        return self.forward(triplet_tensor)
    
    def get_entity_embedding(self, entity_id):
        """获取实体嵌入"""
        return self.entity_embeddings(torch.tensor(entity_id, device=self.device))
    
    def get_relation_embedding(self, relation_id):
        """获取关系嵌入"""
        return self.relation_embeddings(torch.tensor(relation_id, device=self.device))


class TensorNetworkLayer(nn.Module):
    """张量网络层 - 用于更复杂的三元组建模"""
    
    def __init__(self, embed_dim=384):
        super(TensorNetworkLayer, self).__init__()
        self.embed_dim = embed_dim
        
        # 张量核心
        self.tensor_core = nn.Parameter(torch.randn(embed_dim, embed_dim, embed_dim))
        
        # 线性变换层
        self.linear_transform = nn.Linear(embed_dim, embed_dim)
        
        # 初始化
        nn.init.xavier_uniform_(self.tensor_core)
        nn.init.xavier_uniform_(self.linear_transform.weight)
    
    def forward(self, head_embed, relation_embed, tail_embed):
        """
        张量网络前向传播
        Args:
            head_embed: 头实体嵌入 [batch_size, 384]
            relation_embed: 关系嵌入 [batch_size, 384]  
            tail_embed: 尾实体嵌入 [batch_size, 384]
        Returns:
            output: 张量网络输出 [batch_size, 384]
        """
        batch_size = head_embed.size(0)
        
        # 张量乘积运算
        # 计算 head * tensor_core * tail 
        temp = torch.einsum('bi,ijk,bj->bk', head_embed, self.tensor_core, tail_embed)
        
        # 与关系嵌入结合
        combined = temp * relation_embed
        
        # 线性变换
        output = self.linear_transform(combined)
        
        return output


def validate_dimensions():
    """验证维度设置是否正确"""
    print("开始验证TN模块维度设置...")
    
    # 创建测试实例
    compressor = TripleCompressor(embed_dim=384, num_entities=1000, num_relations=50)
    
    # 创建测试数据
    batch_size = 4
    test_triplets = torch.randint(0, 100, (batch_size, 3))
    
    # 前向传播
    with torch.no_grad():
        output = compressor(test_triplets)
    
    # 验证输出维度
    expected_shape = (batch_size, 384)
    actual_shape = output.shape
    
    assert actual_shape == expected_shape, f"维度不匹配: 期望{expected_shape}, 实际{actual_shape}"
    
    print(f"✓ 维度验证通过")
    print(f"  - 输入shape: {test_triplets.shape}")
    print(f"  - 输出shape: {output.shape}")
    print(f"  - 嵌入维度: {compressor.embed_dim}")
    
    return True


if __name__ == "__main__":
    # 运行维度验证
    validate_dimensions()
    print("TN模块配置完成，所有维度已对齐至384维")
