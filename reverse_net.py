import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from DistributionMixture import *

class PatchEmbedding(nn.Module):
    def __init__(self, sequence_length, patch_size, embed_dim):
        super(PatchEmbedding, self).__init__()
        self.patch_size = patch_size
        self.num_patches = sequence_length // patch_size
        self.embed_dim = embed_dim
        self.proj = nn.Linear(patch_size, embed_dim)
    
    def forward(self, x):
        batch_size, seq_len = x.shape
        if seq_len < self.patch_size * self.num_patches:
            padding_size = self.patch_size * self.num_patches - seq_len
            x = F.pad(x, (0., padding_size)) #-> (batch_size, num_patches*patch_size)
        x = x.reshape(batch_size, self.num_patches, self.patch_size)  #-> (batch_size, num_patches, patch_size)
        x = self.proj(x)  #-> (batch_size, num_patches, embed_dim)
        return x

class VisionTransformer(nn.Module):
    def __init__(self, 
                 sequence_length, 
                 patch_size=30, 
                 embed_dim=128, 
                 depth=6, 
                 num_heads=8, 
                 mlp_dim=256,
                 dropout=0.1, 
                 num_distributions=4,
                 num_params=3 
                 ):
        super(VisionTransformer, self).__init__()
        self.num_patches = sequence_length // patch_size
        self.patch_embed = PatchEmbedding(sequence_length, patch_size, embed_dim)
        
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, embed_dim))
        self.pos_drop = nn.Dropout(p=dropout)
        
        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(embed_dim, num_heads, mlp_dim, dropout, batch_first=True),
            num_layers=depth
        )
        
        self.norm = nn.LayerNorm(embed_dim)

        self.num_distributions = num_distributions
        self.num_params = num_params
        self.fc = nn.Linear(embed_dim, self.num_distributions * self.num_params)  # 输出 k 组 [权重, r, p] 参数
    
    def forward(self, x): #<-(batch_size, sequence_length)
        x = self.patch_embed(x)  #-> (batch_size, num_patches, embed_dim)
        x = x + self.pos_embed  # (batch_size, num_patches, embed_dim)
        x = self.pos_drop(x)
        x = self.transformer_encoder(x)  # (batch_size, num_patches, embed_dim)
        x = self.norm(x)
        x = x.mean(dim=1)  # 全局平均池化，得到 (batch_size, embed_dim)
        
        params = self.fc(x)  # (batch_size, k * 3)
        params = params.view(-1, self.num_distributions, self.num_params)  #-> (batch_size, k, 3)
        return params
    
class DistributionEncoder_NegativeBinomial(nn.Module):
    def __init__(self, 
                 sequence_length, 
                 patch_size=30, 
                 embed_dim=64, 
                 depth=6, 
                 num_heads=4, 
                 mlp_dim=64,
                 dropout=0.1, 
                 num_distributions=4) -> None:
        super().__init__()
        self.vit = VisionTransformer(sequence_length, 
                                     patch_size, 
                                     embed_dim, 
                                     depth, 
                                     num_heads,
                                     mlp_dim,
                                     dropout,
                                     num_distributions,
                                     num_params=3)
        self.weights_activation = nn.Softmax(dim=-1)
        self.r_activation = nn.Softplus()
        self.p_activation = nn.Sigmoid()

    def forward(self, probe_seq): #
        params = self.vit(probe_seq)
        w = self.weights_activation(params[:, :, 0])  # 归一化权重，使其在 (0, 1) 之间且总和为 1
        r = self.r_activation(params[:, :, 1])  # 确保 r 为正值
        p = self.p_activation(params[:, :, 2])  # 确保 p 在 (0, 1) 之间
        return w, r, p #-> w, r, p 均为 (batch_size, num_distributions)
    
class DistributionDecoder_NegativeBinomial(nn.Module):
    def __init__(self, distributions_num, output_size):
        super(DistributionDecoder_NegativeBinomial, self).__init__()
        self.fc1 = nn.Linear(distributions_num * 3, 128)  # 输入层，假设隐藏层大小为128
        self.fc2 = nn.Linear(128, 64)     # 隐藏层
        self.fc3 = nn.Linear(64, output_size)       # 输出层
        
    def forward(self, w, p, r):
        # 将三个输入连接起来
        x = torch.cat((w, p, r), dim=1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

