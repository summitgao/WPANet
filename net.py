import os
import math
import torch
import numbers
import math
import torch.nn as nn
import scipy.io as sio
from skimage import io
import torch.optim as optim
from operator import truediv
from einops import rearrange 
import torch.nn.functional as F
from torch_wavelets import DWT_2D, IDWT_2D
import parameter

parameter._init()

# 残差单元
class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        # Resudual connect: fn(x) + x
        return self.fn(x, **kwargs) + x

# 层归一化
class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        # using Layer Normalization before input to fn layer
        return self.fn(self.norm(x), **kwargs)

# 前馈网络
class FeedForward(nn.Module):
    # Feed Forward Neural Network
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        # Two linear network with GELU and Dropout
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)

# 小波特征提取注意力
class WaveAttention(nn.Module):
    def __init__(self, dim, heads, dim_head, sr_ratio=1, dropout=0.):
        super().__init__()
        self.dim = dim # (输入维度) 512
        self.num_heads = heads # (heads数) 4
        self.dim_head = dim_head # (heads维度) 128
        self.inner_dim = self.dim_head * self.num_heads # =128*4=512
        self.scale = self.dim_head ** -0.5
        self.sr_ratio = sr_ratio
        self.dwt = DWT_2D(wave='haar')
        self.idwt = IDWT_2D(wave='haar')
        self.reduce = nn.Sequential(
            nn.Conv2d(self.inner_dim, self.inner_dim//4, kernel_size=1, padding=0, stride=1),
            nn.BatchNorm2d(self.inner_dim//4),
            nn.ReLU(inplace=True),
        )
        self.filter = nn.Sequential(
            nn.Conv2d(self.inner_dim, self.inner_dim, kernel_size=3, padding=1, stride=1, groups=1),
            nn.BatchNorm2d(self.inner_dim),
            nn.ReLU(inplace=True),
        )
        self.kv_embed = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio) if sr_ratio > 1 else nn.Identity()
        self.q = nn.Linear(dim, self.inner_dim, bias=False)
        self.kv = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, self.inner_dim, bias=False)
        )
        self.proj = nn.Linear(self.inner_dim + self.inner_dim // 4, dim)

    def forward(self, x):
        B, N, C = x.shape
        H = W = int(math.sqrt(N))
        q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        x = x.view(B, H, W, C).permute(0, 3, 1, 2)
        x_dwt = self.dwt(self.reduce(x))
        x_dwt = self.filter(x_dwt)
        x_idwt = self.idwt(x_dwt)
        x_idwt = x_idwt.view(B, -1, x_idwt.size(-2)*x_idwt.size(-1)).transpose(1, 2)
        kv = self.kv_embed(x_dwt).reshape(B, C, -1).permute(0, 2, 1)
        kv = self.kv(kv).reshape(B, -1, int(H/2), self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(torch.cat([x, x_idwt], dim = -1))
        return x

# 小波特征提取Transformer
class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout, type):
        super().__init__()
        if type == 0: 
            self.layers = nn.ModuleList([])
            for _ in range(depth):
                self.layers.append(nn.ModuleList([
                    Residual(PreNorm(dim, WaveAttention(dim, heads=heads, dim_head=dim_head, dropout=dropout))),
                    Residual(PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout)))
                ]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x)
            x = ff(x)
        return x

# 小波特征提取模块
class waveBlock(nn.Module):
    def __init__(self, *, image_size, patch_size, dim, depth, heads, mlp_dim,
                channels, dim_head, dropout=0., emb_dropout=0.):
        super().__init__()
        # assert image_size % patch_size == 0, 'Image dimensions must be divisible by the patch size.'
        self.num_patches = (image_size // patch_size) ** 2
        self.patch_dim = channels * patch_size ** 2
        self.image_size = image_size
        self.patch_size = patch_size
        self.pos = nn.Parameter(torch.randn(1, self.num_patches, dim))
        self.to_embedding = nn.Linear(self.patch_dim, dim)
        self.dropout = nn.Dropout(emb_dropout)
        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout, 0)
        self.embedding_to = nn.Linear(dim, self.patch_dim)
        
    def forward(self, x):
        p = self.patch_size 
        if(len(x.shape) == 4):
            b, c, h, w = x.shape
            hh = int(h / p)
            embed = rearrange(x, 'b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = p, p2 = p)
            embed = self.to_embedding(embed)
            b, n, c = embed.shape
            embed += self.pos[:, :n]
            embed = self.dropout(embed)
            embed = self.transformer(embed)
            x = self.embedding_to(embed)
            x = rearrange(x, 'b (h w) (p1 p2 c) -> b c (h p1) (w p2)', h = hh, p1 = p, p2 = p)
        if(len(x.shape) == 5):
            b, c1, c2, h, w = x.shape
            hh = int(h / p)
            embed = rearrange(x, 'b c1 c2 (h p1) (w p2) -> b (h w) (p1 p2 c1 c2)', p1 = p, p2 = p)
            embed = self.to_embedding(embed)
            b, n, c = embed.shape
            embed += self.pos[:, :n]
            embed = self.dropout(embed)
            embed = self.transformer(embed)
            x = self.embedding_to(embed)
            x = rearrange(x, 'b (h w) (p1 p2 c1 c2) -> b c1 c2 (h p1) (w p2)', h = hh, p1 = p, p2 = p, c1 = c1, c2 = c2)
        return x

# 平行注意力特征融合模块
class CDFBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.filter = out_channels
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(self.filter, self.filter // 16, kernel_size = 1),
            nn.ReLU(),
            nn.Conv2d(self.filter // 16, self.filter, kernel_size = 1),
            nn.Sigmoid()
        )
        self.conv2D = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size = 3, padding = 0),
            nn.BatchNorm2d(num_features=out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, hsi, sar):
        sar = self.conv2D(sar)
        jc = hsi * sar
        jc = self.se(jc)
        jd = torch.abs(jc * hsi - jc * sar)
        ja = jc * hsi + jc * sar
        jf = ja + jd
        return hsi + jf, sar + jf

# WPANet
class WPANet(nn.Module):
    def __init__(self, out_features):
        super(WPANet, self).__init__()
        depth = parameter.get_value('depth')
        self.out_features = out_features
        self.hsi_conv1 = nn.Sequential(
            nn.Conv2d(in_channels=30, out_channels=64, kernel_size=3, padding=0), 
            nn.BatchNorm2d(num_features=64), 
            nn.ReLU(inplace=True)
        )
        self.hsi_conv2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=0), 
            nn.BatchNorm2d(num_features=128), 
            nn.ReLU(inplace=True)
        )
        self.hsi_conv3 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=0), 
            nn.BatchNorm2d(num_features=256), 
            nn.ReLU(inplace=True)
        )
        self.WaveBlock1 = waveBlock(image_size=8, patch_size=1, dim=512, depth=depth[1][0], heads=8, mlp_dim=1024, channels=64, dim_head=64, dropout=0., emb_dropout=0)
        self.WaveBlock2 = waveBlock(image_size=6, patch_size=1, dim=512, depth=depth[1][1], heads=8, mlp_dim=1024, channels=128, dim_head=64, dropout=0., emb_dropout=0)
        self.WaveBlock3 = waveBlock(image_size=4, patch_size=1, dim=512, depth=depth[1][2], heads=8, mlp_dim=1024, channels=256, dim_head=64, dropout=0., emb_dropout=0)
        self.CDFBlock1 = CDFBlock(4, 64)
        self.CDFBlock2 = CDFBlock(64, 128)
        self.CDFBlock3 = CDFBlock(128, 256)
        self.drop_hsi = nn.Dropout(0.6)
        self.drop_sar = nn.Dropout(0.6)
        self.drop_fusion = nn.Dropout(0.6)
        self.fusionlinear_hsi = nn.Linear(in_features=1024, out_features = self.out_features)
        self.fusionlinear_sar = nn.Linear(in_features=1024, out_features = self.out_features)
        self.fusionlinear_fusion = nn.Linear(in_features=2048, out_features = self.out_features)
        self.weight = nn.Parameter(torch.ones(2))

    def forward(self, hsi, sar):
        hsi = hsi.reshape(-1, hsi.shape[1] * hsi.shape[2], hsi.shape[3], hsi.shape[4])
        hsi_feat1 = self.hsi_conv1(hsi)
        hsi_feat1 = hsi_feat1 + self.WaveBlock1(hsi_feat1)
        hsi_feat1, sar_feat1 = self.CDFBlock1(hsi_feat1, sar)
        hsi_feat2 = self.hsi_conv2(hsi_feat1)
        hsi_feat2 = hsi_feat2 + self.WaveBlock2(hsi_feat2)
        hsi_feat2, sar_feat2 = self.CDFBlock2(hsi_feat2, sar_feat1)
        hsi_feat3 = self.hsi_conv3(hsi_feat2)
        hsi_feat3 = hsi_feat3 + self.WaveBlock3(hsi_feat3)
        hsi_feat3, sar_feat3 = self.CDFBlock3(hsi_feat3, sar_feat2)
        hsi_feat4 = hsi_feat3.reshape(-1, hsi_feat3.shape[1], hsi_feat3.shape[2]*hsi_feat3.shape[3])
        sar_feat4 = sar_feat3.reshape(-1, sar_feat3.shape[1], sar_feat3.shape[2]*sar_feat3.shape[3])
        fusion_feat = torch.cat((hsi_feat4, sar_feat4), dim=1)
        hsi_feat = F.max_pool1d(hsi_feat4, kernel_size=4)
        hsi_feat = hsi_feat.reshape(-1, hsi_feat.shape[1] * hsi_feat.shape[2])
        sar_feat = F.max_pool1d(sar_feat4, kernel_size=4)
        sar_feat = sar_feat.reshape(-1, sar_feat.shape[1] * sar_feat.shape[2])
        fusion_feat = F.max_pool1d(fusion_feat, kernel_size=4)
        fusion_feat = fusion_feat.reshape(-1, fusion_feat.shape[1] * fusion_feat.shape[2])
        hsi_feat = self.drop_hsi(hsi_feat)
        sar_feat = self.drop_sar(sar_feat)
        fusion_feat = self.drop_fusion(fusion_feat)
        output_hsi = self.fusionlinear_hsi(hsi_feat)
        output_sar = self.fusionlinear_sar(sar_feat)
        output_fusion = self.fusionlinear_fusion(fusion_feat)
        weights = torch.sigmoid(self.weight)
        outputs = weights[0] * output_hsi + weights[1] * output_sar + output_fusion
        return outputs