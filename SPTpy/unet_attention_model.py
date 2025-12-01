#!usr/bin/env python
# -*- coding:utf-8 _*-
"""
@author:Liao Shasha
@file: unet_attention_model.py
@institute:SIAT
@location:Shenzhen,China
@time: 2025/07/26
"""
import os
import time
import glob
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

import albumentations as A
from albumentations.pytorch import ToTensorV2
from PIL import Image
import cv2
# ----------

# ----------------------------
# 自定义 Dataset：同时读取三张图 + 一个 mask
# ----------------------------
class DualInputDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        # 找到所有原始 png
        self.base_names = []
        for p in glob.glob(os.path.join(data_dir, "*.png")):
            if p.endswith("_pos1.png") or p.endswith("_scatter_plot.png"):
                continue
            base = os.path.splitext(os.path.basename(p))[0]
            # 检查对应文件都存在
            rend = os.path.join(data_dir, f"{base}.png")
            scat = os.path.join(data_dir, f"{base}_scatter_plot.png")
            mask = os.path.join(data_dir, f"{base}_pos1_seg.npy")
            if os.path.exists(rend) and os.path.exists(scat) and os.path.exists(mask):
                self.base_names.append(base)
            else:
                print(f"[WARN] skip {base}, missing one of render/scatter/seg")
        self.data_dir  = data_dir
        self.transform = transform

    def __len__(self):
        return len(self.base_names)

    def __getitem__(self, idx):
        base = self.base_names[idx]

        # 1) 读三张灰度图
        img_pil = Image.open(os.path.join(self.data_dir, f"{base}_pos1.png")).convert("L")
        render_pil = Image.open(os.path.join(self.data_dir, f"{base}.png")).convert("L")
        scatter_pil = Image.open(os.path.join(self.data_dir, f"{base}_scatter_plot.png")).convert("L")

        # 2) 全部 resize 到 (320,320)
        target_size = (320, 320)  # (W, H)
        img_pil = img_pil.resize(target_size, resample=Image.BILINEAR)
        render_pil = render_pil.resize(target_size, resample=Image.BILINEAR)
        scatter_pil = scatter_pil.resize(target_size, resample=Image.BILINEAR)

        # 3) 转 numpy
        img = np.array(img_pil, dtype=np.float32)
        render = np.array(render_pil, dtype=np.float32)
        scatter = np.array(scatter_pil, dtype=np.float32)

        # 4) 堆叠成 (H, W, 3)
        stacked = np.stack([img, render, scatter], axis=-1)

        # 5) 读 mask，并“拆包”成纯数值 ndarray
        mask_path = os.path.join(self.data_dir, f"{base}_pos1_seg.npy")
        data = np.load(mask_path, allow_pickle=True)

        if hasattr(data, "files"):
            # .npz
            key = data.files[0]
            mask_arr = data[key]
        elif isinstance(data, np.ndarray) and data.dtype == object:
            obj = data.item()
            mask_arr = next(iter(obj.values())) if isinstance(obj, dict) else obj
        elif isinstance(data, dict):
            mask_arr = next(iter(data.values()))
        elif isinstance(data, np.ndarray):
            mask_arr = data
        else:
            raise ValueError(f"不支持的 mask 类型: {type(data)}")

        # 二值化、填充、转 float32
        mask_bin = (np.asarray(mask_arr) > 0.5).astype(np.uint8)
        contours, _ = cv2.findContours(mask_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        mask_filled = np.zeros_like(mask_bin)
        cv2.fillPoly(mask_filled, pts=contours, color=1)
        mask = mask_filled.astype(np.float32)

        # 6) Albumentations 增强 & 转 Tensor
        if self.transform:
            aug = self.transform(image=stacked, mask=mask)
            x = aug["image"]  # (3, H, W)
            y = aug["mask"].unsqueeze(0)  # (1, H, W)
        else:
            x = torch.from_numpy(stacked).permute(2, 0, 1)  # (3, H, W)
            y = torch.from_numpy(mask).unsqueeze(0)  # (1, H, W)

        return x, y


# ----------------------------------
# 基础模块：DoubleConv / Down / Up
# ----------------------------------
class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
    def forward(self, x):
        return self.net(x)

class Down(nn.Module):
    """ 下采样：MaxPool2d + DoubleConv """
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_ch, out_ch)
        )
    def forward(self, x):
        return self.net(x)

class Up(nn.Module):
    """
    上采样 + 双卷积
    in_ch: 上一级（解码器或 bottleneck）的通道数
    out_ch: 本级输出通道数（解码器路径上的特征通道数）
    """
    def __init__(self, in_ch, out_ch):
        super().__init__()
        # transposed conv: 从 in_ch -> out_ch
        self.up   = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2)
        # fusion 后的 double conv: skip 通道数 和 up 通道数 都是 out_ch
        self.conv = DoubleConv(out_ch * 2, out_ch)

    def forward(self, x1, x2):
        # x1: 来自上一级解码器的特征 (B, in_ch, H, W)
        # x2: 来自跳跃连接的融合特征 (B, out_ch, 2H, 2W)
        x1 = self.up(x1)  # -> (B, out_ch, 2H, 2W)

        # 对齐尺寸（防止奇数倍下采样带来的 mismatched）
        diffY = x2.size(2) - x1.size(2)
        diffX = x2.size(3) - x1.size(3)
        x1 = F.pad(x1, [diffX//2, diffX-diffX//2, diffY//2, diffY-diffY//2])

        # 拼接并双卷积
        x = torch.cat([ x2, x1 ], dim=1)  # (B, 2*out_ch, 2H, 2W)
        return self.conv(x)              # (B, out_ch, 2H, 2W)


# ----------------------------------
# 注意力门：Cross-Stream Attention
# ----------------------------------
class AttentionGate(nn.Module):
    """
    g: gating signal (来自灰度分支)
    x: skip 特征 (来自 render+scatter 分支)
    """
    def __init__(self, F_g, F_l, F_int):
        super().__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, bias=True),
            nn.BatchNorm2d(F_int)
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, bias=True),
            nn.BatchNorm2d(F_int)
        )
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        # 对齐空间尺寸
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        # 相加后 ReLU，再 1x1Conv + Sigmoid 得到 attention map
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        # 用 attention map 乘以原分支特征
        return x * psi

# ----------------------------------
# 多级融合 + 注意力门 双流 UNet
# ----------------------------------
class MultiFusionAttentionUNet(nn.Module):
    def __init__(self):
        super().__init__()
        # ——— 双编码器 ———
        # 灰度分支（1 通道）
        self.incG   = DoubleConv(1,  32)
        self.downG1 = Down(32,  64)
        self.downG2 = Down(64, 128)
        self.downG3 = Down(128,256)
        self.downG4 = Down(256,512)
        # render+scatter 分支（2 通道）
        self.incR   = DoubleConv(2,  32)
        self.downR1 = Down(32,  64)
        self.downR2 = Down(64, 128)
        self.downR3 = Down(128,256)
        self.downR4 = Down(256,512)

        # ——— 注意力门 & 融合卷积 ———
        self.att1  = AttentionGate(F_g=32,  F_l=32,  F_int=16)
        self.fuse1 = DoubleConv(32+32, 32)
        self.att2  = AttentionGate(F_g=64,  F_l=64,  F_int=32)
        self.fuse2 = DoubleConv(64+64, 64)
        self.att3  = AttentionGate(F_g=128, F_l=128, F_int=64)
        self.fuse3 = DoubleConv(128+128,128)
        self.att4  = AttentionGate(F_g=256, F_l=256, F_int=128)
        self.fuse4 = DoubleConv(256+256,256)
        self.attB  = AttentionGate(F_g=512, F_l=512, F_int=256)
        self.fuseB = DoubleConv(512+512,512)

        # ——— 解码器 ———
        self.up1 = Up(512,256)
        self.up2 = Up(256,128)
        self.up3 = Up(128, 64)
        self.up4 = Up(64,  32)
        self.outc = nn.Conv2d(32, 1, kernel_size=1)

    def forward(self, x):
        # x: (B,3,H,W) -> split 成灰度 & render+scatter
        xG = x[:,0:1]   # 灰度分支
        xR = x[:,1:3]   # render+scatter 分支

        # ——— 编码 & 多级融合 ———
        # level1
        x1G = self.incG(xG)
        x1R = self.incR(xR)
        x1R_att = self.att1(x1G, x1R)
        f1 = self.fuse1(torch.cat([x1G, x1R_att], dim=1))

        # level2
        x2G = self.downG1(x1G)
        x2R = self.downR1(x1R)
        x2R_att = self.att2(x2G, x2R)
        f2 = self.fuse2(torch.cat([x2G, x2R_att], dim=1))

        # level3
        x3G = self.downG2(x2G)
        x3R = self.downR2(x2R)
        x3R_att = self.att3(x3G, x3R)
        f3 = self.fuse3(torch.cat([x3G, x3R_att], dim=1))

        # level4
        x4G = self.downG3(x3G)
        x4R = self.downR3(x3R)
        x4R_att = self.att4(x4G, x4R)
        f4 = self.fuse4(torch.cat([x4G, x4R_att], dim=1))

        # bottleneck
        x5G = self.downG4(x4G)
        x5R = self.downR4(x4R)
        x5R_att = self.attB(x5G, x5R)
        f5 = self.fuseB(torch.cat([x5G, x5R_att], dim=1))

        # ——— 解码 & 跳跃连接 ———
        u1 = self.up1(f5, f4)
        u2 = self.up2(u1, f3)
        u3 = self.up3(u2, f2)
        u4 = self.up4(u3, f1)

        out = self.outc(u4)
        return out


# ----------------------------
# 定义数据增强流水线
# 注意 additional_targets 告诉 Albumentations 我们有多张要一起增强的“伪 image”
# ----------------------------
train_transform = A.Compose([
    A.Resize(320,320),
    A.RandomRotate90(p=0.5),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.ElasticTransform(p=0.2, alpha=1, sigma=50),
    A.GridDistortion(p=0.2),
    A.RandomCrop(width=256, height=256),
    A.Normalize(mean=(0.0,0.0,0.0), std=(1.0,1.0,1.0)),
    ToTensorV2()
])


val_transform = A.Compose([
    # A.CenterCrop(width=256, height=256),
    A.Normalize(mean=(0.0,0.0,0.0), std=(1.0,1.0,1.0)),
    ToTensorV2()
])
