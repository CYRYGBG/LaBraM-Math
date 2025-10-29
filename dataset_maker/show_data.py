# -*- coding: utf-8 -*-
# @Time    : 2025/10/29 14:20:58
# @Author  : Chen, Y.R.
# @File    : show_data.py
# @Software: VSCode
import os
import pickle
import random
import numpy as np

# ====== 手动修改你的数据路径 ======
DATA_DIR = "/usr/data/yeqi3/labram_processed/read"   # 改成你想查看的子目录
N_SHOW = 5                                           # 随机展示几个样本
# ==================================

# 获取所有 pkl 文件路径
all_pkl = [os.path.join(DATA_DIR, f) for f in os.listdir(DATA_DIR) if f.endswith(".pkl")]
print(f"✅ 在 {DATA_DIR} 下共找到 {len(all_pkl)} 个样本文件。")

if len(all_pkl) == 0:
    raise SystemExit("未找到任何 .pkl 文件，请检查路径或运行状态。")

# 随机抽样查看
sample_files = random.sample(all_pkl, min(N_SHOW, len(all_pkl)))

for path in sample_files:
    with open(path, "rb") as f:
        data = pickle.load(f)
    
    X = np.array(data["X"])
    y = data["y"]
    
    print(f"\n📂 文件: {os.path.basename(path)}")
    print(f" - X.shape : {X.shape}   (通道数 × 时间点数)")
    print(f" - y        : {y}")
    print(f" - 均值/方差: mean={X.mean():.3f}, std={X.std():.3f}")
    print(f" - 时间长度 : {X.shape[1]/200:.1f}s  (假设重采样到200Hz)")
    print(f" - 数值范围 : [{X.min():.2f}, {X.max():.2f}] µV")

print("\n✅ 检查完毕。每个样本都是一个独立的10秒EEG片段 + 对应标签。")
