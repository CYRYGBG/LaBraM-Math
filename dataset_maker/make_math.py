# -*- coding: utf-8 -*-
# @Time     : 2025/10/29 12:40
# @Author   : Chen, Y.R.
# @File     : make_math.py
# @Software : PyCharm
# @Notes    : 批量处理 /usr/data/yeqi3/data_clean_easy/read 下的 sub_{i}_simplified.mat
#             预处理：带通(0.1-75Hz)+工频陷波(默认50Hz)+重采样至200Hz；按10秒窗切片(2000点)并保存为PKL

import os
import re
import gc
import sys
import math
import pickle
import argparse
from pathlib import Path
from fractions import Fraction

import numpy as np

# scipy.io.loadmat 读取 v5/v7.2；v7.3(HDF5) 用 h5py 兜底
from scipy.io import loadmat
try:
    import h5py
    _HAS_H5PY = True
except Exception:
    _HAS_H5PY = False

import mne
from scipy.signal import resample_poly

mne.set_log_level("WARNING")

# ========== 通道标准顺序（与原TUAB脚本保持一致；仅在名字能对上的情况下做交集重排） ==========
chOrder_standard = [
    'EEG FP1-REF', 'EEG FP2-REF', 'EEG F3-REF', 'EEG F4-REF', 'EEG C3-REF', 'EEG C4-REF',
    'EEG P3-REF', 'EEG P4-REF', 'EEG O1-REF', 'EEG O2-REF', 'EEG F7-REF', 'EEG F8-REF',
    'EEG T3-REF', 'EEG T4-REF', 'EEG T5-REF', 'EEG T6-REF', 'EEG A1-REF', 'EEG A2-REF',
    'EEG FZ-REF', 'EEG CZ-REF', 'EEG PZ-REF', 'EEG T1-REF', 'EEG T2-REF'
]

def now():
    """返回北京时间(UTC+8)字符串。"""
    from datetime import datetime, timedelta, timezone
    return (datetime.now(timezone.utc) + timedelta(hours=8)).strftime("%Y-%m-%d %H:%M:%S")


def _normalize_name(n: str) -> str:
    """将通道名规范化用于匹配：去 'EEG' / '-REF' / 空格，转大写。"""
    return str(n).replace('EEG', '').replace('-REF', '').replace(' ', '').upper()


def _load_mat_auto(mat_path: str):
    """
    自动加载 .mat：
    - 先用 scipy.io.loadmat（v5/v7.2）
    - 如失败且安装了 h5py，则尝试 v7.3 读取
    返回 dict，至少包含 eeg_data, labels, fsample；可包含 ch_names
    """
    try:
        d = loadmat(mat_path, squeeze_me=True, struct_as_record=False)
        return d
    except NotImplementedError:
        if not _HAS_H5PY:
            raise
        # 可能是 v7.3
        with h5py.File(mat_path, "r") as f:
            d = {}
            def _h5read(name):
                obj = f[name]
                arr = np.array(obj)
                # MATLAB cell/char 可能需要额外处理；尽量转为一维Python对象或字符串
                return arr
            for key in ['eeg_data', 'labels', 'fsample', 'ch_names', 'tmin']:
                if key in f:
                    d[key] = _h5read(key)
            return d


def _extract_fields(d: dict):
    """
    将原始 dict 中字段提取为：
    eeg: (E, C, T) float64
    labels: (E,) int 或能被 int() 的数
    ch_names: list[str] 或 []
    fs: float
    """
    # eeg_data
    if 'eeg_data' not in d:
        raise KeyError("mat文件缺少键 'eeg_data'")
    eeg = np.asarray(d['eeg_data'])
    if eeg.ndim != 3:
        raise ValueError(f"'eeg_data' 维度应为3，实际为 {eeg.shape}")

    # labels
    if 'labels' not in d:
        raise KeyError("mat文件缺少键 'labels'")
    labels = np.asarray(d['labels']).ravel()

    # fsample
    fs_key = 'fsample'
    if fs_key not in d:
        # 兼容可能的命名
        for alt in ('fs', 'sfreq', 'Fs', 'FSAMPLE'):
            if alt in d:
                fs_key = alt
                break
        if fs_key not in d:
            raise KeyError("mat文件缺少采样率键 'fsample'（或fs/sfreq）")
    fs_val = d[fs_key]
    try:
        fs = float(np.asarray(fs_val).item())
    except Exception:
        fs = float(np.asarray(fs_val).ravel()[0])

    # ch_names（可选）
    ch_names = []
    if 'ch_names' in d:
        raw = np.asarray(d['ch_names'])
        # 多种可能形态：list/ndarray(object)/二维char
        try:
            if raw.dtype.kind in ('U', 'S'):
                # 纯字符数组
                ch_names = [str(x) for x in raw.tolist()]
            else:
                # object/混合
                lst = raw.tolist()
                # 展平并转 str
                if isinstance(lst, (list, tuple)):
                    ch_names = [str(x) for x in lst]
                else:
                    ch_names = [str(lst)]
        except Exception:
            # 兜底
            try:
                ch_names = [str(x) for x in list(raw)]
            except Exception:
                ch_names = []
    return eeg.astype(np.float64, copy=False), labels, ch_names, fs


def _maybe_reorder_channels(eeg: np.ndarray, ch_names_mat, verbose_name: str):
    """
    按 chOrder_standard 与 ch_names_mat 的交集做重排。
    - eeg: (E, C, T)
    - ch_names_mat: list[str] or []
    返回：eeg(可能重排)、使用的通道名列表（重排后）
    """
    if not ch_names_mat or len(ch_names_mat) != eeg.shape[1]:
        # 无法重排，直接返回
        if not ch_names_mat:
            print(f"[warn {verbose_name}] 无 ch_names，跳过通道重排。")
        else:
            print(f"[warn {verbose_name}] ch_names 长度与数据不一致({len(ch_names_mat)} vs C={eeg.shape[1]})，跳过通道重排。")
        return eeg, ch_names_mat

    norm2idx = {_normalize_name(n): i for i, n in enumerate(ch_names_mat)}
    keep_idx = []
    kept_names = []
    for tgt in chOrder_standard:
        k = _normalize_name(tgt)
        if k in norm2idx:
            keep_idx.append(norm2idx[k])
            kept_names.append(ch_names_mat[norm2idx[k]])

    if len(keep_idx) == 0:
        print(f"[warn {verbose_name}] 与标准通道无交集，保持原顺序。")
        return eeg, ch_names_mat

    eeg2 = eeg[:, keep_idx, :]
    return eeg2, kept_names


def _filter_notch_resample(x: np.ndarray, fs: float, band_low: float, band_high: float, notch_freq: float, out_fs: int) -> np.ndarray:
    """
    对单个epoch的 (C, T) 数据执行：带通 -> 陷波 -> 重采样。
    使用 mne.filter 直接对 numpy 数组滤波，scipy.signal.resample_poly 重采样。
    """
    # 带通
    # 在你的 _filter_notch_resample 中替换
    x = mne.filter.filter_data(
        x,
        sfreq=fs,
        l_freq=band_low,
        h_freq=band_high,
        method='iir',            # 改为 IIR
        iir_params=dict(order=4, ftype='butter'),
        verbose=False
    )

    # 陷波（工频）
    if notch_freq and notch_freq > 0:
        x = mne.filter.notch_filter(x, Fs=fs, freqs=[notch_freq], verbose=False)

    # 重采样
    frac = Fraction(int(round(out_fs)), int(round(fs))).limit_denominator()
    up, down = frac.numerator, frac.denominator
    if abs(out_fs - fs * up / down) > 1e-6:
        # 说明不是严格有理数；直接用 resample_poly 近似
        x = resample_poly(x, up=out_fs, down=int(round(fs)), axis=-1)
    else:
        x = resample_poly(x, up=up, down=down, axis=-1)
    return x


def process_mat_file(mat_path: str,
                     dump_folder: str,
                     band=(0.1, 75.0),
                     notch=50.0,
                     out_fs=200,
                     window_sec=10,
                     reorder=True):
    """
    读取单个 .mat 文件并输出切片 PKL。
    """
    os.makedirs(dump_folder, exist_ok=True)
    base = os.path.splitext(os.path.basename(mat_path))[0]
    verbose_name = base

    try:
        d = _load_mat_auto(mat_path)
        eeg, labels, ch_names, fs = _extract_fields(d)
    except Exception as e:
        errfile = os.path.join(dump_folder, "process_error_files.txt")
        with open(errfile, "a", encoding="utf-8") as f:
            f.write(f"{Path(mat_path).name}\tLOAD_FAIL\t{repr(e)}\n")
        print(f"[{now()}] [ERROR] 读取失败: {mat_path} -> {e}")
        return 0

    # （可选）重排到标准通道交集
    if reorder:
        try:
            eeg, ch_names = _maybe_reorder_channels(eeg, ch_names, verbose_name)
        except Exception as e:
            print(f"[warn {verbose_name}] 通道重排失败，保持原顺序。err={e}")

    E, C, T = eeg.shape

    saved = 0
    for ep in range(E):
        x = np.asarray(eeg[ep], dtype=np.float64)  # (C, T)
        try:
            x = _filter_notch_resample(x, fs=fs,
                                    band_low=band[0], band_high=band[1],
                                    notch_freq=notch, out_fs=out_fs)
        except Exception as e:
            errfile = os.path.join(dump_folder, "process_error_files.txt")
            with open(errfile, "a", encoding="utf-8") as f:
                f.write(f"{Path(mat_path).name}\tFILTER_FAIL(ep={ep})\t{repr(e)}\n")
            print(f"[{now()}] [warn {verbose_name}] 滤波/重采样失败 @ ep={ep}: {e}")
            continue

        try:
            y = int(np.asarray(labels[ep]).item())
        except Exception:
            try:
                y = int(labels[ep])
            except Exception:
                y = labels[ep]

        # 整条 epoch 作为一个样本保存
        if np.all(np.isfinite(x)):
            out_name = f"{base}_ep{ep:04d}.pkl"
            out_path = os.path.join(dump_folder, out_name)
            with open(out_path, "wb") as f:
                pickle.dump({"X": x, "y": y, "fs": out_fs if out_fs is not None else fs}, f)
            saved += 1

    return saved



def _gather_subject_mats(root_dir: str):
    """
    搜索 root_dir 下形如 sub_{i}_simplified.mat 的文件，按数字序排序返回路径列表。
    """
    root = Path(root_dir)
    mats = list(root.glob("sub_*_simplified.mat"))
    def _key(p: Path):
        m = re.search(r"sub_(\d+)_simplified\.mat", p.name, flags=re.I)
        return int(m.group(1)) if m else math.inf
    mats.sort(key=_key)
    return [str(p) for p in mats]


def main():
    # ============= 参数配置区（按需手动修改） =============
    BAND_LOW    = 0.1
    BAND_HIGH   = 75.0
    NOTCH       = 50.0
    OUT_FS      = 200
    WINDOW_SEC  = None
    REORDER     = False  # 先保留你的 64 通道，不做裁剪；若要统一顺序，改 True 并替换 chOrder_standard
    WORKERS_MAX = 1     # 你允许的最大并行数（会与CPU核数、文件数取 min）

    BASE_DATA_PATH = '/usr/data/yeqi3/data_clean_easy'
    BASE_OUT_PATH  = '/usr/data/yeqi3/labram_processed'

    DATA_PATHS = [
        # os.path.join(BASE_DATA_PATH, 'read'),
        os.path.join(BASE_DATA_PATH, 'type'),
        # os.path.join(BASE_DATA_PATH, 'read_new'),
        os.path.join(BASE_DATA_PATH, 'type_new'),
    ]
    OUT_PATHS = [
        # os.path.join(BASE_OUT_PATH, 'read'),
        os.path.join(BASE_OUT_PATH, 'type'),
        # os.path.join(BASE_OUT_PATH, 'read_new'),
        os.path.join(BASE_OUT_PATH, 'type_new'),
    ]
    # ====================================================

    for dataset_dir, out_dir in zip(DATA_PATHS, OUT_PATHS):
        os.makedirs(out_dir, exist_ok=True)

        mats = _gather_subject_mats(dataset_dir)
        if not mats:
            print(f"[{now()}] [WARN] 未找到 {dataset_dir} 下的 sub_*_simplified.mat 文件。")
            continue

        # 动态并行度：不超过 CPU 核心数/文件数/WORKERS_MAX
        cpu_n = os.cpu_count() or 1
        workers = min(len(mats), cpu_n, WORKERS_MAX)

        print("=" * 60)
        print(f"Start Time : {now()}")
        print(f"Dataset Dir: {dataset_dir}")
        print(f"Out Dir    : {out_dir}")
        print(f"Files      : {len(mats)}")
        print(f"Filter     : {BAND_LOW}-{BAND_HIGH} Hz, Notch={NOTCH} Hz")
        if WINDOW_SEC and WINDOW_SEC > 0:
            print(f"Resample   : {OUT_FS} Hz, Window={WINDOW_SEC}s (= {int(OUT_FS * WINDOW_SEC)} pts)")
        else:
            print(f"Resample   : {OUT_FS} Hz, Window=None (save whole epoch)")

        print(f"Reorder    : {REORDER}")
        print(f"Workers    : {workers}")
        print("=" * 60)

        total_saved = 0

        if workers > 1:
            from multiprocessing import Pool
            pack = [(p, out_dir, (BAND_LOW, BAND_HIGH), NOTCH, OUT_FS, WINDOW_SEC, REORDER) for p in mats]

            def _worker(tup):
                p, out_dir, band, notch, out_fs, win_sec, reorder = tup
                return process_mat_file(p, out_dir, band=band, notch=notch, out_fs=out_fs,
                                        window_sec=win_sec, reorder=reorder)

            with Pool(processes=workers) as pool:
                for saved in pool.imap_unordered(_worker, pack):
                    total_saved += int(saved)
        else:
            for p in mats:
                print(f"[{now()}] Processing: {Path(p).name}")
                saved = process_mat_file(
                    p, out_dir,
                    band=(BAND_LOW, BAND_HIGH),
                    notch=NOTCH,
                    out_fs=OUT_FS,
                    window_sec=WINDOW_SEC,
                    reorder=REORDER
                )
                total_saved += int(saved)
                gc.collect()

        print("-" * 60)
        print(f"Done Time  : {now()}")
        print(f"Total PKLs : {total_saved}")
        print(f"Saved To   : {out_dir}")
        print("-" * 60)



if __name__ == "__main__":
    main()

