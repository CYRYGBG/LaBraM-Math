# -*- coding: utf-8 -*-
# @Time     : 2025/10/29 16:54  # 北京时间（UTC+8）
# @Author   : Chen, Y.R.
# @File     : SplitMakeMathToKFold.py
# @Software : PyCharm
# @Notes    : 基于 make_math 已保存好的样本文件，按“被试内 5 折 + (test=k, eval=k+1)”方案重组目录：
#             OUT_ROOT/fold_{k}/{train,eval,test}/ 直接放入原样本文件（硬链接优先，失败则复制）
#             同时输出 split_map.json（全局折分记录）及各 fold 的 manifest.csv（训练可直接读取）

import os
import re
import gc
import csv
import json
import math
import shutil
from pathlib import Path
from typing import Dict, List, Any, Tuple

import numpy as np

# 尝试可选依赖
try:
    import torch
except Exception:
    torch = None

# =========================
# ======== 配置区 =========
# =========================
IN_ROOT = Path("/path/to/make_math/output")   # ← 改成 make_math 生成结果的根目录
OUT_ROOT = Path("/path/to/output/kfold")      # ← 新的五折输出根目录
FOLDS = 5
SEED = 42
SHUFFLE = True
USE_STRATIFIED = True              # 如果能拿到 label 就使用 StratifiedKFold，否则自动回退 KFold
PRESERVE_TREE = False              # True 则在新目录中保留原有相对路径层级；False 则平铺在 split 目录下
PREFER_HARDLINK = True             # 尽量硬链接（节省空间），失败再复制
OVERWRITE_OUTPUT = True            # True 会在创建前清空 OUT_ROOT
# 受支持的输入扩展名
FILE_EXTS = {".npz", ".npy", ".pkl", ".pt"}
# 读取样本元信息时尝试的字段名（按优先级）
SUBJECT_KEYS = ["subject", "sid", "subject_id", "sub_id", "S", "SUBJECT"]
LABEL_KEYS   = ["label", "y", "target", "cls", "LABEL"]
# 从路径推断被试 ID 的正则（按顺序尝试），捕获组(1)为被试 ID
SUBJECT_PATTERNS = [
    r"(?:^|/)(sub[_-]?\d+)(?:/|_|\.)",         # .../sub_01/..., .../sub01_xxx.npz
    r"(?:^|/)(S\d+)(?:/|_|\.)",                # .../S1/..., .../S12_xxx.pt
    r"(?:^|/)(subject[_-]?\w+)(?:/|_|\.)",     # .../subject_A/...
]

# =========================
# ======== 工具函数 ========
# =========================

def _safe_mkdir(dir_path: Path):
    dir_path.mkdir(parents=True, exist_ok=True)

def _cleanup_dir_if_needed(root: Path):
    if root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True, exist_ok=True)

def _try_hardlink_or_copy(src: Path, dst: Path):
    if dst.exists():
        return
    _safe_mkdir(dst.parent)
    if PREFER_HARDLINK:
        try:
            os.link(src, dst)
            return
        except Exception:
            pass
    shutil.copy2(src, dst)

def _infer_subject_from_path(p: Path) -> str:
    s = str(p.as_posix())
    for pat in SUBJECT_PATTERNS:
        m = re.search(pat, s, flags=re.IGNORECASE)
        if m:
            return m.group(1)
    # 兜底：取上一层目录名
    return p.parent.name

def _coerce_str(x: Any) -> str:
    try:
        return str(x)
    except Exception:
        return ""

def _load_npz(fp: Path) -> Dict[str, Any]:
    return dict(np.load(fp, allow_pickle=True))

def _load_npy(fp: Path) -> Dict[str, Any]:
    arr = np.load(fp, allow_pickle=True)
    return {"data": arr}

def _load_pkl(fp: Path) -> Dict[str, Any]:
    import pickle
    with open(fp, "rb") as f:
        obj = pickle.load(f)
    # 尝试几种容器：dict / (data,label,meta) / 自定义对象
    if isinstance(obj, dict):
        return obj
    if isinstance(obj, (tuple, list)) and len(obj) >= 2:
        d = {"data": obj[0], "label": obj[1]}
        if len(obj) >= 3 and isinstance(obj[2], dict):
            d.update(obj[2])
        return d
    # 对象：尝试 __dict__
    if hasattr(obj, "__dict__"):
        return obj.__dict__
    return {"data": obj}

def _load_pt(fp: Path) -> Dict[str, Any]:
    if torch is None:
        raise RuntimeError(f"torch 未安装，无法读取: {fp}")
    obj = torch.load(fp, map_location="cpu")
    if isinstance(obj, dict):
        return obj
    if hasattr(obj, "__dict__"):
        return obj.__dict__
    return {"data": obj}

def _load_meta(fp: Path) -> Dict[str, Any]:
    ext = fp.suffix.lower()
    if ext == ".npz":
        return _load_npz(fp)
    elif ext == ".npy":
        return _load_npy(fp)
    elif ext == ".pkl":
        return _load_pkl(fp)
    elif ext == ".pt":
        return _load_pt(fp)
    else:
        return {}

def _extract_subject_and_label(meta: Dict[str, Any], path_fallback: Path) -> Tuple[str, Any]:
    # subject
    subject = None
    for k in SUBJECT_KEYS:
        if k in meta:
            subject = meta[k]
            break
    if subject is None and "meta" in meta and isinstance(meta["meta"], dict):
        for k in SUBJECT_KEYS:
            if k in meta["meta"]:
                subject = meta["meta"][k]
                break
    sid = _coerce_str(subject) if subject is not None else _infer_subject_from_path(path_fallback)

    # label
    label = None
    for k in LABEL_KEYS:
        if k in meta:
            label = meta[k]
            break
    if label is None and "meta" in meta and isinstance(meta["meta"], dict):
        for k in LABEL_KEYS:
            if k in meta["meta"]:
                label = meta["meta"][k]
                break
    return sid, label

def _iter_all_files(root: Path):
    for fp in root.rglob("*"):
        if fp.is_file() and fp.suffix.lower() in FILE_EXTS:
            yield fp

# =========================
# ========= 主流程 =========
# =========================

def main():
    if OVERWRITE_OUTPUT:
        _cleanup_dir_if_needed(OUT_ROOT)
    else:
        _safe_mkdir(OUT_ROOT)

    # 1) 收集样本 -> {subject: [(path, label), ...]}
    sub2samples: Dict[str, List[Tuple[Path, Any]]] = {}
    all_files = sorted(list(_iter_all_files(IN_ROOT)))
    if not all_files:
        raise RuntimeError(f"在 {IN_ROOT} 下未发现样本文件（支持后缀：{sorted(FILE_EXTS)}）")

    print(f"[INFO] 扫描到样本文件数：{len(all_files)}")
    for i, fp in enumerate(all_files, 1):
        try:
            meta = _load_meta(fp)
        except Exception as e:
            print(f"[WARN] 读取失败，跳过: {fp} | err={e}")
            continue
        sid, label = _extract_subject_and_label(meta, fp)
        sub2samples.setdefault(sid, []).append((fp, label))
        if i % 1000 == 0:
            print(f"[INFO] 解析进行中：{i}/{len(all_files)} ...")
        del meta
        if i % 256 == 0:
            gc.collect()

    print(f"[INFO] 发现被试数：{len(sub2samples)}")
    # 2) 对每个被试做 5 折
    #    优先 StratifiedKFold（若 label 不为空且类别>1），否则 KFold
    from sklearn.model_selection import KFold, StratifiedKFold

    rng = np.random.RandomState(SEED)
    split_map: Dict[str, Dict[str, Any]] = {}  # {relpath: {"subject":sid, "label":label, "fold":i}}
    # 保存以相对路径（相对于 IN_ROOT ）作为 key，方便复现和人读

    for sid, items in sub2samples.items():
        n = len(items)
        idx = np.arange(n)
        labels = np.array([x[1] for x in items], dtype=object)

        use_strat = False
        if USE_STRATIFIED and labels is not None:
            # 标签可用的判定：非 None、非 nan、且 至少包含2个不同类别
            try:
                lab_list = [str(l) for l in labels]
                uniq = sorted(set(lab_list))
                if len(uniq) >= 2:
                    use_strat = True
            except Exception:
                use_strat = False

        if use_strat:
            splitter = StratifiedKFold(n_splits=FOLDS, shuffle=SHUFFLE, random_state=SEED)
            fold_assign = np.empty(n, dtype=int)
            # StratifiedKFold 给出 (train, test)；这里先获得“纯折号”
            # 我们只需一个稳定的 test 划分标号即可
            temp_assign = np.empty(n, dtype=int)
            for f, (_, te) in enumerate(splitter.split(idx, lab_list)):
                temp_assign[te] = f
            fold_assign[:] = temp_assign
            print(f"[INFO] 被试 {sid}: 使用 StratifiedKFold")
        else:
            splitter = KFold(n_splits=FOLDS, shuffle=SHUFFLE, random_state=SEED)
            fold_assign = np.empty(n, dtype=int)
            f = 0
            for _, te in splitter.split(idx):
                fold_assign[te] = f
                f += 1
            print(f"[INFO] 被试 {sid}: 使用 KFold")

        # 写入 split_map
        for j in range(n):
            src_path = items[j][0]
            label = items[j][1]
            rel = src_path.relative_to(IN_ROOT).as_posix()
            split_map[rel] = {"subject": sid, "label": _coerce_str(label), "fold": int(fold_assign[j])}

    # 3) 写全局 split_map.json
    with open(OUT_ROOT / "split_map.json", "w", encoding="utf-8") as f:
        json.dump({
            "in_root": str(IN_ROOT),
            "folds": FOLDS,
            "seed": SEED,
            "shuffle": SHUFFLE,
            "use_stratified": USE_STRATIFIED,
            "map": split_map
        }, f, ensure_ascii=False, indent=2)
    print(f"[OK] 写出：{OUT_ROOT/'split_map.json'}")

    # 4) 依次生成 5 个 fold 目录，并把文件“硬链接/复制”过去
    #    规则：对每个 fold k: test=k, eval=(k+1)%FOLDS, train = 其余
    rels = list(split_map.keys())
    # 为每个 fold 写 manifest.csv
    for k in range(FOLDS):
        fold_dir = OUT_ROOT / f"fold_{k}"
        split_dirs = {
            "train": fold_dir / "train",
            "eval":  fold_dir / "eval",
            "test":  fold_dir / "test",
        }
        for d in split_dirs.values():
            _safe_mkdir(d)

        # 打开 manifest.csv
        with open(fold_dir / "manifest.csv", "w", newline="", encoding="utf-8") as fcsv:
            writer = csv.writer(fcsv)
            writer.writerow(["relpath", "subject", "label", "fold", "split"])  # 保持简单直观

            eval_fold = (k + 1) % FOLDS
            for rel in rels:
                info = split_map[rel]
                src = IN_ROOT / rel
                fold_id = info["fold"]
                if fold_id == k:
                    split = "test"
                elif fold_id == eval_fold:
                    split = "eval"
                else:
                    split = "train"

                # 目标相对路径：保留原相对路径或平铺
                if PRESERVE_TREE:
                    dst = split_dirs[split] / rel
                else:
                    # 平铺，为防撞名，在文件名前加被试ID（若文件名中本就带被试ID也无妨）
                    dst_name = f"{info['subject']}__{src.name}"
                    dst = split_dirs[split] / dst_name

                try:
                    _try_hardlink_or_copy(src, dst)
                except Exception as e:
                    print(f"[WARN] 复制失败: {src} -> {dst} | err={e}")
                    continue

                writer.writerow([str(dst.relative_to(fold_dir).as_posix()),
                                 info["subject"], info["label"], info["fold"], split])

        print(f"[OK] fold_{k} 完成：{fold_dir}")

    print("[DONE] 全部完成。你可以直接用各 fold 的 manifest.csv 来喂训练脚本。")


if __name__ == "__main__":
    main()
