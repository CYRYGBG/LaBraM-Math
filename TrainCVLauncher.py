# -*- coding: utf-8 -*-
# @Time    : 2025/10/29 14:55:13
# @Author  : Chen, Y.R.
# @File    : TrainCVLauncher.py
# @Software: VSCode
# @Notes    : 启动 LaBraM 微调（被试内 5 折 CV），不使用 argparse；仅需在下方 CONFIG 中手动修改参数。
#             会自动设置 CUDA_VISIBLE_DEVICES，并按 GPU 数设置 --nproc_per_node。
#             兼容 torchrun 或回退到 "python -m torch.distributed.run"。

import os
import sys
import shlex
import shutil
import subprocess
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")


# =========================
# ======= 配置区  =========
# =========================
CONFIG = {
    # 必填： run_class_finetuning.py 绝对路径
    "CODE_PATH": "/home/yeqi3/cyr/code/LaBraM/run_math_finetuning.py",

    # 选择用哪些 GPU（按顺序写逻辑编号）
    "GPUS": ["5", "6", "7"],

    # 处理好的 PKL 根目录（可以多个）。支持子目录递归扫描 *.pkl
    "PKL_ROOTS": [
        "/usr/data/yeqi3/labram_processed/read",
        "/usr/data/yeqi3/labram_processed/type",
        "/usr/data/yeqi3/labram_processed/read_new",
        "/usr/data/yeqi3/labram_processed/type_new",
    ],

    # 可选：预训练权重（留空则不加载）
    "FINETUNE": "/home/yeqi3/cyr/code/LaBraM/checkpoints/labram-base.pth",

    # 输出与日志目录
    "OUTPUT_DIR": "/usr/data/yeqi3/LaBraM_log/math/cv",
    "LOG_DIR": "/usr/data/yeqi3/LaBraM_log/math/log",

    # 被试ID提取正则（基于文件名）
    "SUBJECT_REGEX": r"sub_(\d+)_simplified",
    "CV_SPLITS": 5,

    # 模型与超参
    "MODEL": "labram_base_patch200_200",
    "BATCH_SIZE": 32,
    "EPOCHS": 50,
    "LR": 5e-4,
    "WARMUP_EPOCHS": 5,
    "LAYER_DECAY": 0.65,
    "DROP_PATH": 0.1,
    "UPDATE_FREQ": 1,
    "SAVE_CKPT_FREQ": 9999,
    "NUM_WORKERS": 4,

    # 官方脚本常见开关（保持与 README 一致）
    "DISABLE_REL_POS_BIAS": True,
    "ABS_POS_EMB": True,
    "DISABLE_QKV_BIAS": True,

    # 额外传入原脚本支持的参数（整段字符串，留空则不加）
    # 例："--seed 42 --model_ema --model_ema_decay 0.9999"
    "EXTRA": "--seed 42",

    # 环境变量
    "OMP_NUM_THREADS": "1",

    # 分布式 master_port（如端口冲突可改）
    "MASTER_PORT": "29500",
}
# =========================
# ===== 配置区结束 ========
# =========================


def _resolve(path_str: str) -> Path:
    return Path(path_str).expanduser().resolve()


def _find_torch_launcher() -> str:
    """优先使用 torchrun；找不到则退回 python -m torch.distributed.run。"""
    torchrun = shutil.which("torchrun")
    if torchrun:
        return torchrun
    py = shutil.which("python") or sys.executable
    return f"{shlex.quote(py)} -m torch.distributed.run"


def _join_commas(items) -> str:
    return ",".join(map(str, items))


def _as_shell_cmd_list(
    nproc: int,
    code_path: Path,
    model: str,
    finetune: str,
    output_dir: Path,
    log_dir: Path,
    pkl_roots: str,
    subject_regex: str,
    cv_splits: int,
    batch_size: int,
    epochs: int,
    lr: float,
    warmup_epochs: int,
    layer_decay: float,
    drop_path: float,
    update_freq: int,
    save_ckpt_freq: int,
    num_workers: int,
    disable_rel_pos_bias: bool,
    abs_pos_emb: bool,
    disable_qkv_bias: bool,
    extra: str,
) -> str:
    launcher = _find_torch_launcher()
    parts = [
        launcher,
        f"--nproc_per_node={nproc}",
        shlex.quote(str(code_path)),
        "--within_subject_cv",
        f"--model {shlex.quote(model)}",
        f"--output_dir {shlex.quote(str(output_dir))}",
        f"--log_dir {shlex.quote(str(log_dir))}",
        f"--pkl_roots {shlex.quote(pkl_roots)}",
        f"--subject_regex {shlex.quote(subject_regex)}",
        f"--cv_splits {int(cv_splits)}",
        f"--batch_size {int(batch_size)}",
        f"--epochs {int(epochs)}",
        f"--lr {lr}",
        f"--warmup_epochs {int(warmup_epochs)}",
        f"--layer_decay {layer_decay}",
        f"--drop_path {drop_path}",
        f"--update_freq {int(update_freq)}",
        f"--save_ckpt_freq {int(save_ckpt_freq)}",
        f"--num_workers {int(num_workers)}",
        "--disable_rel_pos_bias" if disable_rel_pos_bias else "",
        "--abs_pos_emb" if abs_pos_emb else "",
        "--disable_qkv_bias" if disable_qkv_bias else "",
        (f"--finetune {shlex.quote(finetune)}" if finetune else ""),
        (extra.strip() if extra.strip() else ""),
    ]
    return " ".join([p for p in parts if p])


def main():
    cfg = CONFIG.copy()

    # 环境变量：GPU 与线程
    gpus = cfg["GPUS"]
    if isinstance(gpus, (list, tuple)):
        gpus_env = _join_commas(gpus)
        nproc = len([x for x in gpus if str(x).strip() != ""])
    else:
        gpus_env = str(gpus)
        nproc = len([x for x in gpus_env.split(",") if x.strip() != ""])

    if nproc < 1:
        raise ValueError("未解析到有效 GPU，请检查 CONFIG['GPUS'].")

    os.environ["CUDA_VISIBLE_DEVICES"] = gpus_env
    os.environ["OMP_NUM_THREADS"] = cfg.get("OMP_NUM_THREADS", "1")
    if cfg.get("MASTER_PORT"):
        os.environ["MASTER_PORT"] = str(cfg["MASTER_PORT"])

    # 路径
    code_path = _resolve(cfg["CODE_PATH"])
    if not code_path.exists():
        raise FileNotFoundError(f"未找到代码文件：{code_path}")

    # PKL 根目录
    pkl_roots_list = cfg["PKL_ROOTS"]
    if isinstance(pkl_roots_list, (list, tuple)):
        pkl_roots_paths = [str(_resolve(p)) for p in pkl_roots_list]
        pkl_roots = ",".join(pkl_roots_paths)
    else:
        # 也支持直接给逗号分隔的字符串
        pkl_roots = pkl_roots_list
        pkl_roots_paths = pkl_roots.split(",")

    # 为每个pkl根目录创建对应的输出和日志子目录
    for pkl_root in pkl_roots_paths:
        # 从pkl根目录路径中提取最后一级目录名作为子目录名
        subdir_name = Path(pkl_root).name
        
        # 创建对应的输出和日志子目录
        output_subdir = _resolve(cfg["OUTPUT_DIR"]) / subdir_name
        log_subdir = _resolve(cfg["LOG_DIR"]) / subdir_name
        output_subdir.mkdir(parents=True, exist_ok=True)
        log_subdir.mkdir(parents=True, exist_ok=True)

    # 新代码（替换 main() 末尾同一位置）

    # 逐个 PKL 根目录独立启动
    last_returncode = 0
    for pkl_root in pkl_roots_paths:
        subdir_name = Path(pkl_root).name  # 例如 read / type / read_new / type_new

        output_subdir = _resolve(cfg["OUTPUT_DIR"]) / subdir_name
        log_subdir    = _resolve(cfg["LOG_DIR"]) / subdir_name
        output_subdir.mkdir(parents=True, exist_ok=True)
        log_subdir.mkdir(parents=True, exist_ok=True)

        # 仅把当前根目录传给 --pkl_roots
        cmd = _as_shell_cmd_list(
            nproc=nproc,
            code_path=code_path,
            model=cfg["MODEL"],
            finetune=(str(_resolve(cfg["FINETUNE"])) if cfg.get("FINETUNE") else ""),
            output_dir=output_subdir,
            log_dir=log_subdir,
            pkl_roots=str(_resolve(pkl_root)),
            subject_regex=cfg["SUBJECT_REGEX"],
            cv_splits=cfg["CV_SPLITS"],
            batch_size=cfg["BATCH_SIZE"],
            epochs=cfg["EPOCHS"],
            lr=cfg["LR"],
            warmup_epochs=cfg["WARMUP_EPOCHS"],
            layer_decay=cfg["LAYER_DECAY"],
            drop_path=cfg["DROP_PATH"],
            update_freq=cfg["UPDATE_FREQ"],
            save_ckpt_freq=cfg["SAVE_CKPT_FREQ"],
            num_workers=cfg["NUM_WORKERS"],
            disable_rel_pos_bias=cfg["DISABLE_REL_POS_BIAS"],
            abs_pos_emb=cfg["ABS_POS_EMB"],
            disable_qkv_bias=cfg["DISABLE_QKV_BIAS"],
            extra=cfg.get("EXTRA", ""),
        )

        print("\n===== Launch Command =====")
        print(f"[DATA ROOT] {pkl_root}")
        print(f"[OUTPUT   ] {output_subdir}")
        print(f"[LOG      ] {log_subdir}")
        print(cmd)
        print("==========================\n")

        # 顺序执行：一个根目录跑完再跑下一个（端口等环境变量可复用）
        proc = subprocess.Popen(cmd, shell=True)
        proc.communicate()
        last_returncode = proc.returncode
        if last_returncode != 0:
            print(f"[WARN] 子任务失败（{pkl_root}），返回码 {last_returncode}。后续任务仍将继续。")

    sys.exit(last_returncode)



if __name__ == "__main__":
    main()