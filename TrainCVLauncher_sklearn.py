# -*- coding: utf-8 -*-
# @Time     : 2025/11/07 16:50
# @Author   : Chen, Y.R.
# @File     : TrainCVLauncher_sklearn_multi.py
# @Software : VSCode
# @Notes    : 启动 run_math_sklearn.py，逐数据集 × 全部 sklearn 模型；结果目录为 <OUTPUT_BASE>/<dataset>/<sk_model>/

import os
import sys
import shlex
import subprocess
from pathlib import Path
import shutil
import warnings
warnings.filterwarnings("ignore")

CONFIG = {
    # === 基础配置 ===
    "CODE_PATH": "/home/yeqi3/cyr/code/LaBraM/run_math_sklearn.py",
    "GPUS": ["4"],  # sklearn_eval 模式仅需单卡
    "FINETUNE": "/home/yeqi3/cyr/code/LaBraM/checkpoints/labram-base.pth",  # 可留空 ""

    # === 数据集根目录（全部启用）===
    "PKL_ROOTS": [
        "/usr/data/yeqi3/labram_processed/read",
        "/usr/data/yeqi3/labram_processed/type",
        "/usr/data/yeqi3/labram_processed/read_new",
        "/usr/data/yeqi3/labram_processed/type_new",
    ],

    # === backbone（仅用于提特征，不参与目录分层）===
    "BACKBONE": "labram_base_patch200_200",

    # === 输出与CV配置 ===
    "OUTPUT_BASE": "/home/yeqi3/cyr/code/LaBraM/sklearn_result",
    "SUBJECT_REGEX": r"sub_(\d+)_simplified",
    "CV_SPLITS": 5,
    "SEED": 42,
    "OMP_NUM_THREADS": "1",
    "MASTER_PORT": "29505",

    # 是否把提取的特征也存 npz（每折一个）
    "SAVE_FEATURES": False,
}

# 用“run_math_sklearn.py / build_sklearn_models”中定义的全部可用键
ALL_SK_MODELS = [
    "SVM_RBF",
    "SVM_Linear",
    "SVM_Poly",
    "SVM_Sigmoid",
    "SVM_L1",
    "SVM_ElasticNet",
    "LASSO_then_SVM_RBF",
    "LASSO_then_SVM_Linear",
    "DecisionTree",
    "LogisticRegression",
    "KNeighbors",
    "GaussianNB",
    "RandomForest",
    # "GradientBoosting",
    # "AdaBoost",
    # "LDA",
    # "QDA",
]

def _resolve(p): return Path(p).expanduser().resolve()

def _launcher():
    tr = shutil.which("torchrun")
    if tr:
        return tr
    py = shutil.which("python") or sys.executable
    return f"{py} -m torch.distributed.run"

def main():
    cfg = CONFIG
    os.environ["OMP_NUM_THREADS"] = cfg["OMP_NUM_THREADS"]
    os.environ["MASTER_PORT"] = str(cfg["MASTER_PORT"])
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(cfg["GPUS"])

    code_path = _resolve(cfg["CODE_PATH"])
    if not code_path.exists():
        raise FileNotFoundError(f"未找到代码文件：{code_path}")

    launcher = _launcher()
    launcher_parts = shlex.split(launcher)

    for root in cfg["PKL_ROOTS"]:
        pkl_root = _resolve(root)
        dataset_name = pkl_root.name

        for sk in ALL_SK_MODELS:
            out_dir = _resolve(cfg["OUTPUT_BASE"]) / dataset_name / sk
            out_dir.mkdir(parents=True, exist_ok=True)

            cmd = [
                *launcher_parts,
                "--nproc_per_node=1",
                str(code_path),
                "--within_subject_cv",
                "--sklearn_eval",
                "--model", cfg["BACKBONE"],                 # 仅用于特征提取
                "--pkl_roots", str(pkl_root),
                "--output_dir", str(out_dir),
                "--subject_regex", cfg["SUBJECT_REGEX"],
                "--cv_splits", str(cfg["CV_SPLITS"]),
                "--seed", str(cfg["SEED"]),
                "--sk_models", sk,                          # 每次只跑一个sk模型
            ]
            if cfg.get("FINETUNE"):
                cmd += ["--finetune", cfg["FINETUNE"]]
            if cfg["SAVE_FEATURES"]:
                cmd += ["--save_features", "1"]

            print("\n===== Launch sklearn eval =====")
            print(f"[DATA ] {pkl_root}")
            print(f"[SK   ] {sk}")
            print(f"[OUT  ] {out_dir}")
            # print(" ".join(shlex.quote(c) for c in cmd))
            print("===============================\n")

            subprocess.run(cmd, check=False)

if __name__ == "__main__":
    main()
