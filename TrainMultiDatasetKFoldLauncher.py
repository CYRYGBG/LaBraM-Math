# -*- coding: utf-8 -*-
# @Time    : 2025/11/03 10:48:37
# @Author  : Chen, Y.R.
# @File    : MultiDatasetKFoldLauncher.py
# @Software: VSCode
# @Notes    : 批量启动 LaBraM 微调（MATH_KFOLD 五折目录）。逐个数据集顺序执行，可单折或全折。
#             逻辑：检测 torchrun -> 设定 --nproc_per_node -> 组装命令 -> 依次 Popen。

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
    # run_class_finetuning.py 绝对路径
    "CODE_PATH": "/home/yeqi3/cyr/code/LaBraM/run_class_finetuning.py",

    # 选择用哪些 GPU（按顺序写逻辑编号）
    "GPUS": ["5", "6", "7"],

    # 输出与日志根目录（每个数据集会在其下新建子目录）
    "OUTPUT_BASE": "/usr/data/yeqi3/LaBraM_log/multids/cv",
    "LOG_BASE": "/usr/data/yeqi3/LaBraM_log/multids/log",

    # 是否仅跑单折；None 表示“跑全折并在脚本内做平均”
    # 例如：指定 0 则只跑第 0 折；设为 None 则跑 0..k-1 全部
    "FOLD_INDEX": None,

    # 公共训练超参（所有数据集默认用同一套，可在 DATASETS 里覆盖）
    "COMMON": {
        "model": "labram_base_patch200_200",
        "input_size": 200,          # 你的窗口长度
        "batch_size": 48,
        "epochs": 50,
        "lr": 5e-4,
        "warmup_epochs": 5,
        "layer_decay": 0.9,
        "drop_path": 0.1,
        "update_freq": 1,
        "save_ckpt_freq": 9999,
        "num_workers": 4,
        "seed": 42,
        # 与官方脚本常见开关保持一致
        "disable_rel_pos_bias": True,
        "abs_pos_emb": True,
        "disable_qkv_bias": True,
        # 额外原样透传
        "extra": "",  # 例："--model_ema --model_ema_decay 0.9999"
        # 预训练权重（可空）
        "finetune": "/home/yeqi3/cyr/code/LaBraM/checkpoints/labram-base.pth",
    },

    # 待跑数据集列表：每项至少给 data_root（指向 SplitMakeMathToKFold 的 OUT_ROOT）
    # name 仅用于生成输出/日志子目录名；nb_classes 必填（二分类=1，多分类=类别数）
    "DATASETS": [
        {
            "name": "read",
            "data_root": "/usr/data/yeqi3/labram_fold/read",
            "kfold_num": 5,
            "nb_classes": 1,
        },
        {
            "name": "type",
            "data_root": "/usr/data/yeqi3/labram_fold/type",
            "kfold_num": 5,
            "nb_classes": 1,
        },
        {
            "name": "read_new",
            "data_root": "/usr/data/yeqi3/labram_fold/read_new",
            "kfold_num": 5,
            "nb_classes": 1,
        },
        {
            "name": "type_new",
            "data_root": "/usr/data/yeqi3/labram_fold/type_new",
            "kfold_num": 5,
            "nb_classes": 1,
        },
    ],

    "ENV": {
        "OMP_NUM_THREADS": "1",
        "MASTER_PORT": "29501",  # 如端口冲突可改
    },
}
# =========================
# ===== 配置区结束 ========
# =========================


def _resolve(p: str) -> Path:
    return Path(p).expanduser().resolve()


def _which_torchrun() -> str:
    tr = shutil.which("torchrun")
    if tr:
        return tr
    py = shutil.which("python") or sys.executable
    return f"{shlex.quote(py)} -m torch.distributed.run"


def _join_commas(items) -> str:
    return ",".join(map(str, items))


def _build_cmd(
    nproc: int,
    code_path: Path,
    dataset_cfg: dict,
    common: dict,
    output_dir: Path,
    log_dir: Path,
    fold_index,  # None 或 int
) -> str:
    """
    组装调用 run_math_finetuning.py 的命令：
      --dataset MATH_KFOLD --data_root <root> --kfold_num K --fold_index (可选)
      --nb_classes N 以及公共/覆盖超参
    """
    launcher = _which_torchrun()

    # 合并公共参数与 per-dataset 覆盖
    merged = dict(common)
    override = dataset_cfg.get("override", {}) or {}
    merged.update(override)

    parts = [
        launcher,
        f"--nproc_per_node={nproc}",
        shlex.quote(str(code_path)),
        "--dataset MATH_KFOLD",
        f"--data_root {shlex.quote(str(_resolve(dataset_cfg['data_root'])))}",
        f"--kfold_num {int(dataset_cfg['kfold_num'])}",
        f"--nb_classes {int(dataset_cfg['nb_classes'])}",
        f"--model {shlex.quote(merged['model'])}",
        f"--input_size {int(merged['input_size'])}",
        f"--batch_size {int(merged['batch_size'])}",
        f"--epochs {int(merged['epochs'])}",
        f"--lr {merged['lr']}",
        f"--warmup_epochs {int(merged['warmup_epochs'])}",
        f"--layer_decay {merged['layer_decay']}",
        f"--drop_path {merged['drop_path']}",
        f"--update_freq {int(merged['update_freq'])}",
        f"--save_ckpt_freq {int(merged['save_ckpt_freq'])}",
        f"--num_workers {int(merged['num_workers'])}",
        f"--seed {int(merged['seed'])}",
        f"--output_dir {shlex.quote(str(output_dir))}",
        f"--log_dir {shlex.quote(str(log_dir))}",
        "--disable_rel_pos_bias" if merged.get("disable_rel_pos_bias", False) else "",
        "--abs_pos_emb" if merged.get("abs_pos_emb", False) else "",
        "--disable_qkv_bias" if merged.get("disable_qkv_bias", False) else "",
        (f"--finetune {shlex.quote(str(_resolve(merged['finetune'])))}" if merged.get("finetune") else ""),
        (merged.get("extra", "").strip() if merged.get("extra") else ""),
    ]

    # 单折 / 全折
    if fold_index is not None:
        parts.append(f"--fold_index {int(fold_index)}")

    return " ".join([p for p in parts if p])


def main():
    cfg = CONFIG

    # ----- 环境变量与 GPU -----
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
    for k, v in cfg.get("ENV", {}).items():
        os.environ[str(k)] = str(v)

    code_path = _resolve(cfg["CODE_PATH"])
    if not code_path.exists():
        raise FileNotFoundError(f"未找到代码文件：{code_path}")

    out_base = _resolve(cfg["OUTPUT_BASE"])
    log_base = _resolve(cfg["LOG_BASE"])
    out_base.mkdir(parents=True, exist_ok=True)
    log_base.mkdir(parents=True, exist_ok=True)

    fold_index = cfg.get("FOLD_INDEX", None)
    common = cfg["COMMON"]

    last_rc = 0
    for ds in cfg["DATASETS"]:
        name = ds.get("name") or Path(ds["data_root"]).name
        output_dir = out_base / name
        log_dir = log_base / name
        output_dir.mkdir(parents=True, exist_ok=True)
        log_dir.mkdir(parents=True, exist_ok=True)

        cmd = _build_cmd(
            nproc=nproc,
            code_path=code_path,
            dataset_cfg=ds,
            common=common,
            output_dir=output_dir,
            log_dir=log_dir,
            fold_index=fold_index,
        )

        print("\n========== Launch ==========")
        print(f"[DATASET ] {name}")
        print(f"[DATAROOT] {ds['data_root']}")
        print(f"[OUTPUT  ] {output_dir}")
        print(f"[LOG     ] {log_dir}")
        print(cmd)
        print("============================\n")

        proc = subprocess.Popen(cmd, shell=True)
        proc.communicate()
        rc = proc.returncode
        last_rc = rc
        if rc != 0:
            print(f"[WARN] 子任务失败（{name}），返回码 {rc}。继续下一个数据集。")

    sys.exit(last_rc)


if __name__ == "__main__":
    main()
