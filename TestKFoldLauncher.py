# -*- coding: utf-8 -*-
"""Batch evaluation helper for Math k-fold datasets.

参考 `TrainCVLauncher.py` 的组织方式，这个脚本会逐个数据集调用
`run_class_finetuning.py --dataset MATH_KFOLD --eval`，用于在由
`SplitMakeMathToKFold.py` 生成的目录结构上做测试。
"""

from __future__ import annotations

import json
import os
import re
import shlex
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional


# =========================
# ========= 配置区 =========
# =========================
CONFIG = {
    # run_class_finetuning.py 的绝对路径（默认指向当前仓库文件）
    "CODE_PATH": str(Path(__file__).resolve().parent / "run_class_finetuning.py"),

    # 指定要测试的数据集：名称 -> 根目录。根目录下需包含 fold_0 ... fold_{k-1}
    # 可以把 value 写成 str 或 dict；如需覆盖 nb_classes，可写成
    # {"root": "/path/to/read", "nb_classes": 2}
    "DATASETS": {
        "read": "/path/to/math_kfold/read",
        "type": "/path/to/math_kfold/type",
        "read_new": "/path/to/math_kfold/read_new",
        "type_new": "/path/to/math_kfold/type_new",
    },

    # 模型与推理所需参数
    "MODEL": "labram_base_patch200_200",
    "NB_CLASSES": 2,
    "BATCH_SIZE": 64,
    "NUM_WORKERS": 4,
    "KFOLD_NUM": 5,

    # 预训练 / 微调权重，供 --finetune 使用；留空则不加载
    "FINETUNE": "/path/to/checkpoints/labram-base.pth",

    # 输出根目录、日志根目录（会按数据集名创建子目录）
    "OUTPUT_DIR": "/path/to/output/math_eval",
    "LOG_DIR": "/path/to/output/math_eval_logs",

    # 指定 GPU（字符串或列表）。与 TrainCVLauncher 保持一致
    "GPUS": ["0"],

    # 额外拼接到命令末尾的参数（整段字符串，可选）
    "EXTRA": "--seed 42",

    # 若指定则把每个数据集的解析精度写入该 JSON 文件
    "RESULT_JSON": "/path/to/output/math_eval/results.json",
}
# =========================
# ======= 配置区结束 =======
# =========================


@dataclass
class DatasetConfig:
    name: str
    root: Path
    nb_classes: int


def _resolve(path_str: str) -> Path:
    return Path(path_str).expanduser().resolve()


def _as_dataset_configs(raw: Dict[str, object], default_nb: int) -> List[DatasetConfig]:
    cfgs: List[DatasetConfig] = []
    for name, value in raw.items():
        if isinstance(value, dict):
            root = _resolve(str(value.get("root")))
            nb = int(value.get("nb_classes", default_nb))
        else:
            root = _resolve(str(value))
            nb = int(default_nb)
        cfgs.append(DatasetConfig(name=name, root=root, nb_classes=nb))
    return cfgs


def _python_executable() -> str:
    py = shutil.which("python")
    return py if py else sys.executable


def _build_command(
    python_exec: str,
    code_path: Path,
    dataset: DatasetConfig,
    cfg: dict,
    output_root: Optional[Path],
    log_root: Optional[Path],
) -> List[str]:
    cmd: List[str] = [
        python_exec,
        str(code_path),
        "--dataset", "MATH_KFOLD",
        "--model", cfg["MODEL"],
        "--data_root", str(dataset.root),
        "--nb_classes", str(dataset.nb_classes),
        "--batch_size", str(cfg["BATCH_SIZE"]),
        "--num_workers", str(cfg["NUM_WORKERS"]),
        "--kfold_num", str(cfg["KFOLD_NUM"]),
        "--eval",
    ]
    if cfg.get("FINETUNE"):
        cmd.extend(["--finetune", str(_resolve(cfg["FINETUNE"]))])
    if output_root is not None:
        ds_out = output_root / dataset.name
        ds_out.mkdir(parents=True, exist_ok=True)
        cmd.extend(["--output_dir", str(ds_out)])
    if log_root is not None:
        ds_log = log_root / dataset.name
        ds_log.mkdir(parents=True, exist_ok=True)
        cmd.extend(["--log_dir", str(ds_log)])
    extra = cfg.get("EXTRA", "").strip()
    if extra:
        cmd.extend(shlex.split(extra))
    return cmd


def _parse_accuracy(output: str) -> Optional[float]:
    # 优先匹配“Average accuracy over ...”
    m = re.search(r"Average accuracy over \d+ folds: ([0-9.]+)", output)
    if m:
        try:
            return float(m.group(1))
        except ValueError:
            pass
    # 退回匹配“======Accuracy: mean std ...”
    m = re.search(r"======Accuracy: ([0-9.eE+-]+)", output)
    if m:
        try:
            return float(m.group(1))
        except ValueError:
            pass
    return None


def main() -> None:
    cfg = CONFIG.copy()

    code_path = _resolve(cfg["CODE_PATH"])
    if not code_path.exists():
        raise FileNotFoundError(f"未找到 run_class_finetuning.py：{code_path}")

    datasets = _as_dataset_configs(cfg["DATASETS"], cfg["NB_CLASSES"])
    if not datasets:
        raise ValueError("DATASETS 配置为空")
    for ds in datasets:
        if not ds.root.exists():
            raise FileNotFoundError(f"数据集根目录不存在：{ds.root}")

    output_root = _resolve(cfg["OUTPUT_DIR"]) if cfg.get("OUTPUT_DIR") else None
    if output_root is not None:
        output_root.mkdir(parents=True, exist_ok=True)
    log_root = _resolve(cfg["LOG_DIR"]) if cfg.get("LOG_DIR") else None
    if log_root is not None:
        log_root.mkdir(parents=True, exist_ok=True)

    python_exec = _python_executable()

    # GPU 环境变量设置
    gpus = cfg.get("GPUS")
    if isinstance(gpus, (list, tuple)):
        gpu_list = [str(x).strip() for x in gpus if str(x).strip()]
        gpu_env = ",".join(gpu_list)
    else:
        gpu_env = str(gpus)
    if gpu_env:
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu_env

    results: Dict[str, Optional[float]] = {}

    for dataset in datasets:
        print("\n===== Evaluating dataset: {} =====".format(dataset.name))
        cmd = _build_command(
            python_exec=python_exec,
            code_path=code_path,
            dataset=dataset,
            cfg=cfg,
            output_root=output_root,
            log_root=log_root,
        )
        print("Command:", " ".join(shlex.quote(part) for part in cmd))
        proc = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        print(proc.stdout)
        if proc.returncode != 0:
            print(f"[ERROR] 命令执行失败（returncode={proc.returncode}）。")
            results[dataset.name] = None
            continue
        acc = _parse_accuracy(proc.stdout)
        if acc is not None:
            print(f"[INFO] {dataset.name} 平均准确率：{acc:.4f}")
        else:
            print(f"[WARN] 未能解析 {dataset.name} 的准确率，请检查输出。")
        results[dataset.name] = acc

    result_json = cfg.get("RESULT_JSON")
    if result_json:
        result_path = _resolve(result_json)
        result_path.parent.mkdir(parents=True, exist_ok=True)
        with open(result_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"结果已写入：{result_path}")


if __name__ == "__main__":
    main()
