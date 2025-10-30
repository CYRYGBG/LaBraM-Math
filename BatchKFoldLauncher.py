# -*- coding: utf-8 -*-
"""批量启动基于 KFold 切分数据集的 LaBraM 微调脚本。

该脚本假定已经使用 ``SplitMakeMathToKFold.py`` 对原始数据进行整理，
并在 ``OUT_ROOT`` 下生成 ``fold_*/{train,eval,test}`` 目录结构。

配置方式参考 ``CONFIG``，可按需修改代码、数据与输出路径等参数。
脚本会依次遍历 ``DATASETS`` 中列出的数据集（如 read/type 等），
对每个数据集执行一次 ``run_class_finetuning.py``，并在结束后汇总测试结果。
"""
from __future__ import annotations

import json
import os
import shlex
import shutil
import statistics
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

# =========================
# ======= 配置区  =========
# =========================
CONFIG: Dict[str, object] = {
    # 必填： run_class_finetuning.py 的绝对路径
    "CODE_PATH": "/abs/path/to/LaBraM/run_class_finetuning.py",

    # GPU 设置：可以是列表或以逗号分隔的字符串
    "GPUS": ["0", "1", "2"],

    # 预训练权重（留空则不加载）
    "FINETUNE": "/abs/path/to/LaBraM/checkpoints/labram-base.pth",

    # 统一的输出 / 日志根目录（每个数据集会在内部按名称创建子目录）
    "OUTPUT_ROOT": "/abs/path/to/output/math_kfold",
    "LOG_ROOT": "/abs/path/to/output/math_kfold_logs",

    # 通用训练超参数
    "MODEL": "labram_base_patch200_200",
    "BATCH_SIZE": 32,
    "EPOCHS": 50,
    "LR": 5e-4,
    "WARMUP_EPOCHS": 5,
    "LAYER_DECAY": 0.65,
    "DROP_PATH": 0.1,
    "UPDATE_FREQ": 1,
    "SAVE_CKPT_FREQ": 5,
    "NUM_WORKERS": 4,

    # 5 折设置
    "K_FOLDS": 5,

    # 是否关闭/打开和官方脚本一致的常见开关
    "DISABLE_REL_POS_BIAS": True,
    "ABS_POS_EMB": True,
    "DISABLE_QKV_BIAS": True,

    # 额外追加到命令尾部的公共参数（整段字符串，可留空）
    "EXTRA": "--seed 42",

    # 环境变量
    "OMP_NUM_THREADS": "1",
    "MASTER_PORT": "29510",

    # 需要批量运行的数据集列表
    "DATASETS": [
        {
            "NAME": "read",
            "DATA_ROOT": "/abs/path/to/kfold/read",  # ← 修改为 SplitMakeMathToKFold 输出目录
            "NB_CLASSES": 2,
            "EXTRA": ""
        },
        {
            "NAME": "type",
            "DATA_ROOT": "/abs/path/to/kfold/type",
            "NB_CLASSES": 2,
            "EXTRA": ""
        },
        {
            "NAME": "read_new",
            "DATA_ROOT": "/abs/path/to/kfold/read_new",
            "NB_CLASSES": 2,
            "EXTRA": ""
        },
        {
            "NAME": "type_new",
            "DATA_ROOT": "/abs/path/to/kfold/type_new",
            "NB_CLASSES": 2,
            "EXTRA": ""
        },
    ],
}
# =========================
# ===== 配置区结束 ========
# =========================


@dataclass
class DatasetResult:
    name: str
    command: str
    returncode: int
    stdout_path: Path
    metrics: Dict[str, Dict[str, float]]
    summary: Dict[str, float]


def _resolve(path_str: str) -> Path:
    return Path(path_str).expanduser().resolve()


def _find_torch_launcher() -> str:
    torchrun = shutil.which("torchrun")
    if torchrun:
        return torchrun
    py = shutil.which("python") or shlex.quote(os.environ.get("PYTHON", "python"))
    return f"{py} -m torch.distributed.run"


def _normalize_gpus(gpus_cfg) -> (str, int):
    if isinstance(gpus_cfg, (list, tuple)):
        cleaned = [str(g).strip() for g in gpus_cfg if str(g).strip()]
        return ",".join(cleaned), len(cleaned)
    gpus = str(gpus_cfg).strip()
    parts = [p for p in gpus.split(",") if p.strip()]
    return gpus, len(parts)


def _build_common_parts(cfg: Dict[str, object], *, nproc: int, code_path: Path,
                        output_dir: Path, log_dir: Path, dataset_cfg: Dict[str, object]) -> List[str]:
    parts: List[str] = []
    parts.append(f"--nproc_per_node={nproc}")
    parts.append(shlex.quote(str(code_path)))
    parts.extend([
        "--dataset MATH_KFOLD",
        f"--data_root {shlex.quote(str(_resolve(dataset_cfg['DATA_ROOT'])))}",
        f"--kfold_num {int(cfg['K_FOLDS'])}",
        f"--nb_classes {int(dataset_cfg['NB_CLASSES'])}",
        f"--model {shlex.quote(str(cfg['MODEL']))}",
        f"--batch_size {int(cfg['BATCH_SIZE'])}",
        f"--epochs {int(cfg['EPOCHS'])}",
        f"--lr {cfg['LR']}",
        f"--warmup_epochs {int(cfg['WARMUP_EPOCHS'])}",
        f"--layer_decay {cfg['LAYER_DECAY']}",
        f"--drop_path {cfg['DROP_PATH']}",
        f"--update_freq {int(cfg['UPDATE_FREQ'])}",
        f"--save_ckpt_freq {int(cfg['SAVE_CKPT_FREQ'])}",
        f"--num_workers {int(cfg['NUM_WORKERS'])}",
        f"--output_dir {shlex.quote(str(output_dir))}",
        f"--log_dir {shlex.quote(str(log_dir))}",
    ])
    if cfg.get("DISABLE_REL_POS_BIAS"):
        parts.append("--disable_rel_pos_bias")
    if cfg.get("ABS_POS_EMB"):
        parts.append("--abs_pos_emb")
    if cfg.get("DISABLE_QKV_BIAS"):
        parts.append("--disable_qkv_bias")
    finetune = cfg.get("FINETUNE")
    if finetune:
        parts.append(f"--finetune {shlex.quote(str(_resolve(finetune)))}")
    extra_common = str(cfg.get("EXTRA", "")).strip()
    if extra_common:
        parts.append(extra_common)
    extra_dataset = str(dataset_cfg.get("EXTRA", "")).strip()
    if extra_dataset:
        parts.append(extra_dataset)
    return parts


def _collect_metrics(output_dir: Path) -> Dict[str, Dict[str, float]]:
    metrics: Dict[str, Dict[str, float]] = {}
    for fold_dir in sorted(output_dir.glob("fold_*")):
        log_file = fold_dir / "log.txt"
        if not log_file.exists():
            continue
        last_line_obj = None
        with log_file.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    last_line_obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
        if not last_line_obj:
            continue
        fold_metrics: Dict[str, float] = {}
        for key, value in last_line_obj.items():
            if key.startswith("test_") and isinstance(value, (int, float)):
                fold_metrics[key[len("test_"):]] = float(value)
        if fold_metrics:
            metrics[fold_dir.name] = fold_metrics
    return metrics


def _summarize_metrics(metrics: Dict[str, Dict[str, float]]) -> Dict[str, float]:
    summary: Dict[str, float] = {}
    if not metrics:
        return summary
    # 仅对常见指标做聚合
    for metric_name in sorted({m for fold in metrics.values() for m in fold.keys()}):
        values = [fold_metrics[metric_name] for fold_metrics in metrics.values()
                  if metric_name in fold_metrics]
        if not values:
            continue
        if len(values) == 1:
            summary[metric_name] = values[0]
        else:
            summary[f"{metric_name}_mean"] = float(statistics.mean(values))
            summary[f"{metric_name}_std"] = float(statistics.pstdev(values))
    return summary


def _write_stdout(lines: List[str], dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text("".join(lines), encoding="utf-8")


def _run_one_dataset(cfg: Dict[str, object], dataset_cfg: Dict[str, object], *, launcher: str,
                     gpus_env: str, nproc: int) -> DatasetResult:
    dataset_name = dataset_cfg["NAME"]
    output_root = _resolve(cfg["OUTPUT_ROOT"])
    log_root = _resolve(cfg["LOG_ROOT"])
    output_dir = dataset_cfg.get("OUTPUT_DIR")
    log_dir = dataset_cfg.get("LOG_DIR")
    if output_dir:
        output_dir = _resolve(output_dir)
    else:
        output_dir = output_root / dataset_name
    if log_dir:
        log_dir = _resolve(log_dir)
    else:
        log_dir = log_root / dataset_name
    output_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    code_path = _resolve(cfg["CODE_PATH"])
    if not code_path.exists():
        raise FileNotFoundError(f"CODE_PATH 不存在: {code_path}")

    cmd_parts = [launcher]
    cmd_parts.extend(_build_common_parts(cfg, nproc=nproc, code_path=code_path,
                                         output_dir=output_dir, log_dir=log_dir,
                                         dataset_cfg=dataset_cfg))
    cmd = " ".join(cmd_parts)

    print("\n==============================")
    print(f"[DATASET] {dataset_name}")
    print(cmd)
    print("==============================\n")

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = gpus_env
    if cfg.get("OMP_NUM_THREADS"):
        env["OMP_NUM_THREADS"] = str(cfg["OMP_NUM_THREADS"])
    if cfg.get("MASTER_PORT"):
        env["MASTER_PORT"] = str(cfg["MASTER_PORT"])

    stdout_lines: List[str] = []
    process = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE,
                               stderr=subprocess.STDOUT, env=env, text=True,
                               bufsize=1)
    assert process.stdout is not None
    for line in process.stdout:
        stdout_lines.append(line)
        print(f"[{dataset_name}] {line}", end="")
    process.wait()

    stdout_path = output_dir / "launcher_stdout.log"
    _write_stdout(stdout_lines, stdout_path)

    metrics = _collect_metrics(output_dir)
    summary = _summarize_metrics(metrics)
    return DatasetResult(
        name=dataset_name,
        command=cmd,
        returncode=process.returncode,
        stdout_path=stdout_path,
        metrics=metrics,
        summary=summary,
    )


def main():
    cfg = CONFIG
    gpus_env, nproc = _normalize_gpus(cfg.get("GPUS", ""))
    if nproc <= 0:
        raise ValueError("未检测到可用 GPU，请检查 CONFIG['GPUS'] 设置。")

    launcher = _find_torch_launcher()

    # 预创建根目录
    _resolve(cfg["OUTPUT_ROOT"]).mkdir(parents=True, exist_ok=True)
    _resolve(cfg["LOG_ROOT"]).mkdir(parents=True, exist_ok=True)

    datasets: List[Dict[str, object]] = cfg.get("DATASETS", [])  # type: ignore
    if not datasets:
        raise ValueError("CONFIG['DATASETS'] 为空，至少指定一个数据集。")

    results: List[DatasetResult] = []
    for ds_cfg in datasets:
        try:
            res = _run_one_dataset(cfg, ds_cfg, launcher=launcher, gpus_env=gpus_env, nproc=nproc)
        except Exception as exc:  # noqa: BLE001
            print(f"[ERROR] 数据集 {ds_cfg.get('NAME')} 运行失败：{exc}")
            raise
        results.append(res)

    summary_output = {
        "results": [
            {
                "name": r.name,
                "returncode": r.returncode,
                "command": r.command,
                "stdout_log": str(r.stdout_path),
                "metrics": r.metrics,
                "summary": r.summary,
            }
            for r in results
        ]
    }

    summary_path = _resolve(cfg["OUTPUT_ROOT"]) / "batch_summary.json"
    summary_path.write_text(json.dumps(summary_output, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n===== Batch Summary =====")
    for r in results:
        status = "OK" if r.returncode == 0 else f"FAILED({r.returncode})"
        print(f"- {r.name}: {status}")
        if r.summary:
            for key, value in sorted(r.summary.items()):
                print(f"    {key}: {value:.4f}")
        else:
            print("    未从日志中解析到测试指标，请检查输出目录。")
        print(f"    stdout: {r.stdout_path}")


if __name__ == "__main__":
    main()
