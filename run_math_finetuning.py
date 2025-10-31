# -*- coding: utf-8 -*-
# @Time    : 2025/10/29 14:52:31
# @Author  : Chen, Y.R.
# @File    : run_math_finetuning.py
# @Software: VSCode
# --------------------------------------------------------
# Large Brain Model for Learning Generic Representations with Tremendous EEG Data in BCI
# By Wei-Bang Jiang
# Based on BEiT-v2, timm, DeiT, and DINO code bases
# https://github.com/microsoft/unilm/tree/master/beitv2
# https://github.com/rwightman/pytorch-image-models/tree/master/timm
# https://github.com/facebookresearch/deit/
# https://github.com/facebookresearch/dino
# ---------------------------------------------------------
# --- [PATCH] make pyhealth cache dir creation DDP-safe ---
try:
    import os as _os
    import importlib
    # 先导入 pyhealth.utils 并把 create_directory 改成 exist_ok=True 的版本
    _ph_utils = importlib.import_module("pyhealth.utils")
    def _safe_create_directory(d):
        _os.makedirs(d, exist_ok=True)
    _ph_utils.create_directory = _safe_create_directory
except Exception:
    # 若环境没有 pyhealth 或导入失败，就忽略；后续按原逻辑
    pass
# --- [PATCH END] ---
import argparse
import datetime
from pyexpat import model
import numpy as np
import time
import torch
import torch.backends.cudnn as cudnn
import json
import os


from pathlib import Path
from collections import OrderedDict, defaultdict
from timm.data.mixup import Mixup
from timm.models import create_model
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from timm.utils import ModelEma
from optim_factory import create_optimizer, get_parameter_groups, LayerDecayValueAssigner

from engine_for_finetuning import train_one_epoch, evaluate
from utils import NativeScalerWithGradNormCount as NativeScaler
import utils
from scipy import interpolate
import modeling_finetune
import warnings
warnings.filterwarnings("ignore")
import fcntl

# ======== [NEW] 仅为“被试内5折CV”模式增加的轻量依赖 ========
import re
import csv
import pickle
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import Dataset

# ====================== 原有参数 ======================
def get_args():
    parser = argparse.ArgumentParser('LaBraM fine-tuning and evaluation script for EEG classification', add_help=False)
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--epochs', default=30, type=int)
    parser.add_argument('--update_freq', default=1, type=int)
    parser.add_argument('--save_ckpt_freq', default=5, type=int)

    # robust evaluation
    parser.add_argument('--robust_test', default=None, type=str,
                        help='robust evaluation dataset')
    
    # Model parameters
    parser.add_argument('--model', default='labram_base_patch200_200', type=str, metavar='MODEL',
                        help='Name of model to train')
    parser.add_argument('--qkv_bias', action='store_true')
    parser.add_argument('--disable_qkv_bias', action='store_false', dest='qkv_bias')
    parser.set_defaults(qkv_bias=True)
    parser.add_argument('--rel_pos_bias', action='store_true')
    parser.add_argument('--disable_rel_pos_bias', action='store_false', dest='rel_pos_bias')
    parser.set_defaults(rel_pos_bias=True)
    parser.add_argument('--abs_pos_emb', action='store_true')
    parser.set_defaults(abs_pos_emb=False)
    parser.add_argument('--layer_scale_init_value', default=0.1, type=float, 
                        help="0.1 for base, 1e-5 for large. set 0 to disable layer scale")

    parser.add_argument('--input_size', default=200, type=int,
                        help='EEG input size')

    parser.add_argument('--drop', type=float, default=0.0, metavar='PCT',
                        help='Dropout rate (default: 0.)')
    parser.add_argument('--attn_drop_rate', type=float, default=0.0, metavar='PCT',
                        help='Attention dropout rate (default: 0.)')
    parser.add_argument('--drop_path', type=float, default=0.1, metavar='PCT',
                        help='Drop path rate (default: 0.1)')

    parser.add_argument('--disable_eval_during_finetuning', action='store_true', default=False)

    parser.add_argument('--model_ema', action='store_true', default=False)
    parser.add_argument('--model_ema_decay', type=float, default=0.9999, help='')
    parser.add_argument('--model_ema_force_cpu', action='store_true', default=False, help='')

    # Optimizer parameters
    parser.add_argument('--opt', default='adamw', type=str, metavar='OPTIMIZER',
                        help='Optimizer (default: "adamw"')
    parser.add_argument('--opt_eps', default=1e-8, type=float, metavar='EPSILON',
                        help='Optimizer Epsilon (default: 1e-8)')
    parser.add_argument('--opt_betas', default=None, type=float, nargs='+', metavar='BETA',
                        help='Optimizer Betas (default: None, use opt default)')
    parser.add_argument('--clip_grad', type=float, default=None, metavar='NORM',
                        help='Clip gradient norm (default: None, no clipping)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--weight_decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')
    parser.add_argument('--weight_decay_end', type=float, default=None, help="""Final value of the
        weight decay. We use a cosine schedule for WD and using a larger decay by
        the end of training improves performance for ViTs.""")

    parser.add_argument('--lr', type=float, default=5e-4, metavar='LR',
                        help='learning rate (default: 5e-4)')
    parser.add_argument('--layer_decay', type=float, default=0.9)

    parser.add_argument('--warmup_lr', type=float, default=1e-6, metavar='LR',
                        help='warmup learning rate (default: 1e-6)')
    parser.add_argument('--min_lr', type=float, default=1e-6, metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0 (1e-5)')

    parser.add_argument('--warmup_epochs', type=int, default=5, metavar='N',
                        help='epochs to warmup LR, if scheduler supports')
    parser.add_argument('--warmup_steps', type=int, default=-1, metavar='N',
                        help='num of steps to warmup LR, will overload warmup_epochs if set > 0')

    parser.add_argument('--smoothing', type=float, default=0.1,
                        help='Label smoothing (default: 0.1)')

    # * Random Erase params
    parser.add_argument('--reprob', type=float, default=0.25, metavar='PCT',
                        help='Random erase prob (default: 0.25)')
    parser.add_argument('--remode', type=str, default='pixel',
                        help='Random erase mode (default: "pixel")')
    parser.add_argument('--recount', type=int, default=1,
                        help='Random erase count (default: 1)')
    parser.add_argument('--resplit', action='store_true', default=False,
                        help='Do not random erase first (clean) augmentation split')

    # * Finetuning params
    parser.add_argument('--finetune', default='',
                        help='finetune from checkpoint')
    parser.add_argument('--model_key', default='model|module', type=str)
    parser.add_argument('--model_prefix', default='', type=str)
    parser.add_argument('--model_filter_name', default='gzp', type=str)
    parser.add_argument('--init_scale', default=0.001, type=float)
    parser.add_argument('--use_mean_pooling', action='store_true')
    parser.set_defaults(use_mean_pooling=True)
    parser.add_argument('--use_cls', action='store_false', dest='use_mean_pooling')
    parser.add_argument('--disable_weight_decay_on_rel_pos_bias', action='store_true', default=False)

    # Dataset parameters
    parser.add_argument('--nb_classes', default=0, type=int,
                        help='number of the classification types')

    parser.add_argument('--output_dir', default='',
                        help='path where to save, empty for no saving')
    parser.add_argument('--log_dir', default=None,
                        help='path where to tensorboard log')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default='',
                        help='resume from checkpoint')
    parser.add_argument('--auto_resume', action='store_true')
    parser.add_argument('--no_auto_resume', action='store_false', dest='auto_resume')
    parser.set_defaults(auto_resume=True)

    parser.add_argument('--save_ckpt', action='store_true')
    parser.add_argument('--no_save_ckpt', action='store_false', dest='save_ckpt')
    parser.set_defaults(save_ckpt=True)

    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', action='store_true',
                        help='Perform evaluation only')
    parser.add_argument('--dist_eval', action='store_true', default=False,
                        help='Enabling distributed evaluation')
    parser.add_argument('--num_workers', default=10, type=int)
    parser.add_argument('--pin_mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
    parser.set_defaults(pin_mem=True)

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://',
                        help='url used to set up distributed training')

    parser.add_argument('--enable_deepspeed', action='store_true', default=False)
    parser.add_argument('--dataset', default='TUAB', type=str,
                        help='dataset: TUAB | TUEV')

    # ====================== [NEW] 仅在开启被试内CV时使用的参数 ======================
    parser.add_argument('--within_subject_cv', action='store_true', default=False,
                        help='Enable within-subject K-fold CV on processed PKL datasets.')
    parser.add_argument('--pkl_roots', type=str, default='',
                        help='Comma-separated roots that contain processed .pkl files.')
    parser.add_argument('--subject_regex', type=str, default=r'sub_(\d+)_simplified',
                        help='Regex to extract subject id from file name.')
    parser.add_argument('--cv_splits', type=int, default=5,
                        help='Number of folds for within-subject CV.')
    parser.add_argument('--channels_upper_csv', type=str, default='',
                        help='Optional CSV with one row listing 64 channel names; will be upper-cased and stripped.')

    known_args, _ = parser.parse_known_args()

    if known_args.enable_deepspeed:
        try:
            import deepspeed
            from deepspeed import DeepSpeedConfig
            parser = deepspeed.add_config_arguments(parser)
            ds_init = deepspeed.initialize
        except:
            print("Please 'pip install deepspeed==0.4.0'")
            exit(0)
    else:
        ds_init = None

    return parser.parse_args(), ds_init

# ====================== 原有模型构建 ======================
def get_models(args):
    model = create_model(
        args.model,
        pretrained=False,
        num_classes=args.nb_classes,
        drop_rate=args.drop,
        drop_path_rate=args.drop_path,
        attn_drop_rate=args.attn_drop_rate,
        drop_block_rate=None,
        use_mean_pooling=args.use_mean_pooling,
        init_scale=args.init_scale,
        use_rel_pos_bias=args.rel_pos_bias,
        use_abs_pos_emb=args.abs_pos_emb,
        init_values=args.layer_scale_init_value,
        qkv_bias=args.qkv_bias,
    )
    return model

# ====================== 原有数据集选择（不改） ======================
def get_dataset(args):
    if args.dataset == 'TUAB':
        train_dataset, test_dataset, val_dataset = utils.prepare_TUAB_dataset("path/to/TUAB")
        ch_names = ['EEG FP1', 'EEG FP2-REF', 'EEG F3-REF', 'EEG F4-REF', 'EEG C3-REF', 'EEG C4-REF', 'EEG P3-REF', 'EEG P4-REF', 'EEG O1-REF', 'EEG O2-REF', 'EEG F7-REF', \
                    'EEG F8-REF', 'EEG T3-REF', 'EEG T4-REF', 'EEG T5-REF', 'EEG T6-REF', 'EEG A1-REF', 'EEG A2-REF', 'EEG FZ-REF', 'EEG CZ-REF', 'EEG PZ-REF', 'EEG T1-REF', 'EEG T2-REF']
        ch_names = [name.split(' ')[-1].split('-')[0] for name in ch_names]
        args.nb_classes = 1
        metrics = ["pr_auc", "roc_auc", "accuracy", "balanced_accuracy"]
    elif args.dataset == 'TUEV':
        train_dataset, test_dataset, val_dataset = utils.prepare_TUEV_dataset("path/to/TUEV")
        ch_names = ['EEG FP1-REF', 'EEG FP2-REF', 'EEG F3-REF', 'EEG F4-REF', 'EEG C3-REF', 'EEG C4-REF', 'EEG P3-REF', 'EEG P4-REF', 'EEG O1-REF', 'EEG O2-REF', 'EEG F7-REF', \
                    'EEG F8-REF', 'EEG T3-REF', 'EEG T4-REF', 'EEG T5-REF', 'EEG T6-REF', 'EEG A1-REF', 'EEG A2-REF', 'EEG FZ-REF', 'EEG CZ-REF', 'EEG PZ-REF', 'EEG T1-REF', 'EEG T2-REF']
        ch_names = [name.split(' ')[-1].split('-')[0] for name in ch_names]
        args.nb_classes = 6
        metrics = ["accuracy", "balanced_accuracy", "cohen_kappa", "f1_weighted"]
    return train_dataset, test_dataset, val_dataset, ch_names, metrics

# ====================== [NEW] 被试内CV所需的轻量Dataset ======================
class PKLSegDataset(Dataset):
    """惰性加载的 PKL 数据集：每个样本是 (path, y)，__getitem__ 时读取 {"X","y"}."""
    def __init__(self, samples):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        p, y = self.samples[idx]
        with open(p, 'rb') as f:
            obj = pickle.load(f)
        x = obj['X']  # (C, T_any)
        yy = int(obj['y']) if not isinstance(obj['y'], (int, np.integer)) else int(obj['y'])

        x = np.asarray(x)
        # === [FIX] 确保时间长度为 200 的整数倍，匹配 engine 的 T=200 ===
        T = 200
        L = x.shape[-1]
        if L % T != 0:
            L2 = (L // T) * T
            if L2 <= 0:
                # 极端短片段：用 0 补到 200
                pad = T - L
                x = np.pad(x, ((0, 0), (0, pad)), mode='constant')
            else:
                # 常见情况（如 2101）：裁掉尾部余数 -> 2100
                x = x[..., :L2]

        x = torch.from_numpy(x).float()   # (C, T')
        return x, yy


# ====================== [NEW] 工具：扫描样本，分被试聚合 ======================
def _load_channels_from_csv(path):
    with open(path, 'r', encoding='utf-8') as f:
        row = next(csv.reader(f))
    return [c.replace(' ', '').upper() for c in row]

def _default_channels_upper():
    # 来自你提供的顺序，做了去空格+大写
    base = ['Fp1','AF7','AF3','F1','F3','F5','F7','FT7','FC5','FC3','FC1','C1','C3','C5','T7','TP7','CP5','CP3','CP1','P1','P3','P5','P7','P9',
            'PO7','PO3','O1','Iz','Oz','POz','Pz','CPz','Fpz','Fp2','AF8','AF4','AFz','Fz','F2','F4','F6','F8','FT8','FC6','FC4','FC2','FCz',
            'Cz','C2','C4','C6','T8','TP8','CP6','CP4','CP2','P2','P4','P6','P8','P10','PO8','PO4','O2']
    return [c.upper() for c in base]

def _scan_pkl_roots(pkl_roots, subject_regex):
    roots = [r.strip() for r in pkl_roots.split(',') if r.strip()]
    pat = re.compile(subject_regex, flags=re.I)
    subj_to_samples = defaultdict(list)  # subj_id -> list of (path, y)
    all_labels = []

    for root in roots:
        rootp = Path(root)
        if not rootp.exists(): 
            continue
        for p in rootp.rglob('*.pkl'):
            m = pat.search(p.name)
            if not m:
                continue
            sid = m.group(1)
            # 为了分层K折，需要label；读取一次
            try:
                with open(p, 'rb') as f:
                    obj = pickle.load(f)
                y = int(obj['y']) if not isinstance(obj['y'], (int, np.integer)) else int(obj['y'])
            except Exception as e:
                # 跳过坏样本
                continue
            subj_to_samples[sid].append((str(p), y))
            all_labels.append(y)
    return subj_to_samples, sorted(set(all_labels))

# ====================== [NEW] 仅主进程写 + 增量更新cv_summary.csv ======================
def _is_main_process_safe():
    try:
        return utils.is_main_process()
    except Exception:
        # 兜底：环境变量 RANK
        try:
            return int(os.environ.get("RANK", "0")) == 0
        except Exception:
            return True

def _incremental_update_cv(csv_path: str, row: dict, metrics: list):
    """
    按 subject 去重覆盖写入；并实时重算 OVERALL。
    Header = ["subject"] + metrics + [m+"_std" for m in metrics]
    """
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    fields = ["subject"] + metrics + [m + "_std" for m in metrics]

    # 读取已有内容
    rows = []
    if os.path.exists(csv_path):
        with open(csv_path, "r", encoding="utf-8", newline="") as rf:
            try:
                import csv as _csv
                r = _csv.DictReader(rf)
                if r.fieldnames:
                    for x in r:
                        rows.append(x)
            except Exception:
                rows = []

    # 过滤掉旧的同 subject、以及旧的 OVERALL
    subj = str(row["subject"])
    rows = [r for r in rows if r.get("subject") not in (subj, "OVERALL")]

    # 追加当前 subject 行
    def _fmt(v):
        if v is None: return ""
        if isinstance(v, float): return f"{v:.12g}"
        return str(v)

    rows.append({k: _fmt(row.get(k, "")) for k in fields})

    # 生成 OVERALL
    def _to_float_safe(v):
        try:
            return float(v)
        except Exception:
            return float("nan")

    valid_rows = [r for r in rows if r.get("subject") not in ("OVERALL", "")]
    overall = {"subject": "OVERALL"}
    import numpy as _np
    for m in metrics:
        vals = _np.array([_to_float_safe(r.get(m, "nan")) for r in valid_rows], dtype=float)
        vals = vals[~_np.isnan(vals)]
        overall[m] = "" if vals.size == 0 else f"{_np.mean(vals):.12g}"
        # OVERALL 的 *_std 给出各 subject 均值的 std（可按需要改成空字符串）
        overall[m + "_std"] = "" if vals.size == 0 else f"{_np.std(vals):.12g}"
    rows.append(overall)

    # 按 subject 数字序排序（OVERALL 放最后）
    def _key(r):
        s = r.get("subject", "")
        if s == "OVERALL": return (1e18,)
        digits = "".join(ch for ch in s if ch.isdigit())
        try:
            return (int(digits),)
        except Exception:
            return (s,)

    rows.sort(key=_key)

    # 加锁写回
    with open(csv_path, "w", encoding="utf-8", newline="") as wf:
        fcntl.flock(wf, fcntl.LOCK_EX)
        import csv as _csv
        w = _csv.DictWriter(wf, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
        wf.flush(); os.fsync(wf.fileno())
        fcntl.flock(wf, fcntl.LOCK_UN)

# ====================== [NEW] 每折一次完整训练（尽量复用原逻辑） ======================
def _run_one_fold(args, ds_init, ch_names_upper, metrics, fold_id, subject_id,
                  train_samples, val_samples, test_samples):
    """返回 (val_stats, test_stats) 两个 dict。"""
    # utils.init_distributed_mode(args)

    device = torch.device(args.device)

    # 固定随机种子（与官方一致）
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    cudnn.benchmark = True

    # 构建 DataLoader（与原代码保持方式一致）
    dataset_train = PKLSegDataset(train_samples)
    dataset_val   = PKLSegDataset(val_samples)
    dataset_test  = PKLSegDataset(test_samples)

    if args.disable_eval_during_finetuning:
        dataset_val = None
        dataset_test = None

    if True:  # args.distributed:
        num_tasks = utils.get_world_size()
        global_rank = utils.get_rank()
        sampler_train = torch.utils.data.DistributedSampler(
            dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
        )
        if dataset_val is not None:
            if args.dist_eval:
                if len(dataset_val) % num_tasks != 0:
                    print('Warning: Dist eval with eval set not divisible by number of processes.')
                sampler_val = torch.utils.data.DistributedSampler(
                    dataset_val, num_replicas=num_tasks, rank=global_rank, shuffle=False)
                sampler_test = torch.utils.data.DistributedSampler(
                    dataset_test, num_replicas=num_tasks, rank=global_rank, shuffle=False)
            else:
                sampler_val = torch.utils.data.SequentialSampler(dataset_val)
                sampler_test = torch.utils.data.SequentialSampler(dataset_test)
        else:
            sampler_val = None
            sampler_test = None
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)
        sampler_val = torch.utils.data.SequentialSampler(dataset_val) if dataset_val is not None else None
        sampler_test = torch.utils.data.SequentialSampler(dataset_test) if dataset_test is not None else None

    log_writer = None
    if utils.is_main_process() and args.log_dir is not None:
        os.makedirs(args.log_dir, exist_ok=True)
        log_writer = utils.TensorboardLogger(log_dir=args.log_dir)

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False,
    )

    if dataset_val is not None:
        data_loader_val = torch.utils.data.DataLoader(
            dataset_val, sampler=sampler_val,
            batch_size=int(1.5 * args.batch_size),
            num_workers=args.num_workers,
            pin_memory=args.pin_mem,
            drop_last=False
        )
        data_loader_test = torch.utils.data.DataLoader(
            dataset_test, sampler=sampler_test,
            batch_size=int(1.5 * args.batch_size),
            num_workers=args.num_workers,
            pin_memory=args.pin_mem,
            drop_last=False
        )
    else:
        data_loader_val = None
        data_loader_test = None

    # ====== 模型与优化器（与原逻辑相同，每折重建，避免信息泄漏） ======
    model = get_models(args)
    patch_size = model.patch_size
    print(f"[S{subject_id} F{fold_id}] Patch size = {patch_size}")
    args.window_size = (1, args.input_size // patch_size)
    args.patch_size = patch_size

    if args.finetune:
        if args.finetune.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.finetune, map_location='cpu', check_hash=True)
        else:
            checkpoint = torch.load(args.finetune, map_location='cpu')

        print("Load ckpt from %s" % args.finetune)
        checkpoint_model = None
        for model_key in args.model_key.split('|'):
            if model_key in checkpoint:
                checkpoint_model = checkpoint[model_key]
                print("Load state_dict by model_key = %s" % model_key)
                break
        if checkpoint_model is None:
            checkpoint_model = checkpoint
        if (checkpoint_model is not None) and (args.model_filter_name != ''):
            all_keys = list(checkpoint_model.keys())
            new_dict = OrderedDict()
            for key in all_keys:
                if key.startswith('student.'):
                    new_dict[key[8:]] = checkpoint_model[key]
                else:
                    pass
            checkpoint_model = new_dict

        state_dict = model.state_dict()
        for k in ['head.weight', 'head.bias']:
            if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
                print(f"Removing key {k} from pretrained checkpoint")
                del checkpoint_model[k]

        all_keys = list(checkpoint_model.keys())
        for key in all_keys:
            if "relative_position_index" in key:
                checkpoint_model.pop(key)

        utils.load_state_dict(model, checkpoint_model, prefix=args.model_prefix)

    device = torch.device(args.device)
    model.to(device)

    model_ema = None
    if args.model_ema:
        model_ema = ModelEma(
            model,
            decay=args.model_ema_decay,
            device='cpu' if args.model_ema_force_cpu else '',
            resume='')
        print("Using EMA with decay = %.8f" % args.model_ema_decay)

    model_without_ddp = model
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # print("Model = %s" % str(model_without_ddp))
    print('number of params:', n_parameters)

    total_batch_size = args.batch_size * args.update_freq * utils.get_world_size()
    num_training_steps_per_epoch = max(1, len(dataset_train) // total_batch_size)
    print("LR = %.8f" % args.lr)
    print("Batch size = %d" % total_batch_size)
    print("Update frequent = %d" % args.update_freq)
    print("Number of training examples = %d" % len(dataset_train))
    print("Number of training training per epoch = %d" % num_training_steps_per_epoch)

    num_layers = model_without_ddp.get_num_layers()
    if args.layer_decay < 1.0:
        assigner = LayerDecayValueAssigner(list(args.layer_decay ** (num_layers + 1 - i) for i in range(num_layers + 2)))
    else:
        assigner = None

    if assigner is not None:
        print("Assigned values = %s" % str(assigner.values))

    skip_weight_decay_list = model.no_weight_decay()
    if args.disable_weight_decay_on_rel_pos_bias:
        for i in range(num_layers):
            skip_weight_decay_list.add("blocks.%d.attn.relative_position_bias_table" % i)

    if args.enable_deepspeed:
        loss_scaler = None
        optimizer_params = get_parameter_groups(
            model, args.weight_decay, skip_weight_decay_list,
            assigner.get_layer_id if assigner is not None else None,
            assigner.get_scale if assigner is not None else None)
        model, optimizer, _, _ = ds_init(
            args=args, model=model, model_parameters=optimizer_params, dist_init_required=not args.distributed,
        )

        print("model.gradient_accumulation_steps() = %d" % model.gradient_accumulation_steps())
        assert model.gradient_accumulation_steps() == args.update_freq
    else:
        if args.distributed:
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
            model_without_ddp = model.module

        optimizer = create_optimizer(
            args, model_without_ddp, skip_list=skip_weight_decay_list,
            get_num_layer=assigner.get_layer_id if assigner is not None else None, 
            get_layer_scale=assigner.get_scale if assigner is not None else None)
        loss_scaler = NativeScaler()

    print("Use step level LR scheduler!")
    lr_schedule_values = utils.cosine_scheduler(
        args.lr, args.min_lr, args.epochs, num_training_steps_per_epoch,
        warmup_epochs=args.warmup_epochs, warmup_steps=args.warmup_steps,
    )
    if args.weight_decay_end is None:
        args.weight_decay_end = args.weight_decay
    wd_schedule_values = utils.cosine_scheduler(
        args.weight_decay, args.weight_decay_end, args.epochs, num_training_steps_per_epoch)
    print("Max WD = %.7f, Min WD = %.7f" % (max(wd_schedule_values), min(wd_schedule_values)))

    if args.nb_classes == 1:
        criterion = torch.nn.BCEWithLogitsLoss()
    elif args.smoothing > 0.:
        criterion = LabelSmoothingCrossEntropy(smoothing=args.smoothing)
    else:
        criterion = torch.nn.CrossEntropyLoss()

    print("criterion = %s" % str(criterion))

    utils.auto_load_model(
        args=args, model=model, model_without_ddp=model_without_ddp,
        optimizer=optimizer, loss_scaler=loss_scaler, model_ema=model_ema)
            
    # ====== 单折训练 ======
    print(f"[S{subject_id} F{fold_id}] Start training for {args.epochs} epochs")
    start_time = time.time()
    max_accuracy = 0.0
    best_val_stats = None
    best_test_stats = None

    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch)
        if log_writer is not None:
            log_writer.set_step(epoch * num_training_steps_per_epoch * args.update_freq)

        train_stats = train_one_epoch(
            model, criterion, data_loader_train, optimizer,
            device, epoch, loss_scaler, args.clip_grad, model_ema,
            log_writer=log_writer, start_steps=epoch * num_training_steps_per_epoch,
            lr_schedule_values=lr_schedule_values, wd_schedule_values=wd_schedule_values,
            num_training_steps_per_epoch=num_training_steps_per_epoch, update_freq=args.update_freq,
            ch_names=ch_names_upper, is_binary=args.nb_classes == 1, verbose=False
        )

        if args.output_dir and args.save_ckpt:
            utils.save_model(
                args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                loss_scaler=loss_scaler, epoch=epoch, model_ema=model_ema, save_ckpt_freq=args.save_ckpt_freq)

        if data_loader_val is not None:
            val_stats = evaluate(data_loader_val, model, device, header=f'Val[S{subject_id} F{fold_id}]:',
                                 ch_names=ch_names_upper, metrics=metrics, is_binary=args.nb_classes == 1,
                                 verbose=False)
            test_stats = evaluate(data_loader_test, model, device, header=f'Test[S{subject_id} F{fold_id}]:',
                                  ch_names=ch_names_upper, metrics=metrics, is_binary=args.nb_classes == 1,
                                  verbose=False)
            print(f"[S{subject_id} F{fold_id}] Val Acc: {val_stats.get('accuracy', 0):.2f}%, Test Acc: {test_stats.get('accuracy', 0):.2f}%")
            
            if max_accuracy < val_stats.get("accuracy", 0):
                max_accuracy = val_stats["accuracy"]
                best_val_stats = val_stats
                best_test_stats = test_stats
                if args.output_dir and args.save_ckpt:
                    # 1) 继续按整数 epoch 交给官方 save_model（避免 TypeError）
                    utils.save_model(
                        args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                        loss_scaler=loss_scaler, epoch=epoch, model_ema=model_ema)

                    # 2) 另外手动再保存一份带“best_Sx_Fy”标签的权重（便于区分/复现）
                    if utils.is_main_process():
                        best_tag = f"checkpoint_best_S{subject_id}_F{fold_id}.pth"
                        best_path = os.path.join(args.output_dir, best_tag)
                        torch.save(
                            {
                                "model": model_without_ddp.state_dict(),
                                "epoch": epoch,
                                "subject": subject_id,
                                "fold": fold_id,
                                "val_metrics": best_val_stats,
                                "test_metrics": best_test_stats,
                            },
                            best_path
                        )
                        print(f"[S{subject_id} F{fold_id}] Saved tagged best checkpoint -> {best_path}")

            if log_writer is not None:
                # 关键指标写入
                for key, value in val_stats.items():
                    log_writer.update(**{key: value}, head=f"val_S{subject_id}_F{fold_id}", step=epoch)
                for key, value in test_stats.items():
                    log_writer.update(**{key: value}, head=f"test_S{subject_id}_F{fold_id}", step=epoch)

            log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                         **{f'val_{k}': v for k, v in val_stats.items()},
                         **{f'test_{k}': v for k, v in test_stats.items()},
                         'epoch': epoch,
                         'n_parameters': n_parameters,
                         'subject': subject_id,
                         'fold': fold_id}
        else:
            log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                         'epoch': epoch,
                         'n_parameters': n_parameters,
                         'subject': subject_id,
                         'fold': fold_id}

        if args.output_dir and utils.is_main_process():
            if log_writer is not None:
                log_writer.flush()
            with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                f.write(json.dumps(log_stats) + "\n")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print(f'[S{subject_id} F{fold_id}] Training time {total_time_str}')

    if best_val_stats is None and data_loader_val is not None:
        # 未触发更新则最后一次作为 best 的统计（注意：这不会额外保存 best 权重）
        best_val_stats = val_stats
        best_test_stats = test_stats

    # === 统一写入完成标记（与每折独立目录配合，实现“已完成就跳过”） ===
    try:
        if args.output_dir and utils.is_main_process():
            done_path = os.path.join(args.output_dir, "done.json")
            with open(done_path, "w", encoding="utf-8") as f:
                json.dump({
                    "subject": subject_id,
                    "fold": fold_id,
                    "finished": True,
                    "best_val": best_val_stats if best_val_stats is not None else None,
                    "best_test": best_test_stats if best_test_stats is not None else None,
                }, f, ensure_ascii=False)
            print(f"[S{subject_id} F{fold_id}] Wrote completion marker -> {done_path}")
    except Exception as _e:
        # 标记失败不影响训练流程
        print(f"[S{subject_id} F{fold_id}] WARN: failed to write done.json: {_e}")

    return best_val_stats, best_test_stats


# ====================== 主入口 ======================
def main(args, ds_init):
    # ---------- 常规路径：不启用被试内CV，完全保留官方逻辑 ----------
    if not args.within_subject_cv:
        utils.init_distributed_mode(args)

        if ds_init is not None:
            utils.create_ds_config(args)

        print(args)

        device = torch.device(args.device)

        seed = args.seed + utils.get_rank()
        torch.manual_seed(seed)
        np.random.seed(seed)
        cudnn.benchmark = True

        dataset_train, dataset_test, dataset_val, ch_names, metrics = get_dataset(args)

        if args.disable_eval_during_finetuning:
            dataset_val = None
            dataset_test = None

        if True:  # args.distributed:
            num_tasks = utils.get_world_size()
            global_rank = utils.get_rank()
            sampler_train = torch.utils.data.DistributedSampler(
                dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
            )
            print("Sampler_train = %s" % str(sampler_train))
            if args.dist_eval:
                if len(dataset_val) % num_tasks != 0:
                    print('Warning: Enabling distributed evaluation with an eval dataset not divisible by process number.')
                sampler_val = torch.utils.data.DistributedSampler(
                    dataset_val, num_replicas=num_tasks, rank=global_rank, shuffle=False)
                if type(dataset_test) == list:
                    sampler_test = [torch.utils.data.DistributedSampler(
                        dataset, num_replicas=num_tasks, rank=global_rank, shuffle=False) for dataset in dataset_test]
                else:
                    sampler_test = torch.utils.data.DistributedSampler(
                        dataset_test, num_replicas=num_tasks, rank=global_rank, shuffle=False)
            else:
                sampler_val = torch.utils.data.SequentialSampler(dataset_val)
                sampler_test = torch.utils.data.SequentialSampler(dataset_test)
        else:
            sampler_train = torch.utils.data.RandomSampler(dataset_train)
            sampler_val = torch.utils.data.SequentialSampler(dataset_val)

        if utils.is_main_process() and args.log_dir is not None:
            os.makedirs(args.log_dir, exist_ok=True)
            log_writer = utils.TensorboardLogger(log_dir=args.log_dir)
        else:
            log_writer = None

        data_loader_train = torch.utils.data.DataLoader(
            dataset_train, sampler=sampler_train,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=args.pin_mem,
            drop_last=True,
        )

        if dataset_val is not None:
            data_loader_val = torch.utils.data.DataLoader(
                dataset_val, sampler=sampler_val,
                batch_size=int(1.5 * args.batch_size),
                num_workers=args.num_workers,
                pin_memory=args.pin_mem,
                drop_last=False
            )
            if type(dataset_test) == list:
                data_loader_test = [torch.utils.data.DataLoader(
                    dataset, sampler=sampler,
                    batch_size=int(1.5 * args.batch_size),
                    num_workers=args.num_workers,
                    pin_memory=args.pin_mem,
                    drop_last=False
                ) for dataset, sampler in zip(dataset_test, sampler_test)]
            else:
                data_loader_test = torch.utils.data.DataLoader(
                    dataset_test, sampler=sampler_test,
                    batch_size=int(1.5 * args.batch_size),
                    num_workers=args.num_workers,
                    pin_memory=args.pin_mem,
                    drop_last=False
                )
        else:
            data_loader_val = None
            data_loader_test = None

        model = get_models(args)

        patch_size = model.patch_size
        print("Patch size = %s" % str(patch_size))
        args.window_size = (1, args.input_size // patch_size)
        args.patch_size = patch_size

        if args.finetune:
            if args.finetune.startswith('https'):
                checkpoint = torch.hub.load_state_dict_from_url(
                    args.finetune, map_location='cpu', check_hash=True)
            else:
                checkpoint = torch.load(args.finetune, map_location='cpu')

            print("Load ckpt from %s" % args.finetune)
            checkpoint_model = None
            for model_key in args.model_key.split('|'):
                if model_key in checkpoint:
                    checkpoint_model = checkpoint[model_key]
                    print("Load state_dict by model_key = %s" % model_key)
                    break
            if checkpoint_model is None:
                checkpoint_model = checkpoint
            if (checkpoint_model is not None) and (args.model_filter_name != ''):
                all_keys = list(checkpoint_model.keys())
                new_dict = OrderedDict()
                for key in all_keys:
                    if key.startswith('student.'):
                        new_dict[key[8:]] = checkpoint_model[key]
                    else:
                        pass
                checkpoint_model = new_dict

            state_dict = model.state_dict()
            for k in ['head.weight', 'head.bias']:
                if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
                    print(f"Removing key {k} from pretrained checkpoint")
                    del checkpoint_model[k]

            all_keys = list(checkpoint_model.keys())
            for key in all_keys:
                if "relative_position_index" in key:
                    checkpoint_model.pop(key)

            utils.load_state_dict(model, checkpoint_model, prefix=args.model_prefix)

        model.to(device)

        model_ema = None
        if args.model_ema:
            model_ema = ModelEma(
                model,
                decay=args.model_ema_decay,
                device='cpu' if args.model_ema_force_cpu else '',
                resume='')
            print("Using EMA with decay = %.8f" % args.model_ema_decay)

        model_without_ddp = model
        n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)

        print("Model = %s" % str(model_without_ddp))
        print('number of params:', n_parameters)

        total_batch_size = args.batch_size * args.update_freq * utils.get_world_size()
        num_training_steps_per_epoch = len(dataset_train) // total_batch_size
        print("LR = %.8f" % args.lr)
        print("Batch size = %d" % total_batch_size)
        print("Update frequent = %d" % args.update_freq)
        print("Number of training examples = %d" % len(dataset_train))
        print("Number of training training per epoch = %d" % num_training_steps_per_epoch)

        num_layers = model_without_ddp.get_num_layers()
        if args.layer_decay < 1.0:
            assigner = LayerDecayValueAssigner(list(args.layer_decay ** (num_layers + 1 - i) for i in range(num_layers + 2)))
        else:
            assigner = None

        if assigner is not None:
            print("Assigned values = %s" % str(assigner.values))

        skip_weight_decay_list = model.no_weight_decay()
        if args.disable_weight_decay_on_rel_pos_bias:
            for i in range(num_layers):
                skip_weight_decay_list.add("blocks.%d.attn.relative_position_bias_table" % i)

        if args.enable_deepspeed:
            loss_scaler = None
            optimizer_params = get_parameter_groups(
                model, args.weight_decay, skip_weight_decay_list,
                assigner.get_layer_id if assigner is not None else None,
                assigner.get_scale if assigner is not None else None)
            model, optimizer, _, _ = ds_init(
                args=args, model=model, model_parameters=optimizer_params, dist_init_required=not args.distributed,
            )

            print("model.gradient_accumulation_steps() = %d" % model.gradient_accumulation_steps())
            assert model.gradient_accumulation_steps() == args.update_freq
        else:
            if args.distributed:
                model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
                model_without_ddp = model.module

            optimizer = create_optimizer(
                args, model_without_ddp, skip_list=skip_weight_decay_list,
                get_num_layer=assigner.get_layer_id if assigner is not None else None, 
                get_layer_scale=assigner.get_scale if assigner is not None else None)
            loss_scaler = NativeScaler()

        print("Use step level LR scheduler!")
        lr_schedule_values = utils.cosine_scheduler(
            args.lr, args.min_lr, args.epochs, num_training_steps_per_epoch,
            warmup_epochs=args.warmup_epochs, warmup_steps=args.warmup_steps,
        )
        if args.weight_decay_end is None:
            args.weight_decay_end = args.weight_decay
        wd_schedule_values = utils.cosine_scheduler(
            args.weight_decay, args.weight_decay_end, args.epochs, num_training_steps_per_epoch)
        print("Max WD = %.7f, Min WD = %.7f" % (max(wd_schedule_values), min(wd_schedule_values)))

        if args.nb_classes == 1:
            criterion = torch.nn.BCEWithLogitsLoss()
        elif args.smoothing > 0.:
            criterion = LabelSmoothingCrossEntropy(smoothing=args.smoothing)
        else:
            criterion = torch.nn.CrossEntropyLoss()

        print("criterion = %s" % str(criterion))

        utils.auto_load_model(
            args=args, model=model, model_without_ddp=model_without_ddp,
            optimizer=optimizer, loss_scaler=loss_scaler, model_ema=model_ema)
                
        if args.eval:
            balanced_accuracy = []
            accuracy = []
            for data_loader in data_loader_test:
                test_stats = evaluate(data_loader, model, device, header='Test:', ch_names=ch_names, metrics=metrics, is_binary=(args.nb_classes == 1))
                accuracy.append(test_stats['accuracy'])
                balanced_accuracy.append(test_stats['balanced_accuracy'])
            print(f"======Accuracy: {np.mean(accuracy)} {np.std(accuracy)}, balanced accuracy: {np.mean(balanced_accuracy)} {np.std(balanced_accuracy)}")
            exit(0)

        print(f"Start training for {args.epochs} epochs")
        start_time = time.time()
        max_accuracy = 0.0
        max_accuracy_test = 0.0
        for epoch in range(args.start_epoch, args.epochs):
            if args.distributed:
                data_loader_train.sampler.set_epoch(epoch)
            if log_writer is not None:
                log_writer.set_step(epoch * num_training_steps_per_epoch * args.update_freq)
            train_stats = train_one_epoch(
                model, criterion, data_loader_train, optimizer,
                device, epoch, loss_scaler, args.clip_grad, model_ema,
                log_writer=log_writer, start_steps=epoch * num_training_steps_per_epoch,
                lr_schedule_values=lr_schedule_values, wd_schedule_values=wd_schedule_values,
                num_training_steps_per_epoch=num_training_steps_per_epoch, update_freq=args.update_freq, 
                ch_names=ch_names, is_binary=args.nb_classes == 1
            )
            
            if args.output_dir and args.save_ckpt:
                utils.save_model(
                    args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                    loss_scaler=loss_scaler, epoch=epoch, model_ema=model_ema, save_ckpt_freq=args.save_ckpt_freq)
                
            if data_loader_val is not None:
                val_stats = evaluate(data_loader_val, model, device, header='Val:', ch_names=ch_names, metrics=metrics, is_binary=args.nb_classes == 1)
                print(f"Accuracy of the network on the {len(dataset_val)} val EEG: {val_stats['accuracy']:.2f}%")
                test_stats = evaluate(data_loader_test, model, device, header='Test:', ch_names=ch_names, metrics=metrics, is_binary=args.nb_classes == 1)
                print(f"Accuracy of the network on the {len(dataset_test)} test EEG: {test_stats['accuracy']:.2f}%")
                
                if max_accuracy < val_stats["accuracy"]:
                    max_accuracy = val_stats["accuracy"]
                    if args.output_dir and args.save_ckpt:
                        utils.save_model(
                            args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                            loss_scaler=loss_scaler, epoch="best", model_ema=model_ema)
                    max_accuracy_test = test_stats["accuracy"]

                print(f'Max accuracy val: {max_accuracy:.2f}%, max accuracy test: {max_accuracy_test:.2f}%')
                if log_writer is not None:
                    for key, value in val_stats.items():
                        if key == 'accuracy':
                            log_writer.update(accuracy=value, head="val", step=epoch)
                        elif key == 'balanced_accuracy':
                            log_writer.update(balanced_accuracy=value, head="val", step=epoch)
                        elif key == 'f1_weighted':
                            log_writer.update(f1_weighted=value, head="val", step=epoch)
                        elif key == 'pr_auc':
                            log_writer.update(pr_auc=value, head="val", step=epoch)
                        elif key == 'roc_auc':
                            log_writer.update(roc_auc=value, head="val", step=epoch)
                        elif key == 'cohen_kappa':
                            log_writer.update(cohen_kappa=value, head="val", step=epoch)
                        elif key == 'loss':
                            log_writer.update(loss=value, head="val", step=epoch)
                    for key, value in test_stats.items():
                        if key == 'accuracy':
                            log_writer.update(accuracy=value, head="test", step=epoch)
                        elif key == 'balanced_accuracy':
                            log_writer.update(balanced_accuracy=value, head="test", step=epoch)
                        elif key == 'f1_weighted':
                            log_writer.update(f1_weighted=value, head="test", step=epoch)
                        elif key == 'pr_auc':
                            log_writer.update(pr_auc=value, head="test", step=epoch)
                        elif key == 'roc_auc':
                            log_writer.update(roc_auc=value, head="test", step=epoch)
                        elif key == 'cohen_kappa':
                            log_writer.update(cohen_kappa=value, head="test", step=epoch)
                        elif key == 'loss':
                            log_writer.update(loss=value, head="test", step=epoch)
                    
                log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                             **{f'val_{k}': v for k, v in val_stats.items()},
                             **{f'test_{k}': v for k, v in test_stats.items()},
                             'epoch': epoch,
                             'n_parameters': n_parameters}
            else:
                log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                             'epoch': epoch,
                             'n_parameters': n_parameters}

            if args.output_dir and utils.is_main_process():
                if log_writer is not None:
                    log_writer.flush()
                with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                    f.write(json.dumps(log_stats) + "\n")

        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print('Training time {}'.format(total_time_str))
        return


    # ---------- 被试内CV路径：新增逻辑 ----------
    # [FIX] 在CV总入口只初始化一次分布式，后续各fold沿用同一进程组
    utils.init_distributed_mode(args)
    if ds_init is not None:
        utils.create_ds_config(args)

    print(args)
    device = torch.device(args.device)
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    cudnn.benchmark = True
    # ---------- 被试内CV路径：新增逻辑（仅当 --within_subject_cv 打开时执行） ----------
    assert args.pkl_roots, "--within_subject_cv 需要指定 --pkl_roots（逗号分隔多个根目录）"
    subj_to_samples, label_set = _scan_pkl_roots(args.pkl_roots, args.subject_regex)
    assert len(subj_to_samples) > 0, f"未在 {args.pkl_roots} 找到任何符合命名与内容的 .pkl 文件"

    # [NEW] 汇总并打印检测到的被试与样本总数（仅主进程）
    num_subjects = len(subj_to_samples)
    total_samples = sum(len(v) for v in subj_to_samples.values())
    if _is_main_process_safe():
        # 仅打印前若干个被试以免日志过长
        _subj_sorted = sorted(subj_to_samples.keys(), key=lambda s: int(s) if str(s).isdigit() else s)
        _preview = ", ".join(_subj_sorted)
        print(f"[CV] Detected {num_subjects} subjects, total {total_samples} samples.")
        print(f"[CV] Subjects preview: {_preview}")

        # 可选：把被试清单与每被试样本数写入 output_dir/subject_index.json（若指定了 output_dir）
        try:
            if args.output_dir:
                os.makedirs(args.output_dir, exist_ok=True)
                _idx = {
                    "num_subjects": num_subjects,
                    "total_samples": total_samples,
                    "subjects": [
                        {"subject": sid, "num_samples": len(subj_to_samples[sid])}
                        for sid in _subj_sorted
                    ]
                }
                with open(os.path.join(args.output_dir, "subject_index.json"), "w", encoding="utf-8") as _f:
                    json.dump(_idx, _f, ensure_ascii=False, indent=2)
                print(f"[CV] subject_index.json written -> {os.path.join(args.output_dir, 'subject_index.json')}")
        except Exception as _e:
            print(f"[CV] WARN: failed to write subject_index.json: {_e}")

    # 自动设置 nb_classes
    if args.nb_classes == 0:
        args.nb_classes = len(label_set)

    # 通道名（大写、去空格）
    if args.channels_upper_csv:
        ch_names_upper = _load_channels_from_csv(args.channels_upper_csv)
    else:
        ch_names_upper = _default_channels_upper()

    # 指标沿用 TUAB / TUEV 的风格；若二分类，含 AUC；多分类给常见分类指标
    metrics = ["accuracy", "balanced_accuracy"]
    if args.nb_classes == 1:
        metrics = ["pr_auc", "roc_auc", "accuracy", "balanced_accuracy"]
    elif args.nb_classes >= 2:
        metrics = ["accuracy", "balanced_accuracy", "f1_weighted"]

    # 汇总容器
    subject_rows = []

    # 遍历每个被试
    for sub_id, samples in sorted(subj_to_samples.items(), key=lambda x: int(x[0]) if str(x[0]).isdigit() else x[0]):
        # y 用于分层K折
        y_all = np.array([y for _, y in samples])
        skf = StratifiedKFold(n_splits=args.cv_splits, shuffle=True, random_state=args.seed)
        folds = list(skf.split(np.arange(len(samples)), y_all))

        fold_metrics = {m: [] for m in metrics}  # 收集 test 上的各指标（按“验证最优轮”策略选的 test 结果）
        print(f"\n===== Subject {sub_id}: {len(samples)} samples, {args.cv_splits}-fold CV =====")

        for k in range(args.cv_splits):
            test_index = folds[k][1]
            val_index  = folds[(k + 1) % args.cv_splits][1]
            train_index = np.setdiff1d(np.arange(len(samples)), np.union1d(test_index, val_index))

            train_samples = [samples[i] for i in train_index.tolist()]
            val_samples   = [samples[i] for i in val_index.tolist()]
            test_samples  = [samples[i] for i in test_index.tolist()]
            
            # === 每折独立输出目录 + 跳过已完成 ===
            fold_out = None
            if args.output_dir:
                fold_out = os.path.join(args.output_dir, f"S{sub_id}", f"F{k}")
                os.makedirs(fold_out, exist_ok=True)

            # 完成标记：优先看 done.json；其次看 best checkpoint
            done_path = os.path.join(fold_out, "done.json") if fold_out else None
            best_tag  = f"checkpoint_best_S{sub_id}_F{k}.pth"
            best_path = os.path.join(fold_out, best_tag) if fold_out else None

            if fold_out and (os.path.exists(done_path) or os.path.exists(best_path)):
                print(f"[S{sub_id} F{k}] Found completion marker, skip this fold.")
                continue

            # 临时切换 output_dir 到该折目录（保证 save/auto_resume/日志都落在独立目录）
            orig_out = args.output_dir
            if fold_out:
                args.output_dir = fold_out

            # 每折一次完整训练
            val_stats, test_stats = _run_one_fold(args, ds_init, ch_names_upper, metrics, k, sub_id,
                                                train_samples, val_samples, test_samples)

            # 还原 output_dir（避免影响后续总表输出等）
            args.output_dir = orig_out


            # 取该折在“验证最优轮”对应的 test 指标
            if test_stats is not None:
                for m in metrics:
                    if m in test_stats:
                        fold_metrics[m].append(float(test_stats[m]))

        # 被试内折均值与std
        row = {"subject": sub_id}
        for m in metrics:
            vals = fold_metrics[m]
            row[m] = float(np.mean(vals)) if len(vals) else float('nan')
            row[m + "_std"] = float(np.std(vals)) if len(vals) else float('nan')
        subject_rows.append(row)
        # [NEW] 立刻增量更新一次（仅主进程）
        if args.output_dir and _is_main_process_safe():
            out_csv = os.path.join(args.output_dir, "cv_summary.csv")
            _incremental_update_cv(out_csv, row, metrics)
            print(f"[S{sub_id}] cv_summary.csv updated -> {out_csv}")

    # 输出/合并 CSV 汇总（避免把已有有效结果用 NaN 覆盖）
    if args.output_dir and _is_main_process_safe():
        out_csv = os.path.join(args.output_dir, "cv_summary.csv")
        fieldnames = ["subject"] + [m for m in metrics] + [m + "_std" for m in metrics]

        # 读取已有（去掉 OVERALL）
        existing = {}
        if os.path.exists(out_csv):
            with open(out_csv, "r", encoding="utf-8", newline="") as rf:
                rr = csv.DictReader(rf)
                if rr.fieldnames:
                    for x in rr:
                        subj = x.get("subject", "")
                        if subj and subj != "OVERALL":
                            existing[subj] = x

        # 合并逻辑：本轮有有效值覆盖；否则保留旧值
        def _to_float(v):
            try:
                fv = float(v)
                return fv
            except Exception:
                return float("nan")

        merged = {}
        # 先把本轮的放进去
        for r in subject_rows:
            subj = str(r["subject"])
            row = {"subject": subj}
            for m in metrics:
                v = r.get(m, float("nan"))
                s = r.get(m + "_std", float("nan"))
                row[m] = float(v) if not (isinstance(v, float) and np.isnan(v)) else float("nan")
                row[m + "_std"] = float(s) if not (isinstance(s, float) and np.isnan(s)) else float("nan")
            merged[subj] = row

        # 再把旧的补齐/保留（仅当本轮对应项是 NaN 才用旧值）
        for subj, x in existing.items():
            if subj not in merged:
                merged[subj] = {"subject": subj}
                for m in metrics:
                    merged[subj][m] = _to_float(x.get(m, "nan"))
                    merged[subj][m + "_std"] = _to_float(x.get(m + "_std", "nan"))
            else:
                for m in metrics:
                    if np.isnan(merged[subj][m]):
                        merged[subj][m] = _to_float(x.get(m, "nan"))
                    if np.isnan(merged[subj][m + "_std"]):
                        merged[subj][m + "_std"] = _to_float(x.get(m + "_std", "nan"))

        # 排序（数字在前，OVERALL 最后写）
        def _subj_key(s):
            digits = "".join(ch for ch in s if ch.isdigit())
            try:
                return (int(digits),)
            except Exception:
                return (s,)

        rows = [merged[k] for k in sorted(merged.keys(), key=_subj_key)]

        # 重新计算 OVERALL
        overall = {"subject": "OVERALL"}
        for m in metrics:
            vals = [float(r[m]) for r in rows if isinstance(r.get(m), (int, float)) and not np.isnan(float(r[m]))]
            overall[m] = float(np.mean(vals)) if len(vals) else float("nan")
            overall[m + "_std"] = float(np.std(vals)) if len(vals) else float("nan")

        # 写回
        os.makedirs(args.output_dir, exist_ok=True)
        with open(out_csv, "w", newline="", encoding="utf-8") as wf:
            w = csv.DictWriter(wf, fieldnames=fieldnames)
            w.writeheader()
            for r in rows:
                # 将 NaN 以空串写出更友好（也可保留为 'nan' 字符串）
                rr = {}
                for k in fieldnames:
                    v = r.get(k, "")
                    if isinstance(v, float) and np.isnan(v):
                        rr[k] = ""
                    else:
                        rr[k] = v
                w.writerow(rr)
            # OVERALL 行
            oo = {}
            for k in fieldnames:
                v = overall.get(k, "")
                if isinstance(v, float) and np.isnan(v):
                    oo[k] = ""
                else:
                    oo[k] = v
            w.writerow(oo)
        print(f"\nMerged & saved CV summary to: {out_csv}")


if __name__ == '__main__':
    opts, ds_init = get_args()
    if opts.output_dir:
        Path(opts.output_dir).mkdir(parents=True, exist_ok=True)
    main(opts, ds_init)
