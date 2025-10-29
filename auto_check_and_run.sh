#!/usr/bin/env bash
# -*- coding: utf-8 -*-
# @Time     : (UTC+8)
# @Author   : Chen, Y.R.
# @File     : auto_check_and_run.sh
# @Notes    : 极简版：仅检测是否运行；未运行则启动（后台 nohup 记录日志）；保留 cron_heartbeat.log

set -euo pipefail
export TZ="Asia/Shanghai"

# ===== 可配置区（按需手改） =====
BASE_DIR="/home/yeqi3/cyr/code/LaBraM"          # 代码目录
ENV_NAME="cyr"                                  # conda 环境名
PYTHON_BIN="${PYTHON_BIN:-python}"              # Python 可执行
TARGET_FILE="TrainCVLauncher.py"                # 要运行的 Python 文件
TARGET_PATTERN="${TARGET_PATTERN:-${TARGET_FILE}}"  # 进程匹配关键字
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-5,6,7}"  # 限制可见GPU

# 日志：程序运行日志 & 心跳日志
LOG_FILE="${BASE_DIR}/log/$(date +'%Y%m%d_%H%M').log"
HEARTBEAT_LOG="${BASE_DIR}/cron_heartbeat.log"
# =================================

now() { date "+%Y-%m-%d %H:%M:%S"; }

mkdir -p "${BASE_DIR}" >/dev/null 2>&1 || true
mkdir -p "$(dirname "${LOG_FILE}")" >/dev/null 2>&1 || true
touch "${HEARTBEAT_LOG}" >/dev/null 2>&1 || true

cd "${BASE_DIR}"

# 1) 并发检测
if pgrep -f "${TARGET_PATTERN}" >/dev/null 2>&1; then
  echo "[`now`] 检测到正在运行（pattern='${TARGET_PATTERN}'），无需启动。" >> "${HEARTBEAT_LOG}"
  exit 0
fi

# 2) 激活 conda（失败会写心跳并退出）
if command -v conda >/dev/null 2>&1; then
  eval "$(conda shell.bash hook)"
elif [[ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]]; then
  # shellcheck disable=SC1091
  source "$HOME/anaconda3/etc/profile.d/conda.sh"
else
  echo "[`now`] [ERR] 未找到 conda，可执行 '${TARGET_FILE}' 取消。BASE_DIR='${BASE_DIR}'" >> "${HEARTBEAT_LOG}"
  exit 1
fi

if ! conda activate "${ENV_NAME}" >/dev/null 2>&1; then
  echo "[`now`] [ERR] conda env '${ENV_NAME}' 激活失败，取消启动。" >> "${HEARTBEAT_LOG}"
  exit 1
fi

# 3) 启动（后台 + nohup，输出到 LOG_FILE）
echo "[`now`] 启动 '${TARGET_FILE}' | CUDA_VISIBLE_DEVICES='${CUDA_VISIBLE_DEVICES}' | LOG='${LOG_FILE}'" >> "${HEARTBEAT_LOG}"

set +e
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" \
  nohup "${PYTHON_BIN}" -u "${BASE_DIR}/${TARGET_FILE}" >> "${LOG_FILE}" 2>&1 &
pid=$!
rc=$?
set -e

if [[ $rc -ne 0 || -z "${pid}" ]]; then
  echo "[`now`] [ERR] 启动失败（rc=${rc} pid='${pid:-}）。" >> "${HEARTBEAT_LOG}"
  exit 1
fi

echo "[`now`] 已后台运行 pid=${pid}，详见 ${LOG_FILE}" >> "${HEARTBEAT_LOG}"
exit 0
