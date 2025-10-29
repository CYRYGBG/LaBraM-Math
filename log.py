# -*- coding: utf-8 -*-
# @Time     : 2025/10/29 15:45
# @Author   : Chen, Y.R.
# @File     : Log2CSV.py
# @Software : PyCharm
# @Notes    : 将每行 JSON 格式的日志文件（log.txt）转换为 CSV 表格（log.csv）

import json
import csv

# ======== 配置区 ========
input_file = "/home/yeqi3/cyr/code/LaBraM/math/cv/log.txt"   # 输入日志文件路径
output_file = "/home/yeqi3/cyr/code/LaBraM/math/cv/log.csv"  # 输出CSV文件路径
# ========================

with open(input_file, "r", encoding="utf-8") as f_in:
    lines = [json.loads(line.strip()) for line in f_in if line.strip()]

# 提取所有字段名（保证顺序一致）
fieldnames = list(lines[0].keys())

with open(output_file, "w", newline="", encoding="utf-8") as f_out:
    writer = csv.DictWriter(f_out, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(lines)

print(f"✅ 已将 {input_file} 转换为 {output_file}")
