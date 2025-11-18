#!/usr/bin/env python
# coding=utf-8

"""
@author: zgw
@date: 2025/5/25 13:54
@source from: 
"""
import json

# 模拟多标签模型预测函数（替换为你自己的）
def model(prompt: str) -> str:
    return "气虚血瘀证|气阴两虚证"  # 示例：模型输出多个标签

# 多标签分割为集合
def normalize_labels(output_str):
    return set(label.strip() for label in output_str.split("|") if label.strip())

# 加载数据
path = '/Volumes/PSSD/NetThink/pythonProject/7-19-Project/TCM-Syndrome-and-Disease-Differentiation/baseline/task1/test_syndrome_instruction_data.json'
with open(path, 'r', encoding='utf-8') as f:
    data = json.load(f)

exact_match = 0
partial_match = 0
total = 0

for item in data:
    prompt = f"{item['instruction'].strip()}\n{item['input'].strip()}"
    pred = normalize_labels(model(prompt).strip())
    true = normalize_labels(item.get("output", "").strip())

    if pred == true:
        exact_match += 1
    if pred & true:
        partial_match += 1
    total += 1

print(f"✅ 多标签完全匹配准确率: {exact_match / total:.2%} （{exact_match}/{total}）")
print(f"🟡 多标签部分命中准确率: {partial_match / total:.2%} （{partial_match}/{total}）")