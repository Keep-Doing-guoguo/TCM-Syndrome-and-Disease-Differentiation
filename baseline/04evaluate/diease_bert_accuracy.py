#!/usr/bin/env python
# coding=utf-8

"""
@author: zgw
@date: 2025/5/24 21:33
@source from: 
"""
import json

from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
path = "/gemini/code/checkpoints/checkpoint-2000"
path = '/Volumes/PSSD/NetThink/pythonProject/7-19-Project/TCM-Syndrome-and-Disease-Differentiation/baseline/task1/val_disease_instruction_data1.json'

model = AutoModelForSequenceClassification.from_pretrained(path).to('cuda')
tokenizer = AutoTokenizer.from_pretrained(path)
# 模拟单标签模型预测函数（替换为你自己的）
def model(prompt: str) -> str:
    # '胸痹心痛病', '心衰病', '眩晕病', '心悸病'
    id2_label = {0: "胸痹心痛病", 1: "心衰病", 0: "眩晕病", 1: "心悸病"}
    model.eval()
    with torch.inference_mode():
        inputs = tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.cuda() for k, v in inputs.items()}
        logits = model(**inputs).logits
        pred = torch.argmax(logits, dim=-1)
        print(f"输入：模型预测结果:{id2_label.get(pred.item())}")
    return id2_label.get(pred.item())

# 加载数据
with open(path, 'r', encoding='utf-8') as f:
    data = json.load(f)

correct = 0
total = 0

for item in data:
    prompt = f"{item['instruction'].strip()}\n{item['input'].strip()}"
    pred = model(prompt).strip()
    true = item.get("output", "").strip()

    if pred == true:
        correct += 1
    total += 1

accuracy = correct / total if total > 0 else 0
print(f"✅ 单标签准确率: {accuracy:.2%} （{correct}/{total}）")



