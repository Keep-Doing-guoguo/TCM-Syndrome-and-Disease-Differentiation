import json
import pandas as pd

# 加载原始数据
path = '/Volumes/PSSD/NetThink/pythonProject/7-19-Project/TCM-Syndrome-and-Disease-Differentiation/data/train.json'
path = '/Volumes/PSSD/NetThink/pythonProject/7-19-Project/TCM-Syndrome-and-Disease-Differentiation/data/test.json'

patients_df = pd.read_json(path)

# 固定提示说明（instruction）
instruction = "你是一个中医疾病分类的专家，请根据患者的相关信息，判断其最符合的疾病类型。"

# 处理为 instruction / input / output 格式
converted_items = []
for _, p in patients_df.iterrows():
    base_info = f"性别: {p['性别']}，年龄: {p['年龄']}，职业: {p['职业']}，婚姻: {p['婚姻']}，发病节气: {p['发病节气']}，病史陈述者: {p['病史陈述者']}"
    input_text = (
        f"[疾病候选]: 胸痹心痛病, 心衰病, 眩晕病, 心悸病\n"
        f"[基本信息]: {base_info}\n"
        f"[主诉]: {p.get('主诉', '')}\n"
        f"[症状]: {p.get('症状', '')}\n"
        f"[中医望闻切诊]: {p.get('中医望闻切诊', '')}\n"
        f"[病史]: {p.get('病史', '')}\n"
        f"[体格检查]: {p.get('体格检查', '')}\n"
        f"[辅助检查]: {p.get('辅助检查', '')}"
    )
    converted_items.append({
        "instruction": instruction,
        "input": input_text.strip(),
        "output": p.get("疾病", "")
    })

# 保存为 JSONL 文件
output_path = "test_disease_instruction_data.json"
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(converted_items, f, ensure_ascii=False, indent=2)

print(f"已保存为 JSON 文件（list）：{output_path}")