import json
import pandas as pd

# 加载原始 JSON 文件为 DataFrame
#/Volumes/PSSD/NetThink/pythonProject/7-19-Project/TCM-Syndrome-and-Disease-Differentiation/data/test.json
path = '/Volumes/PSSD/NetThink/pythonProject/7-19-Project/TCM-Syndrome-and-Disease-Differentiation/data/test.json'
patients_df = pd.read_json(path)

# 修改后的 instruction 内容
instruction = """任务：根据患者[基本信息]、[主诉]、[症状]、[中医望闻切诊]、[病史]、[体格检查]、[辅助检查]等信息，判断该患者最符合的中医证型（可多选）。

# 要求1：从以下[证型候选]中选择；
# 要求2：可以输出1-3个证型；
# 要求3：多个证型之间用“|”分隔；
# 要求4：只输出证型名称，不需要解释。"""

# 构造新格式数据集
output_items = []
for _, p in patients_df.iterrows():
    base_info = f"性别: {p['性别']}，年龄: {p['年龄']}，职业: {p['职业']}，婚姻: {p['婚姻']}，发病节气: {p['发病节气']}，病史陈述者: {p['病史陈述者']}"

    input_text = f"""[证型候选]: 气虚血瘀证、痰瘀互结证、气阴两虚证、气滞血瘀证、肝阳上亢证、阴虚阳亢证、痰热蕴结证、痰湿痹阻证、阳虚水停证、肝肾阴虚证
[基本信息]: {base_info}
[主诉]: {p.get("主诉", "")}
[症状]: {p.get("症状", "")}
[中医望闻切诊]: {p.get("中医望闻切诊", "")}
[病史]: {p.get("病史", "")}
[体格检查]: {p.get("体格检查", "")}
[辅助检查]: {p.get("辅助检查", "")}"""

    output_items.append({
        "instruction": instruction,
        "input": input_text.strip(),
        "output": p.get("证型", "")
    })

# 保存为 JSON 数组格式
output_path = "test_syndrome_instruction_data.json"
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(output_items, f, ensure_ascii=False, indent=2)

print(f"✅ 数据已保存至：{output_path}")