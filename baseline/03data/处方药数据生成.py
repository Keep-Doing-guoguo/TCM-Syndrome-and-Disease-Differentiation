#!/usr/bin/env python
# coding=utf-8

"""
@author: zgw
@date: 2025/7/28 14:37
@source from: 
"""
import json
from tqdm import tqdm

# 输入输出路径
data_path = '/Volumes/PSSD/NetThink/pythonProject/7-19-Project/TCM-Syndrome-and-Disease-Differentiation/data/test.json'
output_path = '/Volumes/PSSD/NetThink/pythonProject/7-19-Project/TCM-Syndrome-and-Disease-Differentiation/baseline/03data/chufang_test.json'

# 加载数据
data = json.load(open(data_path, 'r', encoding='utf-8'))
print(f'总病历数：{len(data)}')

# 指令模板
instruction = '''
任务：根据患者[基本信息],[主诉],[症状],[中医望闻切诊],[病史],[体格检查],[辅助检查]等信息,在[草药]中为患者推荐需要使用的[推荐草药]。
# 要求1: 草药局限在下方列表中。
[草药]: '冬瓜皮', '沉香', ... , '钩藤', '天冬'
# 要求2: 有多个中草药，每个中草药之间用逗号分隔。
# 要求3: 没有的中草药不要多写。
# 要求4: 输出中仅需要输出草药名称，不需要给出任何解释和其他信息，数量控制在10-15个左右。
'''.strip()

# 构造新数据
output_data = []
for idx, item in tqdm(enumerate(data), total=len(data)):
    input_text = f'''
[基本信息]:患者性别为{item["性别"]},职业为{item["职业"]},年龄为{item["年龄"]},婚姻为{item["婚姻"]},发病节气在{item["发病节气"]}。
[主诉]:{item["主诉"]}
[症状]:{item["症状"]}
[中医望闻切诊]:{item["中医望闻切诊"]}
[病史]:{item["病史"]}
[体格检查]:{item["体格检查"]}
[辅助检查]:{item["辅助检查"]}
[推荐草药]:
'''.strip()

    output_text = item['处方'][1:-1]  # 去掉最外层括号

    output_data.append({
        'instruction': instruction,
        'input': input_text,
        'output': output_text
    })

# 保存为 JSON list 格式
with open(output_path, 'w', encoding='utf-8') as writer:
    json.dump(output_data, writer, ensure_ascii=False, indent=2)

print(f"✅ 已保存至: {output_path}")

'''
test的数据条数为80条；
train的数据条数为800条；
如果需要
'''
