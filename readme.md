# 中医辨证辨病及中药处方生成

> 🏆 参赛项目：天池大赛 - CCL 2025 中文计算语言学评测（CCL25-Eval）  
> 🔗 赛题链接：[https://tianchi.aliyun.com/competition/entrance/532301/information](https://tianchi.aliyun.com/competition/entrance/532301/information)

## 📌 任务简介

本项目参与 **CCL25-Eval 任务9：中医辨证辨病及中药处方生成评测**，旨在构建一个能够根据患者主诉、症状等临床信息，自动完成以下三项子任务的智能系统：

1. **中医辨证**（Syndrome Differentiation）  
   识别患者的中医证型（如“肝郁脾虚证”、“风寒感冒证”等）。

2. **中医辨病**（Disease Diagnosis）  
   判断患者所患的中医病名（如“感冒”、“胃脘痛”、“眩晕”等）。

3. **中药处方生成**（Herbal Prescription Generation）  
   生成符合中医理论、针对证型与病名的合理中药方剂（如“银翘散”、“逍遥散”等，含具体药物组成）。

该任务融合了**中医知识理解、自然语言处理与生成式 AI**，对模型的领域知识、逻辑推理与生成能力提出较高要求。

## 🎯 项目目标

- 构建端到端或分阶段的中医诊疗推理模型；
- 在官方测试集上取得高准确率（辨证、辨病）与高合理性（处方生成）；
- 探索大语言模型（LLM）在中医专业领域的微调与提示工程策略；
- 遵循中医诊疗规范，确保输出内容的**专业性、安全性与可解释性**。

## 📂 项目结构

```bash
.
├── data/                   # 数据集（需从天池下载后放入）
│   ├── train.json          # 训练集
│   ├── dev.json            # 验证集
│   └── test.json           # 测试集（无标签）
├── src/
│   ├── models/             # 模型定义与加载
│   ├── utils/              # 工具函数（数据预处理、评估等）
│   ├── prompts/            # 提示模板（Prompt Engineering）
│   └── train.py            # 训练脚本
│   └── predict.py          # 推理脚本（生成提交结果）
├── output/                 # 模型输出与预测结果
├── checkpoints/            # 保存的模型权重
├── requirements.txt        # 依赖库
├── README.md               # 本文件
└── submit/                 # 提交文件目录（符合天池格式）

```

数据分为训练集、验证集和测试集，数据量分别为800、200和500。

test.json：80条数据

train.jsonl：640条

train_chufang.json：800条数据

训练数据为720条，测试数据为80。

diease-bert：训练epoch为20，准确率为75、f1为73.显存使用为为23个g，batch=32。4分钟即可。
syndrome-bert：训练为epoch大概是再50，20是已经降低到最低，准确率为30%、f1为20多。训练时长为：4分钟即可。batch=16，使用12g即可。


大模型微调：
使用的yaml文件为：qwen_7b_lora_sft_diease文件。数据文件在data下，总又800条。9:1划分。文件保存为/models/lora/sft。cut_len=4096
per_device_train_batch_size: 1
gradient_accumulation_steps: 2
使用的显存为40G。时间为85分钟。








