#!/usr/bin/env python
# coding=utf-8

"""
@author: zgw
@date: 2025/5/15 17:03
@source from: 
"""
from tqdm import tqdm
import json
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, Dataset
import torch
from torch import nn

# 路径替换成你自己的
train_path = '/Volumes/PSSD/NetThink/pythonProject/7-19-Project/TCM-Syndrome-and-Disease-Differentiation/data/train.json'
test_path = '/Volumes/PSSD/NetThink/pythonProject/7-19-Project/TCM-Syndrome-and-Disease-Differentiation/data/test.json'
val_path = '/Volumes/PSSD/NetThink/pythonProject/7-19-Project/TCM-Syndrome-and-Disease-Differentiation/data/test.json'
model_path = '/Volumes/mac_win/models/tiansz/bert-base-chinese'

class MyDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=512):
        self.encodings = tokenizer(texts, truncation=True, padding='max_length',#token_type_ids代表的是
                                   max_length=max_length, return_tensors="pt")#只要是经过bert的tokenizer都会变成三个向量、input_ids、token_type_ids、attention_mask
        self.labels = torch.tensor(labels)

    def __getitem__(self, idx):
        # for key,val in self.encodings.items():
        #     print(key)
        #     print(val[idx])
        #     print('debug')
        item = {key: val[idx] for key, val in self.encodings.items()}#它是遍历上述字典的一种方式。将其变成字典。
        item["labels"] = self.labels[idx]
        '''
        最终item变成为：
        item = {
            "input_ids":[2,512],
            "token_type_ids":[2,512],
            "attention_mask":[2,512],
            "labels":[2,4]
        }
        '''
        return item

    def __len__(self):
        return len(self.labels)


def TCM_SD_Data_Loader(tokenizer, batch_size=2):
    syndromes = ['胸痹心痛病', '心衰病', '眩晕病', '心悸病']
    syndrome2id_dict = {v: i for i, v in enumerate(syndromes)}
    id2syndrome_dict = {i: v for i, v in enumerate(syndromes)}

    def load_data(path):
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        texts, labels = [], []
        for content in tqdm(data, desc=f'Loading {path}'):
            text = content['症状'] + content['中医望闻切诊']
            label = syndrome2id_dict[content['疾病']]  # 🟢 changed to integer
            texts.append(text)
            labels.append(label)
        return texts, labels

    train_texts, train_labels = load_data(train_path)
    test_texts, test_labels = load_data(test_path)
    val_texts, val_labels = load_data(val_path)

    train_dataset = MyDataset(train_texts, train_labels, tokenizer)
    test_dataset = MyDataset(test_texts, test_labels, tokenizer)
    val_dataset = MyDataset(val_texts, val_labels, tokenizer)


    return train_dataset, test_dataset, val_dataset, id2syndrome_dict

import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from transformers import EvalPrediction

def compute_metrics(eval_pred: EvalPrediction):
    logits, labels = eval_pred.predictions, eval_pred.label_ids
    # 将 one-hot 标签转为整数类别，如 [0, 1, 0, 0] -> 1
    #labels = np.argmax(labels, axis=1)
    preds = np.argmax(logits, axis=1)

    return {
        "accuracy": accuracy_score(labels, preds),
        "f1_macro": f1_score(labels, preds, average="macro"),
        "f1_weighted": f1_score(labels, preds, average="weighted")
    }
from transformers import AutoModelForSequenceClassification,AutoTokenizer
from transformers import Trainer, TrainingArguments
from transformers import BertTokenizer, BertForSequenceClassification

tokenizer = BertTokenizer.from_pretrained(model_path)
model = BertForSequenceClassification.from_pretrained(model_path, num_labels=4)
train_dataset, test_dataset, val_dataset, id2syndrome_dict = TCM_SD_Data_Loader(tokenizer)
print('debug')
# 遍历 dataset 中每一条数据
# for idx in range(len(train_dataset)):
#         sample = train_dataset[idx]
#         print(f"Sample {idx}:")
#         print("  input_ids:", sample["input_ids"])
#         print("  token_type_ids:", sample["token_type_ids"])
#         print("  attention_mask:", sample["attention_mask"])
#         print("  labels:", sample["labels"])
#         print("-" * 30)
training_args = TrainingArguments(
    output_dir="./checkpoints",
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    num_train_epochs=5,
    logging_dir='./logs',
    load_best_model_at_end=True,
    metric_for_best_model='accuracy',
    no_cuda=True,  # ✅ 强制只使用 CPU
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics  # 返回 accuracy 等指标
)
trainer.train()



'''
output_dir="./checkpoints"
模型保存的输出路径，会保存每个 epoch 生成的权重（包括最优模型）。


per_device_train_batch_size=8
每个设备（GPU/CPU）上的训练 batch size。Mac 没有 GPU，这里就是每次训练用 8 个样本。


per_device_eval_batch_size=8
每个设备上验证时的 batch size，同样是 8 个样本。


evaluation_strategy="epoch"
评估策略，每个 epoch 结束后自动在验证集上评估一次。


save_strategy="epoch"
模型保存策略，每个 epoch 保存一次 checkpoint（包含 model/optimizer/scheduler 状态）。


num_train_epochs=5
模型训练的总轮数（epoch）。


logging_dir='./logs'
训练过程的日志目录（比如 TensorBoard 可视化日志）。


load_best_model_at_end=True
是否在训练结束后加载验证集上表现最好的模型。需要 metric_for_best_model 配合。


metric_for_best_model='accuracy'
判断“最优模型”所使用的评估指标，这里是 'accuracy'（你需要自己定义 compute_metrics 返回这个字段）。


no_cuda=True
✅ 禁用 CUDA，只用 CPU 训练，适用于 Mac 或无 GPU 情况。


learning_rate
float
学习率，默认 5e-5，常用于微调。


weight_decay
float
权重衰减 (L2正则)，通常设为 0.01。


adam_beta1 / adam_beta2
float
Adam 优化器的 β 参数，默认 (0.9, 0.999)。


adam_epsilon
float
防止除零的小数，默认 1e-8。


max_grad_norm
float
梯度裁剪阈值，避免梯度爆炸，默认 1.0。


lr_scheduler_type
str
学习率调度器类型：linear、cosine、constant 等。


warmup_steps
int
训练前多少步进行学习率 warmup（线性升高），适合大模型或低学习率情况。


gradient_accumulation_steps
int
梯度累计步数，用于 batch size 太小时“模拟”更大的 batch。




'''

