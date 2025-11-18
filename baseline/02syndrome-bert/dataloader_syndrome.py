#!/usr/bin/env python
# coding=utf-8

"""
@author: zgw
@date: 2025/5/15 17:47
@source from: 
"""
from tqdm import tqdm
import json
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, Dataset
from transformers import BertTokenizer, BertModel, AdamW
import torch
from torch import nn

# 路径替换成你自己的
train_path = '/Volumes/PSSD/NetThink/pythonProject/7-19-Project/TCM-Syndrome-and-Disease-Differentiation/TCM-TBOSD/TCM-TBOSD-train.json'
test_path = '/Volumes/PSSD/NetThink/pythonProject/7-19-Project/TCM-Syndrome-and-Disease-Differentiation/TCM-TBOSD/TCM-TBOSD-test.json'
val_path = '/Volumes/PSSD/NetThink/pythonProject/7-19-Project/TCM-Syndrome-and-Disease-Differentiation/TCM-TBOSD/TCM-TBOSD-val.json'
tokenizer = BertTokenizer.from_pretrained('/Volumes/mac_win/models/tiansz/bert-base-chinese')

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
    syndromes = ['气虚血瘀证', '痰瘀互结证', '气阴两虚证', '气滞血瘀证', '肝阳上亢证', '阴虚阳亢证', '痰热蕴结证', '痰湿痹阻证', '阳虚水停证', '肝肾阴虚证']
    syndrome2id_dict = {v: i for i, v in enumerate(syndromes)}
    id2syndrome_dict = {i: v for i, v in enumerate(syndromes)}

    def load_data(path):
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        texts = []
        labels = []

        for content in tqdm(data, desc=f'Loading {path}'):
            text = content['症状'] + content['中医望闻切诊']
            label = [0] * len(syndromes)
            label[syndrome2id_dict[content['疾病']]] = 1
            texts.append(text)
            labels.append(label)

        return texts, labels

    train_texts, train_labels = load_data(train_path)
    test_texts, test_labels = load_data(test_path)
    val_texts, val_labels = load_data(val_path)

    train_dataset = MyDataset(train_texts, train_labels, tokenizer)
    test_dataset = MyDataset(test_texts, test_labels, tokenizer)
    val_dataset = MyDataset(val_texts, val_labels, tokenizer)
    # 遍历 dataset 中每一条数据
    # for idx in range(len(train_dataset)):
    #     sample = train_dataset[idx]
    #     print(f"Sample {idx}:")
    #     print("  input_ids:", sample["input_ids"])
    #     print("  token_type_ids:", sample["token_type_ids"])
    #     print("  attention_mask:", sample["attention_mask"])
    #     print("  labels:", sample["labels"])
    #     print("-" * 30)
    train_dataloader = DataLoader(train_dataset, sampler=RandomSampler(train_dataset),
                                  batch_size=batch_size, drop_last=True)
    test_dataloader = DataLoader(test_dataset, sampler=SequentialSampler(test_dataset),
                                 batch_size=batch_size, drop_last=True)
    val_dataloader = DataLoader(val_dataset, sampler=SequentialSampler(val_dataset),
                                batch_size=batch_size, drop_last=True)

    return train_dataloader, test_dataloader, val_dataloader, id2syndrome_dict

from sklearn.metrics import f1_score, accuracy_score

def compute_metrics_multi_label(pred):
    logits, labels = pred
    probs = torch.sigmoid(torch.tensor(logits))
    preds = (probs > 0.5).int().numpy()
    labels = labels.astype(int)
    return {
        "f1_micro": f1_score(labels, preds, average='micro'),
        "f1_macro": f1_score(labels, preds, average='macro'),
        "accuracy": accuracy_score(labels, preds)
    }

class BertMultiLabel(nn.Module):
    def __init__(self, num_labels):
        super().__init__()
        self.bert = BertModel.from_pretrained("/Volumes/mac_win/models/tiansz/bert-base-chinese")
        self.dropout = nn.Dropout(0.3)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_labels)

    def forward(self, input_ids, attention_mask, token_type_ids):
        out = self.bert(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        pooled = self.dropout(out.pooler_output)
        return self.classifier(pooled)
# 训练配置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = BertMultiLabel(num_labels=10).to(device)
optimizer = AdamW(model.parameters(), lr=2e-5)
loss_fn = nn.BCEWithLogitsLoss()
train_dataloader, test_dataloader, val_dataloader, id2syndrome_dict = TCM_SD_Data_Loader(tokenizer)
# 单 epoch 训练
model.train()
for batch in tqdm(train_dataloader):
    inputs = {k: v.to(device) for k, v in batch.items() if k != 'labels'}
    # 正确写法（多标签任务需要 float 格式）
    disease_label = batch['labels'].float().to(device)

    optimizer.zero_grad()
    logits = model(**inputs)
    loss = loss_fn(logits, disease_label)
    loss.backward()
    optimizer.step()
print('debug')

def train():
    from transformers import get_scheduler
    from sklearn.metrics import f1_score, accuracy_score
    import numpy as np

    model = BertMultiLabel(num_labels=10).to(device)
    optimizer = AdamW(model.parameters(), lr=2e-5)

    # 学习率调度器（线性）
    num_training_steps = len(train_dataloader) * 5  # 假设 5 个 epoch
    lr_scheduler = get_scheduler(
        name="linear", optimizer=optimizer,
        num_warmup_steps=0, num_training_steps=num_training_steps
    )

    loss_fn = nn.BCEWithLogitsLoss()
    best_f1 = 0

    for epoch in range(5):  # 多个 epoch
        model.train()
        total_loss = 0

        for batch in tqdm(train_dataloader, desc=f"Epoch {epoch + 1}"):
            inputs = {k: v.to(device) for k, v in batch.items() if k != 'labels'}
            labels = batch['labels'].float().to(device)

            optimizer.zero_grad()
            logits = model(**inputs)
            loss = loss_fn(logits, labels)
            loss.backward()
            optimizer.step()
            lr_scheduler.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_dataloader)
        print(f"Epoch {epoch + 1} - Avg Train Loss: {avg_loss:.4f}")

        # ✅ 验证集评估
        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for batch in val_dataloader:
                inputs = {k: v.to(device) for k, v in batch.items() if k != 'labels'}
                labels = batch['labels'].cpu().numpy()
                logits = model(**inputs)
                probs = torch.sigmoid(logits).cpu().numpy()
                preds = (probs > 0.5).astype(int)

                all_preds.append(preds)
                all_labels.append(labels)

        all_preds = np.vstack(all_preds)
        all_labels = np.vstack(all_labels)
        micro_f1 = f1_score(all_labels, all_preds, average='micro')
        macro_f1 = f1_score(all_labels, all_preds, average='macro')
        acc = accuracy_score(all_labels, all_preds)

        print(f"Validation - Micro-F1: {micro_f1:.4f} | Macro-F1: {macro_f1:.4f} | Acc: {acc:.4f}")

        # ✅ 保存最佳模型
        if micro_f1 > best_f1:
            best_f1 = micro_f1
            torch.save(model.state_dict(), "best_model.pth")
            print("✅ Best model saved.")


'''
	•	如果你是单标签多分类任务（每条数据只属于一个类别），用 CrossEntropyLoss。
	•	如果你是多标签多分类任务（每条数据可属于多个标签），用 BCEWithLogitsLoss。
	•	你的标签是形如 [0,0,1,0] 的 one-hot 多标签，这说明你做的是 多标签分类任务，不要使用 argmax()！
'''