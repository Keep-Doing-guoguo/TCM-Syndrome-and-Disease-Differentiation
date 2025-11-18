#!/usr/bin/env python
# coding=utf-8

"""
@author: zgw
@date: 2025/5/15 10:03
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
    syndromes = ['胸痹心痛病', '心衰病', '眩晕病', '心悸病']
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
model = BertMultiLabel(num_labels=4).to(device)
optimizer = AdamW(model.parameters(), lr=2e-5)
loss_fn = nn.CrossEntropyLoss()
train_dataloader, test_dataloader, val_dataloader, id2syndrome_dict = TCM_SD_Data_Loader(tokenizer)
# 单 epoch 训练
model.train()
for batch in tqdm(train_dataloader):
    inputs = {k: v.to(device) for k, v in batch.items() if k != 'labels'}
    disease_label = torch.argmax(batch['labels'], dim=1).long()  # ✅ 转为 [1, 2]

    optimizer.zero_grad()
    logits = model(**inputs)
    loss = loss_fn(logits, disease_label)
    loss.backward()
    optimizer.step()
print('debug')

