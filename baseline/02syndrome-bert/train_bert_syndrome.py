from tqdm import tqdm
import json
from torch.utils.data import Dataset
from transformers import BertTokenizer, BertModel, BertPreTrainedModel, TrainingArguments, Trainer
from transformers import EvalPrediction
import torch
from torch import nn
import numpy as np
from sklearn.metrics import f1_score, accuracy_score

# 路径替换成你自己的
train_path = '/Volumes/PSSD/NetThink/pythonProject/7-19-Project/TCM-Syndrome-and-Disease-Differentiation/data/train.json'
test_path = '/Volumes/PSSD/NetThink/pythonProject/7-19-Project/TCM-Syndrome-and-Disease-Differentiation/data/test.json'
val_path = '/Volumes/PSSD/NetThink/pythonProject/7-19-Project/TCM-Syndrome-and-Disease-Differentiation/data/test.json'
model_path = '/Volumes/mac_win/models/tiansz/bert-base-chinese'

class MyDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=512):
        self.encodings = tokenizer(texts, truncation=True, padding='max_length', max_length=max_length, return_tensors="pt")
        self.labels = torch.tensor(labels).float()

    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        item["labels"] = self.labels[idx]
        return item

    def __len__(self):
        return len(self.labels)

def TCM_SD_Data_Loader(tokenizer):
    #证型是一个多分类多标签的问题，
    syndromes = ['气虚血瘀证', '痰瘀互结证', '气阴两虚证', '气滞血瘀证', '肝阳上亢证',
                 '阴虚阳亢证', '痰热蕴结证', '痰湿痹阻证', '阳虚水停证', '肝肾阴虚证']  #单使用bert进行预测证型时标签

    syndrome2id = {v: i for i, v in enumerate(syndromes)}

    # def load_data(path):
    #     with open(path, 'r', encoding='utf-8') as f:
    #         data = json.load(f)
    #     texts, labels = [], []
    #     for item in data:
    #         text = item['症状'] + item['中医望闻切诊']
    #         label = [0] * len(syndromes)
    #         for disease in item['证型']:#"证型": "肝阳上亢证|痰湿痹阻证",
    #             label[syndrome2id[disease]] = 1
    #         texts.append(text)
    #         labels.append(label)
    #     return texts, labels

    def load_data(path):
        texts, labels = [], []
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        for item in tqdm(data, desc=f"Loading {path}"):
            text = item['症状'] + item['中医望闻切诊']
            label_vec = [0] * len(syndromes)
            syndrome_str = item.get('证型', '')
            for syn in syndrome_str.split('|'):
                syn = syn.strip()
                if syn in syndrome2id:
                    label_vec[syndrome2id[syn]] = 1
            texts.append(text)
            labels.append(label_vec)
        return texts, labels

    train_texts, train_labels = load_data(train_path)
    val_texts, val_labels = load_data(val_path)
    test_texts, test_labels = load_data(test_path)

    return (MyDataset(train_texts, train_labels, tokenizer),
            MyDataset(val_texts, val_labels, tokenizer),
            MyDataset(test_texts, test_labels, tokenizer))

class BertMultiLabelClassifier(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, labels=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        pooled_output = self.dropout(outputs.pooler_output)
        logits = self.classifier(pooled_output)
        loss = None
        if labels is not None:
            loss_fct = nn.BCEWithLogitsLoss()
            loss = loss_fct(logits, labels)
        return {'loss': loss, 'logits': logits} if loss is not None else {'logits': logits}

def compute_metrics(eval_pred: EvalPrediction):
    logits, labels = eval_pred
    probs = torch.sigmoid(torch.tensor(logits)).numpy()
    preds = (probs > 0.5).astype(int)
    return {
        "accuracy": (preds == labels).mean(),
        "f1_macro": f1_score(labels, preds, average="macro"),
        "f1_micro": f1_score(labels, preds, average="micro")
    }

tokenizer = BertTokenizer.from_pretrained(model_path)
train_dataset, val_dataset, test_dataset = TCM_SD_Data_Loader(tokenizer)

model = BertMultiLabelClassifier.from_pretrained(model_path, num_labels=10)

training_args = TrainingArguments(
    output_dir="./checkpoints",
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=5,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    logging_dir="./logs",
    load_best_model_at_end=True,
    metric_for_best_model="f1_micro",
    no_cuda=True
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

trainer.train()
'''
	•	CrossEntropy：logits 形状 (B, C)，labels 是整数 LongTensor 形状 (B,)（每条是 0..C-1）。
	•	BCEWithLogits：logits 形状 (B, C)，labels 是 multi-hot float 向量形状 (B, C)，元素为 0 或 1。
'''