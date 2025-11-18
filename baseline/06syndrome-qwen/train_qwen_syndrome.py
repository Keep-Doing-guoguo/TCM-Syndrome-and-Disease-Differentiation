#!/usr/bin/env python
# coding=utf-8

"""
@author: zgw
@date: 2025/5/19 14:35
@source from: 
"""
from dataclasses import dataclass
from typing import Optional, List
import torch
from torch.utils.data import Dataset
from transformers import Trainer, TrainingArguments, AutoTokenizer, AutoModelForCausalLM

# ======================= 参数定义 =======================
'''

<|im_start|>system
你是一个文本分类领域的专家，你会接收到一段文本和几个潜在的分类选项，请输出文本内容的正确类型。<|im_end|>\n
<|im_start|>user
文本：患者症状如下：憋喘、乏力、下肢水肿。\n分类项：气虚血瘀证、阳虚水停证、肝肾阴虚证、痰湿痹阻证\n请问正确的证型是哪一个？<|im_end|>\n
<|im_start|>assistant
阳虚水停证<|im_end|>\n

'''
@dataclass
class CustomArguments:
    model_name_or_path: str = "/Volumes/mac_win/models/Qwen1___5-0___5B"
    train_file: str = "train.json"
    eval_file: Optional[str] = "dev.json"
    output_dir: str = "./checkpoints"
    max_seq_length: int = 2048
    use_lora: bool = False

# ======================= 数据预处理 =======================
def format_chatml(system: str, user: str, assistant: str) -> str:
    return (
        f"<|im_start|>system\n{system}<|im_end|>\n"
        f"<|im_start|>user\n{user}<|im_end|>\n"
        f"<|im_start|>assistant\n{assistant}<|im_end|>\n"
    )

from torch.utils.data import Dataset
import torch

class QwenDataset(Dataset):
    def __init__(self, data, tokenizer, max_len):
        self.samples = []
        self.tokenizer = tokenizer
        self.max_len = max_len

        system_prompt = "你是一个文本分类领域的专家，你会接收到一段文本和几个潜在的分类选项，请输出文本内容的正确类型。"

        for item in data:
            user_input = item["prompt"]
            answer = item["label"]

            # 构建 ChatML prompt（注意 <|im_end|> 必须加上）
            instruction_text = (
                f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
                f"<|im_start|>user\n{user_input}<|im_end|>\n"
                f"<|im_start|>assistant\n"
            )
            response_text = f"{answer}<|im_end|>"

            # Tokenize
            instruction = tokenizer(instruction_text, add_special_tokens=False)
            response = tokenizer(response_text, add_special_tokens=False)

            input_ids = instruction["input_ids"] + response["input_ids"]
            attention_mask = instruction["attention_mask"] + response["attention_mask"]

            # 构造 labels：只监督回答部分
            labels = [-100] * len(instruction["input_ids"]) + response["input_ids"]

            # 补上 pad（使用 eos 作为 pad）
            pad_len = max_len - len(input_ids)
            if pad_len > 0:#这个是需要进行补pad的。
                input_ids += [tokenizer.pad_token_id] * pad_len
                attention_mask += [0] * pad_len
                labels += [-100] * pad_len
            else:#这个是需要进行删除的。
                input_ids = input_ids[:max_len]
                attention_mask = attention_mask[:max_len]
                labels = labels[:max_len]

            self.samples.append({
                "input_ids": torch.tensor(input_ids, dtype=torch.long),
                "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
                "labels": torch.tensor(labels, dtype=torch.long),
            })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

# ======================= 主训练逻辑 =======================
def run_train(args: CustomArguments):
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True)
    tokenizer.pad_token_id = tokenizer.eos_token_id

    import json
    with open(args.train_file, encoding="utf-8") as f:
        train_raw = json.load(f)
    train_dataset = QwenDataset(train_raw, tokenizer, args.max_seq_length)

    if args.eval_file:
        with open(args.eval_file, encoding="utf-8") as f:
            eval_raw = json.load(f)
        eval_dataset = QwenDataset(eval_raw, tokenizer, args.max_seq_length)
    else:
        eval_dataset = None

    model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path, trust_remote_code=True)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        evaluation_strategy="steps",
        save_strategy="steps",
        save_steps=100,
        logging_steps=10,
        num_train_epochs=3,
        learning_rate=2e-5,
        gradient_accumulation_steps=8,
        warmup_steps=100,
        fp16=True,
        logging_dir="./logs",
        report_to=[],
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
    )

    trainer.train()

if __name__ == '__main__':
    args = CustomArguments(
        model_name_or_path="/Volumes/mac_win/models/Qwen/Qwen1.5-0.5B",
        train_file="/Volumes/PSSD/NetThink/pythonProject/7-19-Project/TCM-Syndrome-and-Disease-Differentiation/baseline/task1/证型任务提示词.json",
        eval_file="/Volumes/PSSD/NetThink/pythonProject/7-19-Project/TCM-Syndrome-and-Disease-Differentiation/baseline/task1/证型任务提示词.json",
        output_dir="./checkpoints"
    )
    run_train(args)
# ✅ 使用 DataCollatorForSeq2Seq 的优势：
# 	•	自动处理 padding；
# 	•	自动对 labels 中 pad 部分设为 -100（避免干扰 loss）；
# 	•	动态 padding：每个 batch 只 pad 到该 batch 内最大长度，更高效。