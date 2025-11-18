#!/usr/bin/env python
# coding=utf-8

"""
@author: zgw
@date: 2025/10/30 15:12
@source from: 
"""
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
split_9_1.py
把一个 JSON 数组 或 JSONL 文件 按 9:1 划分为 train/test (输出 jsonl 格式)。

用法:
    python 划分数据集.py --input TCM-TBOSD-train.json --out-prefix output --seed 42
    处理完之后，还需要手动将其变为list，直接在数据里面进行变换即可。
    

输出:
    train.json
    test.json
"""

import json
import random
import argparse
from pathlib import Path
from typing import List, Any

def read_input(path: Path) -> List[Any]:
    text = path.read_text(encoding='utf-8').strip()
    if not text:
        return []
    # 先尝试解析为 JSON 数组
    try:
        obj = json.loads(text)
        if isinstance(obj, list):
            return obj
    except Exception:
        pass
    # 回退为 jsonl：每行一个 json
    items = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        items.append(json.loads(line))
    return items

def write_jsonl(items: List[Any], path: Path):
    with path.open('w', encoding='utf-8') as f:
        for it in items:
            f.write(json.dumps(it, ensure_ascii=False) + '\n')

def split_and_save(items: List[Any], out_prefix: Path, seed: int = 42):
    random.seed(seed)
    n = len(items)
    if n == 0:
        raise ValueError("输入数据为空")
    idxs = list(range(n))
    random.shuffle(idxs)
    n_test = max(1, int(n * 0.1))  # 至少保留1个作为测试集
    test_idxs = set(idxs[:n_test])
    train_items = [items[i] for i in range(n) if i not in test_idxs]
    test_items = [items[i] for i in range(n) if i in test_idxs]

    train_path = out_prefix.with_suffix('.train.jsonl')
    test_path = out_prefix.with_suffix('.test.jsonl')
    write_jsonl(train_items, train_path)
    write_jsonl(test_items, test_path)

    print(f"总样本数: {n}")
    print(f"训练集: {len(train_items)} -> {train_path}")
    print(f"测试集: {len(test_items)} -> {test_path}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--input', '-i', required=True, help='输入文件 (json array 或 jsonl)')
    ap.add_argument('--out-prefix', '-o', default='split_output', help='输出前缀 (会生成 <prefix>.train.jsonl 和 <prefix>.test.jsonl)')
    ap.add_argument('--seed', type=int, default=42, help='随机种子（默认为42）')
    args = ap.parse_args()

    inp = Path(args.input)
    if not inp.exists():
        raise FileNotFoundError(f"找不到输入文件: {inp}")

    items = read_input(inp)
    split_and_save(items, Path(args.out_prefix), seed=args.seed)

if __name__ == '__main__':
    main()