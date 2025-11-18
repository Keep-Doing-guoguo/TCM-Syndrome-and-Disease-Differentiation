task1:
1.bert_disese：
该类主要是用来对疾病来进行分类，这是一个单分类的问题，一个问题对应一个标签。使用CrossEntropyLoss作为损失函数，使用AdamW作为优化器。模型的输入是一个句子，输出是一个标签。模型的训练和测试都在该类中完成。

1.1.data_utils：
该方法主要是对疾病的数据进行处理。
1.1.bert_model：
这个主要是用来使用LAAT 注意力机制+bert模型。该注意力主要是对多标签问题。

2.bert_symptom：
该类主要是用来对症状进行分类，这是一个多分类的问题，一个问题对应多个标签。使用BCEWithLogitsLoss作为损失函数，使用AdamW作为优化器。模型的输入是一个句子，输出是一个标签。模型的训练和测试都在该类中完成。




task2：
1.data_process:
这个是用来处理数据用的。这里可以处理症状的、处理疾病的。
以及对数据处方的预处理。



答：大约 512 个汉字以内的纯中文文本。如果有英文、数字、标点混合，实际能处理的字符数可能略多。

outputs.last_hidden_state.shape == [2, 512, 768]
outputs.pooler_output.shape == [2, 768]


名称 形状 含义
last_hidden_state
[B, L, H] = [2, 512, 768]
每个 token 的表示向量（768维）

pooler_output
[B, H] = [2, 768]
[CLS] 位置的向量（用于分类任务）

✅ 用途区别
	•	last_hidden_state：
	•	常用于序列标注（如 NER、QA、注意力机制处理 token）
	•	每个 token 都有一个表示
	•	pooler_output：
	•	用于句子级任务，如分类、情感分析
	•	是 CLS token 的输出 + 一个非线性变换（如 tanh）


