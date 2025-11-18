import warnings

warnings.filterwarnings('ignore', category=FutureWarning)
import os
import torch
from torch import nn
from torch.nn.parallel import DistributedDataParallel
from tqdm import tqdm
import shutil
import json
import sys
import random
sys.path.append(os.path.abspath(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import numpy as np
import time
from transformers import get_linear_schedule_with_warmup
import numpy as np
import warnings
import torch.nn.functional as F
import math
# from peft import LoraModel, LoraConfig
from transformers import BertTokenizer
from transformers import BertConfig, BertModel, AdamW
# from sklearn.metrics import accuracy_score
from data_utils import TCM_SD_Data_Loader

# 设定随机种子值，以确保输出是确定的
seed_val = 42

random.seed(seed_val)
np.random.seed(seed_val)
torch.manual_seed(seed_val)
torch.cuda.manual_seed_all(seed_val)

class ClassifiyZYBERT(nn.Module):
    def __init__(self, Bertmodel_path, Bertmodel_config):
        super(ClassifiyZYBERT, self).__init__()
        self.PreBert = BertModel.from_pretrained(Bertmodel_path, config=Bertmodel_config)

        self.dropout = nn.Dropout(0.2)

        self.clssificion_diease = nn.Linear(768,4,bias=True)

    def forward(self, batch=None, token_type_ids=None, return_dict=None):
        input_ids = batch[0]#input_ids
        attention_mask = batch[1]
        y = batch[2]  #真实标签
        x_student = self.PreBert(input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask,return_dict=return_dict)#输出的结果为【2，512，768】【2，768】

        diease_yeict = self.clssificion_diease(self.dropout(x_student[1]))

        max_values, _ = torch.max(diease_yeict, dim=1, keepdim=True)#拿出来最大的值
        # 将最大值的位置设置为True，其余位置为False
        yhat_diease = torch.eq(diease_yeict, max_values)#和最大的值相等，将其设置为True

        return { "yhat_raw_diease": diease_yeict,  "yhat_diease": yhat_diease, "y": y}


model_path = '/Volumes/mac_win/models/tiansz/bert-base-chinese'
model_config = BertConfig.from_pretrained('/Volumes/mac_win/models/tiansz/bert-base-chinese/config.json')
tokenizer = BertTokenizer.from_pretrained(model_path)
model = ClassifiyZYBERT(Bertmodel_path=model_path, Bertmodel_config=model_config)
model = model
# gpus = [0, 1]

train_dataloader, test_dataloader, val_dataloader, id2syndrome_dict = TCM_SD_Data_Loader(tokenizer)

lr = 1e-5
optimizer = AdamW(model.parameters(),
                  lr=lr,  # args.learning_rate - default is 5e-5
                  eps=1e-8  # args.adam_epsilon  - default is 1e-8
                  )#定义优化器：AdamW
total_steps = len(train_dataloader)
scheduler = get_linear_schedule_with_warmup(optimizer,
                                            num_warmup_steps=0,
                                            num_training_steps=total_steps)#定义学习率调度器（scheduler）。前若干步学习率线性增加；设为 0 表示不使用 warmup。
epochs = 15
criterions = nn.CrossEntropyLoss()#是用于多分类任务的标准损失函数。

best_micro_metric = 0
best_epoch = 0

for epoch_i in range(1, epochs + 1):
    model.train()
    model.zero_grad()
    sum_loss = 0
    outputs = []
    t_bar = tqdm(enumerate(train_dataloader), total=len(train_dataloader))
    for step, batch in t_bar:
        #batch = [tensor.to('cuda') for tensor in batch]
        #在这里的数据为list，这个list为3个。分别代表的是input_tensor、attention_maski、以及标签。分别为【2，512】，【2，512】，【2，4】4代表的是所有的类别，类似0，0，1，0
        now_res = model(batch=batch, token_type_ids=None, return_dict=False)
        outputs.append({key: value.cpu().detach() for key, value in now_res.items()})


        diease_label = torch.argmax(batch[2], dim=1).long()  # ✅ 转为 [1, 3]

        yhat_raw_diease = now_res['yhat_raw_diease']
        diease_loss = criterions(yhat_raw_diease, diease_label)#这里计算损失函数。

        total_loss =  diease_loss
        sum_loss += total_loss
        avg_loss = sum_loss / step#计算每一步的平均loss

        t_bar.update(1)  # 更新进度
        t_bar.set_description("avg_total_loss:{}, diease_loss:{}".format(avg_loss, diease_loss))  # 更新描述
        t_bar.refresh()  # 立即显示进度条更新结果

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        scheduler.step()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10)#梯度裁剪，防止梯度爆炸。如果某些参数的梯度太大（范数超过10），就会被等比例缩放，保持稳定训练。

    # ===========分割线/下面的是对训练数据集进行评估验证==============
    #每一个epoch进行一次val和test的数据集准确度的验证输出。

    #预测疾病
    yhat_raw_diease = torch.cat([output['yhat_diease'].to(torch.int) for output in outputs]).cpu().detach().numpy()
    y_diease = torch.cat([output['y'] for output in outputs]).cpu().detach().numpy()
    # 使用 np.all 逐行比较疾病的正确率
    comparison_diease = np.all(yhat_raw_diease == y_diease, axis=1)
    matching_rows_count_diease = np.sum(comparison_diease)#在这里只计算True的总和。
    ACC_diease = matching_rows_count_diease / y_diease.shape[0]#然后在这里除以总数
    #ACC是syndrome acc和diease acc的平均值
    total_ACC = ACC_diease
    print('Train:-----------Total ACC:{}, diease ACC:{}'.format(total_ACC, ACC_diease))

    # ===========分割线/下面的是对验证数据集进行评估验证==============

    # model.eval()
    # outputs = []
    # for step, batch in tqdm(enumerate(val_dataloader), total=len(val_dataloader), desc='Dev'):
    #     batch = [tensor.to('cuda') for tensor in batch]
    #     with torch.no_grad():
    #         now_res = model(batch=batch, token_type_ids=None, return_dict=False)
    #     outputs.append({key: value.cpu().detach() for key, value in now_res.items()})
    # #预测疾病
    # yhat_raw_diease = torch.cat([output['yhat_diease'].to(torch.int) for output in outputs]).cpu().detach().numpy()
    # y_diease = torch.cat([output['y'] for output in outputs]).cpu().detach().numpy()
    # # 使用 np.all 逐行比较疾病的正确率
    # comparison_diease = np.all(yhat_raw_diease == y_diease, axis=1)
    # matching_rows_count_diease = np.sum(comparison_diease)
    # ACC_diease = matching_rows_count_diease / y_diease.shape[0]
    # #ACC是syndrome acc和diease acc的平均值
    # total_ACC = ACC_diease
    # print('Dev:-----------Total ACC:{}, diease ACC:{}'.format(total_ACC, ACC_diease))

#===========分割线/下面的是对测试数据集进行评估验证==============

    model.eval()
    outputs = []
    for step, batch in tqdm(enumerate(test_dataloader), total=len(test_dataloader),desc='Test'):
        batch = [tensor.to('cuda') for tensor in batch]
        with torch.no_grad():
            now_res = model(batch=batch, token_type_ids=None, return_dict=False)
        outputs.append({key: value.cpu().detach() for key, value in now_res.items()})
    #预测疾病
    yhat_raw_diease = torch.cat([output['yhat_diease'].to(torch.int) for output in outputs]).cpu().detach().numpy()
    y_diease = torch.cat([output['y'] for output in outputs]).cpu().detach().numpy()
    # print(yhat_raw_diease)
    # 使用 np.all 逐行比较疾病的正确率
    comparison_diease = np.all(yhat_raw_diease == y_diease, axis=1)
    matching_rows_count_diease = np.sum(comparison_diease)
    ACC_diease = matching_rows_count_diease / y_diease.shape[0]
    #ACC是syndrome acc和diease acc的平均值
    total_ACC = ACC_diease
    print('Test:-----------Total ACC:{}, diease ACC:{}'.format(total_ACC, ACC_diease))

    best_ACC_diease = 0

    # 保存最优模型
    if ACC_diease > best_ACC_diease:
        best_ACC_diease = ACC_diease
        best_epoch = epoch_i
        torch.save(model.state_dict(), f'/home/model/public/real_zhangguowen/models/tiansz/best_model_epoch_{epoch_i}.pth')
        print(f"✅ 保存最优模型: epoch {epoch_i}, ACC={ACC_diease:.4f}")
# print(best_epoch)
