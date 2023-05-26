from MyNet import MyNet
from CSIResNet import ResNet, BasicBlock
from LoadMyData import MyData
import paddle
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from visualdl import LogWriter
import os
import paddle.nn as nn
import numpy as np
import matplotlib.pyplot as plt
root_dir='data/data141031/CSI-GAF2/data'
train_dir='s_traindata'
vali_dir='s_validata'
tar_dir='target_data'
traindata=MyData(root_dir,train_dir,2)
validata=MyData(root_dir,vali_dir,2)
targetdata=MyData(root_dir,tar_dir,2)
print(len(traindata))
print(len(validata))
batsize=50
EPOCH_NUM=20
learning_rate=0.001
number=0
# 定义数据读取器
train_loader = paddle.io.DataLoader(traindata, batch_size=batsize, shuffle=True,drop_last=True)
valid_loader = paddle.io.DataLoader(validata, batch_size=batsize, shuffle=False,drop_last=True)
target_loader=paddle.io.DataLoader(targetdata, batch_size=batsize, shuffle=True,drop_last=True)
# mode=MyNet(6)
model=ResNet(BasicBlock,[1,1,1,1],6)
# model = ResNet(BasicBlock,[1,1,1,1],6)
# BasicBlock BottleNeck
# model = ResNet(BasicBlock,[1,1,1,1],6)
model_state_dict=paddle.load('model/model2.pdparams')

model.set_state_dict(model_state_dict)
# model.fc = nn.Linear(512, 6)
# number=0
# #61,56,51
for param in model.parameters(): 
        number=number+1 
        if number<61:
           param.stop_gradient = True
        

use_gpu = True
paddle.device.set_device('gpu:0') if use_gpu else paddle.device.set_device('cpu')

# optim = paddle.optimizer.Adam(parameters=model.parameters(),learning_rate=learning_rate)
# 设置损失函数
loss_fn = paddle.nn.CrossEntropyLoss()

# 设置优化器
scheduler = paddle.optimizer.lr.MultiStepDecay(learning_rate=learning_rate, milestones=[4, 9, 13, 17], gamma=0.1)
optim=paddle.optimizer.Adam(parameters=model.parameters(),learning_rate=scheduler)
print('start training ... ')
best_acc = 0
vali_acc = []
train_acc =[]
train_loss=[]
vali_loss=[]
test_acc = 0

for epoch in range(EPOCH_NUM):
    
    model.train()
    train_loss_b=[]
    vali_loss_b=[]
    train_acc_b=[]
    vali_acc_b=[]
    correct_t = 0
    for batch_id, data in enumerate(target_loader()):
        img = data[0]
        label = data[1] 
        # 计算模型输出
        predict_label = model(img)
        # 计算损失函数
        loss = loss_fn(predict_label, label)
        train_loss_b.append(loss.numpy())
        # print(train_loss_b)
        #计算准确率
        correct_t=paddle.metric.accuracy(predict_label,label)
        train_acc_b.append(correct_t.numpy())
        #反向传播
        loss.backward()
        optim.step()
        # 梯度清零
        optim.clear_grad()
    scheduler.step() 
    print("epoch: {},  trainloss is: {}, trainacc is: {}".format(epoch,  np.mean(train_loss_b), np.mean(train_acc_b)))
    train_loss.append(np.mean(train_loss_b))
    train_acc.append(np.mean(train_acc_b))