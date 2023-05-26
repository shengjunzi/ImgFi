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

# Set the directory paths for training and validation data
root_dir='data/data141031/CSI-GAF2/data'
train_dir='s_traindata'
vali_dir='s_validata'

# Load the training and validation data using MyData class
traindata=MyData(root_dir,train_dir,2)
validata=MyData(root_dir,vali_dir,2)

# Print the length of training and validation data
print(len(traindata))
print(len(validata))

# Set the batch size, number of epochs, and learning rate
batsize=128
EPOCH_NUM=15
learning_rate=0.001

# Create data loaders for training and validation data
train_loader = paddle.io.DataLoader(traindata, batch_size=batsize, shuffle=True,drop_last=True)
valid_loader = paddle.io.DataLoader(validata, batch_size=batsize, shuffle=False,drop_last=True)

# Create the neural network model
model=MyNet(6)   

# Set the device to GPU if available, otherwise use CPU
use_gpu = True
paddle.device.set_device('gpu:0') if use_gpu else paddle.device.set_device('cpu')

# Define the loss function and optimizer
loss_fn = paddle.nn.CrossEntropyLoss()
scheduler = paddle.optimizer.lr.MultiStepDecay(learning_rate=learning_rate, milestones=[4, 9, 13, 17], gamma=0.1)
optim=paddle.optimizer.Adam(parameters=model.parameters(),learning_rate=scheduler)

# Train the model
print('start training ... ')
best_acc = 0
vali_acc = []
train_acc =[]
train_loss=[]
vali_loss=[]
test_acc = 0

for epoch in range(EPOCH_NUM):
    
    # Set the model to training mode
    model.train()
    train_loss_b=[]
    vali_loss_b=[]
    train_acc_b=[]
    vali_acc_b=[]
    correct_t = 0
    
    # Iterate over the training data batches
    for batch_id, data in enumerate(train_loader()):
        img = data[0]
        label = data[1] 
        predict_label = model(img)
        loss = loss_fn(predict_label, label)
        train_loss_b.append(loss.numpy())
        correct_t=paddle.metric.accuracy(predict_label,label)
        train_acc_b.append(correct_t.numpy())
        loss.backward()
        optim.step()
        optim.clear_grad()
    
    # Update the learning rate using the scheduler
    scheduler.step() 
    
    # Print the training loss and accuracy for the current epoch
    print("epoch: {},  trainloss is: {}, trainacc is: {}".format(epoch,  np.mean(train_loss_b), np.mean(train_acc_b)))
    train_loss.append(np.mean(train_loss_b))
    train_acc.append(np.mean(train_acc_b))
    correct_t = 0
    
    # Set the model to evaluation mode
    model.eval()
    with paddle.no_grad():
        # Iterate over the validation data batches
        for batch_id, data in enumerate(valid_loader):
           img = data[0]
           label = data[1] 
     
           predict_label = model(img)
          
           loss = loss_fn(predict_label, label)
           vali_loss_b.append(loss.numpy())
        
           correct_t=paddle.metric.accuracy(predict_label,label)
           vali_acc_b.append(correct_t.numpy())

        # Print the validation loss and accuracy for the current epoch
        print("epoch: {},  valiloss is: {}, valiacc is: {}".format(epoch,  np.mean(vali_loss_b), np.mean(vali_acc_b)))
        vali_loss.append( np.mean(vali_loss_b))
        vali_acc.append(np.mean(vali_acc_b))
        if np.mean(vali_acc_b) > best_acc:
            best_acc = np.mean(vali_acc_b)
            best_epoch = epoch
            paddle.save(model.state_dict(), "model/model2.pdparams")
print("best_acc:{},best_epoch:{}".format (best_acc,best_epoch))