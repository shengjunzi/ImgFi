import os
import random
from DSAN import DSAN
from MyNet import MyNet
from LOAD import MyData
import paddle
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import paddle.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import math
import pandas as pd
import math
def ganlist(gaf_class):
    train_ratio=0.7
    train=open('data/data141031/CSIGAF'+str(gaf_class)+'/s_traindata.txt','w+')
    val=open('data/data141031/CSIGAF'+str(gaf_class)+'/s_validata.txt','w+')
    target=open('data/data141031/CSIGAF'+str(gaf_class)+'/target_data.txt','w+')
    source=open('data/data141031/CSIGAF'+str(gaf_class)+'/source_data.txt','w+')
    test=open('data/data141031/CSIGAF'+str(gaf_class)+'/test_data.txt','w+')
    root_dir='data/data141031/CSIGAF'+str(gaf_class)
    train_dir='data'
    path=os.path.join(root_dir,train_dir)
    img_path=os.listdir(path)
    for line in img_path:
            if random.uniform(0, 1) < train_ratio: 
         		train.writelines(line)
                  train.write('\r\n')   
              
            else:
                if random.uniform(0, 1) < 0.5:
                    val.writelines(line)
                    val.write('\r\n')
                else:
                    test.writelines(line)
                    test.write('\r\n')      
    train.close()
    val.close()
    test.close()
 