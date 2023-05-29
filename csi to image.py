# -*- coding = utf-8 -*-
# @Time : 2022/7/29 15:18
# @Author : 盛君子
# @File : cross.py
# @Software : PyCharm
# import os
# from matplotlib import image
# import scipy.io as sio
# from pyts.image import GramianAngularField
# from pyts.image import RecurrencePlot
# path3 = "./RT/data/"
# isExists = os.path.exists(path3)
# if not isExists:
#     os.makedirs(path3)
# path4='./CSIdata/CSI2/data/'
# img_path=os.listdir(path4)
# for line in img_path:
#         data = sio.loadmat('./CSIdata/CSI2/data/'+line)
#         train_data = data['traindata']
#         img_num = train_data.shape[1]
#         img_size =90
#         count = 1
#         for k in range(img_num):
#             test = train_data[0:90, k].reshape(1, -1)
#             rp = RecurrencePlot(dimension=1, time_delay=1)
#             X_rp = rp.fit_transform(test)
#             imagename = f"./RT/data/" +line.split('.')[0] +'-'+str(count)+ ".png"
#             image.imsave(imagename, X_rp[0])
#             count = count + 1

import os
from matplotlib import image
import scipy.io as sio
from pyts.image import GramianAngularField
path3 = "./GADF/data/"
isExists = os.path.exists(path3)
if not isExists:
    os.makedirs(path3)
path4='./CSIdata/CSI1/data/'
img_path=os.listdir(path4)
for line in img_path:
        data = sio.loadmat('./CSIdata/CSI1/data/'+line)
        train_data = data['traindata']
        img_num = train_data.shape[1]
        img_size =224
        count = 1
        for k in range(img_num):
            test = train_data[0:224, k].reshape(1, -1)
            gaf = GramianAngularField(image_size=2240, sample_range=(0, 1), method='difference')
            img_gaf1 = gaf.fit_transform(test)
#             rp = RecurrencePlot(dimension=1, time_delay=1)
#             X_rp = rp.fit_transform(test)
            imagename = f"./GADF/data/" +line.split('.')[0] +'-'+str(count)+ ".png"
            image.imsave(imagename, img_gaf1[0])
            count = count + 1
