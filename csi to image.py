# -*- coding = utf-8 -*-
# @Time : 2022/7/29 15:18
# @Author : 盛君子
# @File : cross.py
# @Software : PyCharm
import matplotlib.pyplot as plt
from pyts.image import GramianAngularField
import scipy.io as sio
from pyts.approximation import PiecewiseAggregateApproximation
from pyts.preprocessing import MinMaxScaler
import numpy as np
from pylab import xticks,yticks,np
from matplotlib.pyplot import MultipleLocator
config = {
            "font.family": 'serif',
            "mathtext.fontset": 'stix',
            "font.serif": ['Times New Roman'],

         }

data = sio.loadmat('./CSIGAF/CSIGAF1/data/1-7-3.mat')
train_data = data['traindata']
test = train_data[0:224, 1].reshape(1, -1)
plt.figure(figsize=(3,3),dpi=600)
ax=plt.axes()
x_major_locator=MultipleLocator(30)
ax.xaxis.set_major_locator(x_major_locator)
#plt.subplots_adjust(top=0.98, bottom = 0.15, right =0.98, left = 0.1, hspace = 0, wspace = 0)
plt.plot(test.reshape(-1,1))
# plt.xlabel('Packet numbers',fontsize=10)
# plt.ylabel('Amplitude',fontsize=10)
# plt.tick_params(labelsize=8)
plt.axis('off')
plt.title('')
plt.savefig('./csi.svg',bbox_inches='tight')



# gaf = GramianAngularField(image_size=224, sample_range=(0, 1), method='summation')
# img_gaf = gaf.fit_transform(test)
# plt.figure(figsize=(3,3),dpi=600)
# ax=plt.axes()
# x_major_locator=MultipleLocator(30)
# y_major_locator=MultipleLocator(30)
# ax.xaxis.set_major_locator(x_major_locator)
# ax.yaxis.set_major_locator(y_major_locator)
# #plt.subplots_adjust(top=0.98, bottom = 0.12, right =0.98, left = 0.1, hspace = 0, wspace = 0)
# plt.tick_params(labelsize=8)
# plt.imshow(img_gaf[0],cmap='RdBu')
# plt.savefig('./gasf.png',bbox_inches='tight')




# gaf = GramianAngularField(image_size=224, sample_range=(0, 1), method='difference')
# img_gaf1 = gaf.fit_transform(test)
# plt.figure(figsize=(3.5,3.5),dpi=600)
# ax=plt.axes()
# x_major_locator=MultipleLocator(30)
# y_major_locator=MultipleLocator(30)
# ax.xaxis.set_major_locator(x_major_locator)
# ax.yaxis.set_major_locator(y_major_locator)
# plt.subplots_adjust(top=0.98, bottom = 0.1, right =1, left = 0.02, hspace = 0, wspace = 0)
# plt.tick_params(labelsize=8)
# plt.imshow(img_gaf1[0],cmap='RdBu')
# import matplotlib.ticker as mticker
# locator = mticker.MultipleLocator(0.5)
# plt.colorbar(ticks=locator,fraction=0.05,pad=0.04)
# plt.savefig('./gadf.pdf',bbox_inches='tight')


from pyts.image import RecurrencePlot

#Recurrence plot transformation
rp = RecurrencePlot(dimension=1, time_delay=1)
X_rp = rp.fit_transform(test)

# Show the results for the first time series
plt.figure(figsize=(3.5,3.5),dpi=600)
ax=plt.axes()
x_major_locator=MultipleLocator(30)
y_major_locator=MultipleLocator(30)
ax.xaxis.set_major_locator(x_major_locator)
ax.yaxis.set_major_locator(y_major_locator)
plt.subplots_adjust(top=0.98, bottom = 0.1, right =1, left = 0.02, hspace = 0, wspace = 0)
plt.tick_params(labelsize=8)
#,cmap='RdBu'
plt.imshow(X_rp[0],cmap='RdBu')
plt.colorbar(fraction=0.05,pad=0.05)
plt.axis('off')
plt.title('')
plt.savefig('./rt.svg',bbox_inches='tight')


# from pyts.image import MarkovTransitionField
# mtf = MarkovTransitionField(n_bins=8)
# fullimage = mtf.transform(test)
# plt.figure(figsize=(3,3),dpi=600)
# ax=plt.axes()
# x_major_locator=MultipleLocator(30)
# y_major_locator=MultipleLocator(30)
# ax.xaxis.set_major_locator(x_major_locator)
# ax.yaxis.set_major_locator(y_major_locator)
# # plt.subplots_adjust(top=0.98, bottom = 0.1, right =1, left = 0.02, hspace = 0, wspace = 0)
# plt.tick_params(labelsize=8)
# plt.imshow(fullimage[0],cmap='RdBu')

# plt.savefig('./mtf.pdf',bbox_inches='tight')


# plt.figure(figsize=(3,3),dpi=600)
# ax=plt.axes()
# x_major_locator=MultipleLocator(1)
# Pxx, freqs, bins, im = plt.specgram(train_data[0:224, 1], NFFT=10, Fs=1000, noverlap=5)
# #plt.subplots_adjust(top=0.98, bottom = 0.12, right =0.98, left = 0.1, hspace = 0, wspace = 0)
# plt.xlabel('Time(s)',fontsize=10)
# plt.ylabel('Frequency (HZ)')
# plt.tick_params(labelsize=8)
# plt.savefig('./stft.pdf',bbox_inches='tight')
# plt.show()
