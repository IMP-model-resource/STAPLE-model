# -*- coding: utf-8 -*-
"""
Created on Tue Mar 29 11:53:06 2022

@author: Geng
"""

import os
import sys 
import gc
import imageio
import numpy as np
import utils_PG
import provider
import exporter
import ParamOptimizer
import CreateModel
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping
import predicting
from sklearn.preprocessing import label_binarize
import time

time_start = time.time()
nodata = -9999
# 采样及网络参数
winSize = 11
stride = 6
nframes = 5 # 期
    
initialMap = provider.load_initialMap()
groundTruth = provider.load_groundTruth()
drivingFactors = provider.load_drivingFactors() # 载入的同时将 drivingFactors 归一化
ascMetaData = provider.load_metaData()
n_class = np.nanmax(initialMap)    # 统计土地利用共有几类

# # 分区采样用
# initialMap = np.load(r'E:\graduate\LandscapeCA\exp\input\subRegion\4\0.npy')
# groundTruth = np.load(r'E:\graduate\LandscapeCA\exp\input\subRegion\4\1.npy')
# drivingFactors = []
# for i in range(2,75):
#     drivingFactors.append(np.load(os.path.join(r'E:\graduate\LandscapeCA\exp\input\subRegion\4',str(i)+'.npy')))
# n_class = 6

dataList = []
dataList.append(initialMap)
# dataList.append(groundTruth)
for df in drivingFactors:
    if df.shape != initialMap.shape:
        print("error：Missmatching size between drivingFactor and initialMap")     # 若 initialMap 和 drivingFactor 尺寸不一致，则报错
        sys.exit(0)
    dataList.append(df)  # 将 drivingFactors 与 initialMap 合成 dataList
    
# 构建训练集
# ySamples, xSamples = utils_PG.sampling_2dcnn_PAS(groundTruth, dataList[1:], winSize, Rate = 0.1, nodata=-9999) # PAS 采样策略
# ySamples, xSamples = utils_PG.sampling_point_PEAS(initialMap, groundTruth, dataList[1:])
# ySamples, xSamples = utils_PG.sampling_2dcnn_PEAS(initialMap, groundTruth, dataList[1:], winSize, Rate = 0.1, nodata=-9999)
ySamples, xSamples = utils_PG.sampling_3dcnn_PEAS(initialMap, groundTruth, dataList[1:], nframes, winSize, Rate = 0.05, nodata = nodata)    # 从变化和不变像元采样
# ySamples, xSamples = utils_PG.sampling_3dcnn_PAS(initialMap, groundTruth, dataList[1:], nframes, winSize, method = 'uniform', Rate = 0.05, nodata=nodata)
exporter.saveSamples(ySamples, xSamples)
gc.collect()

# ySamples = np.load(r'E:\graduate\LandscapeCA\exp\output\PG_module\3dcnn\Samples_LEAS\ySamples.npy')
# xSamples = np.load(r'E:\graduate\LandscapeCA\exp\output\PG_module\3dcnn\Samples_PEAS\xSamples.npy')

ySamples = ySamples-1   # 训练中需要把label调整为0至n-1
# ySamples, xSamples = utils_PG.sampleExpanding(ySamples, xSamples, fold=1) # 样本扩充
ySamples, xSamples = utils_PG.shuffle(ySamples, xSamples)

model = utils_PG.train_3dcnn(ySamples, xSamples)
utils_PG.ROC_test(model,ySamples,xSamples,n_class)

# model = tf.keras.models.load_model(r'E:\graduate\LandscapeCA\exp\output\PG_module\fcn\Model_fcn_PEAS')

# 预测转移概率
# prob = predicting.predict_fcn(dataList[1:], model, nodata=-9999)
# prob = predicting.predict_2dcnn(dataList[1:], model, winSize, nPacking=5000, nodata=nodata)
prob = predicting.predict_3dcnn(dataList[1:], model, winSize, nframes, nPacking=5000, nodata=nodata)

# 保存转移概率
exporter.savePotentials(prob,ascMetaData)

time_end = time.time()
print('=================== Program running time: %dmin - %ds ===================' 
      % ((time_end - time_start)//60,(time_end - time_start)-(time_end - time_start)//60*60))











