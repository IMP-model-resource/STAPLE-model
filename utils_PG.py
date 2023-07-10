# -*- coding: utf-8 -*-
"""
Created on Tue Mar 29 11:53:52 2022

@author: Geng
"""

import numpy as np
import os
from scipy import ndimage
import random
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt
from collections import  Counter
import ParamOptimizer
import CreateModel
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping
import exporter

def normalizing(data, nodata=-9999): # 输入原始npy矩阵，输出归一化后的npy矩阵
    temp = np.where(data==nodata,np.nan,data)
    Max = np.nanmax(temp)
    Min = np.nanmin(temp)
    data_norm = np.where(data==nodata,nodata,(temp-Min)/(Max-Min))
    return data_norm

def edgeExtension(data,value,r):  # 样本矩阵的边缘扩充
    m,n = np.shape(data)
    result = np.full((m+(2*r),n+(2*r)),value).astype(np.float64)
    result[r:r+m,r:r+n] = data
    return result

def ySampling (data,winSize,stride,nodata=-9999):
    m,n = np.shape(data)
#   r = int((winSize-1)/2)
    temp = np.where(data==nodata,np.nan,data) # 训练时，将nodata转为nan
    #temp = np.where(data==nodata,0,data) # 预测时
    row_start = int((winSize-1)/2)
    col_start = int((winSize-1)/2)
    row_end = int(row_start+(((m-winSize)//stride)*stride))
    col_end = int(col_start+(((n-winSize)//stride)*stride))
    nSamples = int(((row_end-row_start)/stride+1)*((col_end-col_start)/stride+1))    
    y = np.zeros(nSamples)
    ii = 0
    for i in range(row_start,row_end+1,stride):     # 注意是行优先
        for j in range(col_start,col_end+1,stride):
            y[ii] = temp[i,j]
            ii = ii + 1     
    return y
        
def xSampling (data,winSize,stride,nodata=-9999):
    m,n = np.shape(data)
    r = int((winSize-1)/2)
    temp = np.where(data==nodata,np.nan,data) # 训练时，将nodata转为nan
    #temp = np.where(data==nodata,0,data) # 预测时
    row_start = int((winSize-1)/2)
    col_start = int((winSize-1)/2)
    row_end = int(row_start+(((m-winSize)//stride)*stride))
    col_end = int(col_start+(((n-winSize)//stride)*stride))
    # nSamples = int((row_end/stride)*(col_end/stride))
    nSamples = int(((row_end-row_start)/stride+1)*((col_end-col_start)/stride+1))     
    x = np.zeros([nSamples,winSize,winSize])
    ii = 0
    for i in range(row_start,row_end+1,stride):     # 注意是行优先
        for j in range(col_start,col_end+1,stride):
            x[ii] = temp[i-r:i+r+1,j-r:j+r+1]
            ii = ii + 1     
    return x

def dropNoData(data): # 输入一个list，第一层为landuse的ndarray，其余层分别为每个自变量的ndarray
    flagNan = np.zeros(len(data[0]))
    for i_sample in range(len(data[0])):    # 第i号样本
        if np.isnan(data[0][i_sample]):
            flagNan[i_sample] = True
            continue;
        for k_arg in range(1,len(data)):      # 第k号变量
            if True in np.isnan(data[k_arg][i_sample]): 
                flagNan[i_sample] = True
                break;

    indexNan = np.argwhere(flagNan==1)
    for k_arg in range(len(data)):      # 第k号变量,k有9
        data[k_arg] = np.delete(data[k_arg],indexNan.astype(int),axis=0)
    return data



def sampling_rf_PAS(yData, xData, stride, nodata=-9999):
    data = [] # 准备将采样结果摞在一起
    # 读土地利用数据，并采样
    landuse = yData
    y_samps = ySampling(landuse,1,stride,nodata=-9999)
    data.append(y_samps)
    print('Training: landuse successfully sampled')
    # 读协变量数据，并采样
    for i in range(len(xData)):
        x = xData[i]
        print("Sampling drivingFactor {0}".format(i+1))
        x_samps = ySampling(x, 1, stride, nodata)
        data.append(x_samps)
    
    # 删除含有NoData的样本
    flagNan = np.zeros(len(data[0]))
    for i_sample in range(len(data[0])):    # 第i号样本   
        for k_arg in range(len(data)):      # landuse及第k号变量
            if np.isnan(data[k_arg][i_sample]): 
                flagNan[i_sample] = True
                break;
            
    indexNan = np.argwhere(flagNan==1)
    for k_arg in range(len(data)):      # 第k号变量,k有9
        data[k_arg] = np.delete(data[k_arg],indexNan.astype(int),axis=0)
    
    ySamples = data[0]
    xSamples = np.zeros([data[1].shape[0],len(data)-1])   # nSamples,winSize,winSize,nDrivingFactors
    for k in range(xSamples.shape[1]):
        xSamples[:,k] = data[k+1]
    # for k in range(xSamples.shape[3]):
    #     xSamples[:,:,:,k] = normalize(xSamples[:,:,:,k])
    
    return ySamples.astype(np.uint8), xSamples.astype(np.float32)

def showModelPerformance(history):
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    # plt.ylim([0, 1])
    plt.legend(loc='lower right')
    plt.show()

    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('loss')
    # plt.ylim([0, 1])
    plt.legend(loc='upper right')
    plt.show()

def train_2dcnn(ySamples, xSamples):
    n_class = np.max(ySamples) + 1
    # 超参寻优
    # lr_section = np.arange(0.001,0.01+0.001,0.001)   # GridSearch的搜索域
    # batchSize_section = np.arange(32,256+32,32)
    # print('================== Optimizing Hyper-Parameters ==================')
    # lr_best, bs_best, valAccuracy_hst = ParamOptimizer.Gridsearch_2dcnn(ySamples, xSamples, n_class, lr_section, batchSize_section)
    # ParamOptimizer.show(valAccuracy_hst, lr_section, batchSize_section)

    # 搭建2dcnn网络
    model = CreateModel.Create_2dcnn(inputShape=xSamples.shape[1:], output_nclass=n_class)
    model.compile(loss='sparse_categorical_crossentropy',
                  optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), metrics=['accuracy'])
    print(model.summary())
    # 设置EarlyStop，防止过拟合
    es = EarlyStopping(monitor='val_loss', patience=20, mode='min', verbose=1, restore_best_weights=True)

    # 开始训练
    print('================== Training ==================')
    history = model.fit(x=xSamples, y=ySamples, validation_split=0.25, epochs=500, batch_size=128, verbose=2, callbacks=[es])
    showModelPerformance(history)
    # 保存模型
    exporter.saveModel(model)
    
    return model

def train_3dcnn(ySamples, xSamples):
    n_class = np.max(ySamples) + 1
    # 超参寻优
    # lr_section = np.arange(0.0002,0.004+0.0002,0.0002)   # GridSearch的搜索域
    # batchSize_section = np.arange(32,256+32,32)
    # print('================== Optimizing Hyper-Parameters ==================')
    # lr_best, bs_best, valAccuracy_hst = ParamOptimizer.Gridsearch_3dcnn(ySamples, xSamples, n_class, lr_section, batchSize_section)
    # ParamOptimizer.show(valAccuracy_hst, lr_section, batchSize_section)
    
    # 搭建3dcnn网络
    model = CreateModel.Create_3dcnn(inputShape=xSamples.shape[1:], output_nclass=n_class)
    model.compile(loss='sparse_categorical_crossentropy',
                  optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), metrics=['accuracy'])
    print(model.summary())
    # 设置EarlyStop，防止过拟合
    es = EarlyStopping(monitor='val_loss', patience=20, mode='min', verbose=1, restore_best_weights=True)

    # 开始训练
    print('================== Training ==================')
    history = model.fit(x=xSamples, y=ySamples, validation_split=0.25, epochs=500, batch_size=128, verbose=2, callbacks=[es])
    showModelPerformance(history)
    # 保存模型
    exporter.saveModel(model)
    
    return model
    
    
def train_lr(ySamples, xSamples):
    model = CreateModel.Create_lr()
    print('================== Training ==================')
    model.fit(xSamples,ySamples)
    return model

def train_rf(ySamples, xSamples):
    model = CreateModel.Create_rf()
    print('================== Training ==================')
    model.fit(xSamples, ySamples)
    return model

def train_fcn(ySamples, xSamples):
    n_class = np.max(ySamples) + 1
    model = CreateModel.Create_fcn(input_nFeatures=xSamples.shape[1], output_nclass=n_class)
    model.compile(loss='sparse_categorical_crossentropy',
                  optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), metrics=['accuracy'])
    print(model.summary())
    # 设置EarlyStop，防止过拟合
    es = EarlyStopping(monitor='val_loss', patience=20, mode='min', verbose=1, restore_best_weights=True)
    # 开始训练
    print('================== Training ==================')
    history = model.fit(x=xSamples, y=ySamples, validation_split=0.25, epochs=500, batch_size=128, verbose=2, callbacks=[es])
    showModelPerformance(history)
    # 保存模型
    exporter.saveModel(model)
    
    return model


def sampleExpanding(ySamples, xSamples, fold=2):    # fold=1扩充为2倍，fold=2扩充为4倍    
    for i in range(1,fold+1):        
        extra_Samples = ndimage.rotate(xSamples,180/i,axes=(len(xSamples.shape)-3,len(xSamples.shape)-2))
        xSamples = np.append(xSamples,extra_Samples,axis=0)
        ySamples = np.append(ySamples, ySamples)
    return ySamples, xSamples

def shuffle(ySamples, xSamples):
    index = [i for i in range(len(ySamples))]
    random.shuffle(index)
    xSamples = xSamples[index]
    ySamples = ySamples[index]
    return ySamples, xSamples

def roc(y,prob):
    fpr,tpr,threshold = roc_curve(y,prob) ###计算真正率和假正率
    auc_value = auc(fpr,tpr) ###计算auc的值
    plt.figure()
    lw = 2
    plt.figure(figsize=(10,10))
    plt.plot(fpr, tpr, color='darkorange',
             lw=lw, label='ROC curve (AUC_value = %0.3f)' % auc_value) ###假正率为横坐标，真正率为纵坐标做曲线
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.tick_params(labelsize=18)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=18)
    plt.ylabel('True Positive Rate', fontsize=18)
    # plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right", fontsize=18)
    plt.show()
    return auc_value

def ROC_test(model,ySamples,xSamples,n_class):
    # train_prob = model.predict(xSamples)  # 2dcnn\3dcnn用
    train_prob = model.predict_proba(xSamples)  # rf用
    train_auc_values = np.zeros(n_class)   
    y_label = label_binarize(ySamples, classes=list(range(n_class))) 
    for i in range(n_class):
        train_auc_values[i] = roc(y_label[:, i], train_prob[:, i])
        
def ySampling_byIndex (data,indexs,nodata=-9999):
    m,n = np.shape(data)
    data = np.where(data==nodata,np.nan,data) # 训练时，将nodata转为nan
    nSamples = len(indexs)
    y = np.zeros(nSamples)
    for k in range(nSamples):
        i,j = indexs[k]
        y[k] = data[i,j]  
    return y
    
def xSampling_byIndex (data,indexs,winSize,nodata=-9999):
    m,n = np.shape(data)
    r = int((winSize-1)/2)
    data = np.where(data==nodata,np.nan,data) # 训练时，将nodata转为nan
    #temp = np.where(data==nodata,0,data) # 预测时
    nSamples = len(indexs)  
    x = np.zeros([nSamples,winSize,winSize])
    for k in range(nSamples):
        i,j = indexs[k]+ [r,r]
        x[k,:,:] = data[i-r:i+r+1,j-r:j+r+1]   
    return x
     

    
# ======================================================== 采用PAS的采样 ===================================================

def sampling_2dcnn_PAS(landuse, xData, winSize, Rate = 0.1, nodata=-9999):
    r = int((winSize-1)/2)          # 窗口半径
    data = []
    landuse = np.where(np.isnan(landuse),nodata,landuse)  # 确保没有nan, 避免出现判断nan==nan的问题
    n_class = np.max(landuse)    # 统计土地利用共有几类
    N_Pix = np.sum(landuse != nodata)
    n_Pix_class = np.zeros(n_class)
    for i in range(n_class):
        n_Pix_class[i] = np.sum(landuse == i + 1)
    p = np.array([N_Pix/n_Pix for n_Pix in n_Pix_class])
    p_select = np.zeros(landuse.size)
    landuse_flat = landuse.flatten()
    for i in range(len(p_select)):
        if landuse_flat[i] == nodata:
            p_select[i] = 0
        else:
            p_select[i] = p[(landuse_flat[i]-1).astype(np.int8)]
    p_select = p_select/np.sum(p_select)     # 抽中的概率调整为0-1
    index_selected = np.random.choice(landuse.size, size=int(np.ceil(N_Pix*Rate)), replace=False, p=p_select)
    index_selected = np.array(np.unravel_index(index_selected, landuse.shape)).T
    
    y_samps = ySampling_byIndex(landuse, index_selected)
    data.append(y_samps)
    
    # 读协变量数据，并采样
    for i in range(len(xData)):
        print("Sampling drivingFactor {0}".format(i+1))
        x = xData[i]        
        x = edgeExtension(x,nodata,r)
        x_samps = xSampling_byIndex(x, index_selected, winSize)
        data.append(x_samps)
        
    # 删除含有NoData的样本
    data = dropNoData(data)
    
    ySamples = data[0]
    xSamples = np.zeros([data[1].shape[0],data[1].shape[1],data[1].shape[2],len(data)-1])   # nSamples,winSize,winSize,nDrivingFactors
    for k in range(xSamples.shape[3]):
        xSamples[:,:,:,k] = data[k+1]
        
    print(Counter(ySamples))
    return ySamples.astype(np.uint8), xSamples.astype(np.float32)


    
#======================================================== 采用LEAS的采样 ===================================================

def sampling_2dcnn_LEAS(landuse_1, landuse_2, xData, winSize, method = 'uniform', Rate = 0.5, nodata=-9999):
    r = int((winSize-1)/2)          # 窗口半径
    data = []
    landuse_1 = np.where(np.isnan(landuse_1),nodata,landuse_1)  # 确保没有nan，避免出现判断nan==nan的问题
    landuse_2 = np.where(np.isnan(landuse_2),nodata,landuse_2)
    index_change = np.argwhere(landuse_1!=landuse_2)
    
    y_samps = ySampling_byIndex(landuse_2, index_change)
    data.append(y_samps)
    
    # 读协变量数据，并采样
    for i in range(len(xData)):
        print("Sampling drivingFactor {0}".format(i+1))
        x = xData[i]        
        x = edgeExtension(x,nodata,r)
        x_samps = xSampling_byIndex(x, index_change, winSize)
        data.append(x_samps)
        
    # 删除含有NoData的样本
    data = dropNoData(data)
    
    ySamples = data[0]
    xSamples = np.zeros([data[1].shape[0],data[1].shape[1],data[1].shape[2],len(data)-1])   # nSamples,winSize,winSize,nDrivingFactors
    for k in range(xSamples.shape[3]):
        xSamples[:,:,:,k] = data[k+1]
        
    ################# 不同采样方式 ######################
    landuse_1 = np.where(landuse_1==nodata,np.nan,landuse_1)
    n_class = np.nanmax(landuse_1).astype(np.int8)    # 统计土地利用共有几类
    
    N_Pix = ySamples.size   # 样本总数    
    n_Pix_class = np.zeros(n_class)     # 各个类别的样本数
    for i in range(n_class):
        n_Pix_class[i] = np.sum(ySamples==i+1)
    
    if (method == 'uniform'):       # 均匀抽样
        p = np.array([N_Pix/n_Pix for n_Pix in n_Pix_class])
        p_select = np.zeros_like(ySamples)
        for i in range(len(p_select)):
            p_select[i] = p[(ySamples[i]-1).astype(np.int8)]
        p_select = p_select/np.sum(p_select)     # 抽中的概率调整为0-1
        
        index_selected = np.random.choice(ySamples.size, size=int(np.ceil(N_Pix*Rate)), replace=False, p=p_select)
        ySamples = ySamples[index_selected]
        xSamples = xSamples[index_selected,:,:,:]
    
    if (method == 'random'):    # 随机抽样
        p_select = np.ones_like(ySamples)
        p_select = p_select/np.sum(p_select)     # 抽中的概率调整为0-1
        index_selected = np.random.choice(ySamples.size, size=int(np.ceil(N_Pix*Rate)), replace=False, p=p_select)
        ySamples = ySamples[index_selected]
        xSamples = xSamples[index_selected,:,:,:]
        
    print('Original total sample amount: {0}'.format(ySamples.size))
    print(Counter(ySamples))
    
    return ySamples.astype(np.uint8), xSamples.astype(np.float32)

def sampling_3dcnn_LEAS(landuse_1, landuse_2, xData, nframes, winSize, method = 'uniform', Rate = 0.5, nodata=-9999):
    r = int((winSize-1)/2)          # 窗口半径
    nvariables = int(len(xData)/nframes) # 自变量数
    data = []
    landuse_1 = np.where(np.isnan(landuse_1),nodata,landuse_1)  # 确保没有nan，避免出现判断nan==nan的问题
    landuse_2 = np.where(np.isnan(landuse_2),nodata,landuse_2)
    index_change = np.argwhere(landuse_1!=landuse_2)
    
    y_samps = ySampling_byIndex(landuse_2, index_change)
    data.append(y_samps)
    
    # 读协变量数据，并采样
    for i in range(len(xData)):
        print("Sampling drivingFactor {0}".format(i+1))
        x = xData[i]        
        x = edgeExtension(x,nodata,r)
        x_samps = xSampling_byIndex(x, index_change, winSize)
        data.append(x_samps)
        
    # 删除含有NoData的样本
    data = dropNoData(data)
    
    ySamples = data[0]
    xSamples = np.zeros([data[1].shape[0],nframes,data[1].shape[1],data[1].shape[2],nvariables])   # nSamples,nframes,winSize,winSize,nDrivingFactors
    for f in range(nframes):
        for v in range(nvariables):
            xSamples[:,f,:,:,v] = data[f*nvariables+v+1]
        
    ################# 不同采样方式 ######################
    landuse_1 = np.where(landuse_1==nodata,np.nan,landuse_1)
    n_class = np.nanmax(landuse_1).astype(np.int8)    # 统计土地利用共有几类
    
    N_Pix = ySamples.size   # 样本总数    
    n_Pix_class = np.zeros(n_class)     # 各个类别的样本数
    for i in range(n_class):
        n_Pix_class[i] = np.sum(ySamples==i+1)
    
    if (method == 'uniform'):       # 均匀抽样
        p = np.array([N_Pix/n_Pix for n_Pix in n_Pix_class])
        p_select = np.zeros_like(ySamples)
        for i in range(len(p_select)):
            p_select[i] = p[(ySamples[i]-1).astype(np.int8)]
        p_select = p_select/np.sum(p_select)     # 抽中的概率调整为0-1
        
        index_selected = np.random.choice(ySamples.size, size=int(np.ceil(N_Pix*Rate)), replace=False, p=p_select)
        ySamples = ySamples[index_selected]
        xSamples = xSamples[index_selected,:,:,:,:]
    
    if (method == 'random'):    # 随机抽样
        p_select = np.ones_like(ySamples)
        p_select = p_select/np.sum(p_select)     # 抽中的概率调整为0-1
        index_selected = np.random.choice(ySamples.size, size=int(np.ceil(N_Pix*Rate)), replace=False, p=p_select)
        ySamples = ySamples[index_selected]
        xSamples = xSamples[index_selected,:,:,:,:]
        
    print('Original total sample amount: {0}'.format(ySamples.size))
    print(Counter(ySamples))
    
    return ySamples.astype(np.uint8), xSamples.astype(np.float32)

def sampling_point_LEAS(landuse_1, landuse_2, xData, method = 'uniform', Rate = 0.01, nodata=-9999):        # 适用于单点训练样本的采集，用于LR, RF, FCN
    data = []
    landuse_1 = np.where(np.isnan(landuse_1),nodata,landuse_1)  # 确保没有nan，避免出现判断nan==nan的问题
    landuse_2 = np.where(np.isnan(landuse_2),nodata,landuse_2)
    index_change = np.argwhere(landuse_1!=landuse_2)
    
    y_samps = ySampling_byIndex(landuse_2, index_change)
    data.append(y_samps)
    
    # 读协变量数据，并采样
    for i in range(len(xData)):
        print("Sampling drivingFactor {0}".format(i+1))
        x = xData[i]        
        x_samps = ySampling_byIndex(x, index_change)
        data.append(x_samps)
    
    # 删除含有NoData的样本
    flagNan = np.zeros(len(data[0]))
    for i_sample in range(len(data[0])):    # 第i号样本   
        for k_arg in range(len(data)):      # 第k号变量
            if np.isnan(data[k_arg][i_sample]): 
                flagNan[i_sample] = True
                break;
    indexNan = np.argwhere(flagNan==1)
    for k_arg in range(len(data)):      # 第k号变量，包括landuse
        data[k_arg] = np.delete(data[k_arg],indexNan.astype(int),axis=0)
    
    ySamples = data[0]
    xSamples = np.zeros([data[1].shape[0],len(data)-1])   # nSamples,nDrivingFactors
    for k in range(xSamples.shape[1]):
        xSamples[:,k] = data[k+1]
        
    ################# 不同采样方式 ######################
    landuse_1 = np.where(landuse_1==nodata,np.nan,landuse_1)
    n_class = np.nanmax(landuse_1).astype(np.int8)    # 统计土地利用共有几类
    
    N_Pix = ySamples.size   # 样本总数    
    n_Pix_class = np.zeros(n_class)     # 各个类别的样本数
    for i in range(n_class):
        n_Pix_class[i] = np.sum(ySamples==i+1)
    
    if (method == 'uniform'):       # 均匀抽样
        p = np.array([N_Pix/n_Pix for n_Pix in n_Pix_class])
        p_select = np.zeros_like(ySamples)
        for i in range(len(p_select)):
            p_select[i] = p[(ySamples[i]-1).astype(np.int8)]
        p_select = p_select/np.sum(p_select)     # 抽中的概率调整为0-1
        
        index_selected = np.random.choice(ySamples.size, size=int(np.ceil(N_Pix*Rate)), replace=False, p=p_select)
        ySamples = ySamples[index_selected]
        xSamples = xSamples[index_selected,:]
    
    if (method == 'random'):    # 随机抽样
        p_select = np.ones_like(ySamples)
        p_select = p_select/np.sum(p_select)     # 抽中的概率调整为0-1
        index_selected = np.random.choice(ySamples.size, size=int(np.ceil(N_Pix*Rate)), replace=False, p=p_select)
        ySamples = ySamples[index_selected]
        xSamples = xSamples[index_selected,:]
        
    print('Total sample amount without expanding: {0}'.format(ySamples.size))
    print(Counter(ySamples))
    
    return ySamples.astype(np.uint8), xSamples.astype(np.float32)


#======================================================== 采用PEAS的采样 ===================================================

def sampling_2dcnn_PEAS(landuse_1, landuse_2, xData, winSize, Rate = 0.05, nodata=-9999):
    r = int((winSize-1)/2)          # 窗口半径
    data = []
    landuse_1 = np.where(np.isnan(landuse_1),nodata,landuse_1)  # 确保没有nan，避免出现判断nan==nan的问题
    landuse_2 = np.where(np.isnan(landuse_2),nodata,landuse_2)
    index_change = np.argwhere(landuse_1!=landuse_2)        # 变化的像元下标, 大小为 (n*2) 的ndarray
    index_remain = np.argwhere((landuse_1==landuse_2) & (landuse_2!=nodata))    # 不变的像元下标, 大小为 (n*2) 的ndarray
    mapChange = np.where(landuse_1!=landuse_2, landuse_2, nodata)
    mapRemain = np.where(landuse_1==landuse_2, landuse_2, nodata)
    
    n_class = np.max(landuse_1)    # 统计土地利用共有几类
    N_Pix_change = index_change.shape[0]
    N_Pix_remain = index_remain.shape[0]
    
    n_Pix_class_change = np.zeros(n_class)  # 变化像元中各个类别的数量统计
    for i in range(n_class):
        n_Pix_class_change[i] = np.sum(mapChange == i + 1)   
    n_Pix_class_remain = np.zeros(n_class)  # 不变像元中各个类别的数量统计
    for i in range(n_class):
        n_Pix_class_remain[i] = np.sum(mapRemain == i + 1)
        
    p_change = np.array([N_Pix_change/n_Pix for n_Pix in n_Pix_class_change])     # 变化像元中各个地类被选中的概率
    p_remain = np.array([N_Pix_remain/n_Pix for n_Pix in n_Pix_class_remain])     # 不变像元中各个地类被选中的概率
    
    # 计算变化像元中各个位置被选中的概率
    p_select_change = np.zeros(landuse_2.size)
    landuse_change_flat = mapChange.flatten()
    for i in range(len(p_select_change)):
        if landuse_change_flat[i] == nodata:
            p_select_change[i] = 0
        else:
            p_select_change[i] = p_change[(landuse_change_flat[i]-1).astype(np.int8)]
    p_select_change = p_select_change/np.sum(p_select_change)     # 抽中的概率调整为0-1
    
    # 计算不变像元中各个位置被选中的概率
    p_select_remain = np.zeros(landuse_2.size)
    landuse_remain_flat = mapRemain.flatten()
    for i in range(len(p_select_remain)):
        if landuse_remain_flat[i] == nodata:
            p_select_remain[i] = 0
        else:
            p_select_remain[i] = p_remain[(landuse_remain_flat[i]-1).astype(np.int8)]
    p_select_remain = p_select_remain/np.sum(p_select_remain)     # 抽中的概率调整为0-1
    
    index_change_selected = np.random.choice(landuse_2.size, size=int(np.ceil(N_Pix_change*Rate)), replace=False, p=p_select_change)
    index_change_selected = np.array(np.unravel_index(index_change_selected, landuse_2.shape)).T
    index_remain_selected = np.random.choice(landuse_2.size, size=int(np.ceil(N_Pix_remain*Rate)), replace=False, p=p_select_remain)
    index_remain_selected = np.array(np.unravel_index(index_remain_selected, landuse_2.shape)).T
    
    y_samps_change = ySampling_byIndex(landuse_2, index_change_selected)
    y_samps_remain = ySampling_byIndex(landuse_2, index_remain_selected)
    data.append(np.append(y_samps_change, y_samps_remain))
    
    # 读协变量数据，并采样
    for i in range(len(xData)):
        print("Sampling drivingFactor {0}".format(i+1))
        x = xData[i]        
        x = edgeExtension(x,nodata,r)
        x_samps_change = xSampling_byIndex(x, index_change_selected, winSize)
        x_samps_remain = xSampling_byIndex(x, index_remain_selected, winSize)
        data.append(np.append(x_samps_change, x_samps_remain, axis=0))
       
    # 删除含有NoData的样本
    data = dropNoData(data)
    
    ySamples = data[0]
    xSamples = np.zeros([data[1].shape[0],data[1].shape[1],data[1].shape[2],len(data)-1])   # nSamples,winSize,winSize,nDrivingFactors
    for k in range(xSamples.shape[3]):
        xSamples[:,:,:,k] = data[k+1]
        
    print(Counter(ySamples))
    return ySamples.astype(np.uint8), xSamples.astype(np.float32)

def sampling_3dcnn_PEAS(landuse_1, landuse_2, xData, nframes, winSize, Rate = 0.05, nodata=-9999):
    r = int((winSize-1)/2)          # 窗口半径
    nvariables = int(len(xData)/nframes) # 自变量数
    data = []
    landuse_1 = np.where(np.isnan(landuse_1),nodata,landuse_1)  # 确保没有nan，避免出现判断nan==nan的问题
    landuse_2 = np.where(np.isnan(landuse_2),nodata,landuse_2)
    index_change = np.argwhere(landuse_1!=landuse_2)        # 变化的像元下标, 大小为 (n*2) 的ndarray
    index_remain = np.argwhere((landuse_1==landuse_2) & (landuse_2!=nodata))    # 不变的像元下标, 大小为 (n*2) 的ndarray
    mapChange = np.where(landuse_1!=landuse_2, landuse_2, nodata)
    mapRemain = np.where(landuse_1==landuse_2, landuse_2, nodata)
    
    n_class = np.max(landuse_1)    # 统计土地利用共有几类
    N_Pix_change = index_change.shape[0]
    N_Pix_remain = index_remain.shape[0]
    
    n_Pix_class_change = np.zeros(n_class)  # 变化像元中各个类别的数量统计
    for i in range(n_class):
        n_Pix_class_change[i] = np.sum(mapChange == i + 1)   
    n_Pix_class_remain = np.zeros(n_class)  # 不变像元中各个类别的数量统计
    for i in range(n_class):
        n_Pix_class_remain[i] = np.sum(mapRemain == i + 1)
        
    p_change = np.array([N_Pix_change/n_Pix for n_Pix in n_Pix_class_change])     # 变化像元中各个地类被选中的概率
    p_remain = np.array([N_Pix_remain/n_Pix for n_Pix in n_Pix_class_remain])     # 不变像元中各个地类被选中的概率
    
    # 计算变化像元中各个位置被选中的概率
    p_select_change = np.zeros(landuse_2.size)
    landuse_change_flat = mapChange.flatten()
    for i in range(len(p_select_change)):
        if landuse_change_flat[i] == nodata:
            p_select_change[i] = 0
        else:
            p_select_change[i] = p_change[(landuse_change_flat[i]-1).astype(np.int8)]
    p_select_change = p_select_change/np.sum(p_select_change)     # 抽中的概率调整为0-1
    
    # 计算不变像元中各个位置被选中的概率
    p_select_remain = np.zeros(landuse_2.size)
    landuse_remain_flat = mapRemain.flatten()
    for i in range(len(p_select_remain)):
        if landuse_remain_flat[i] == nodata:
            p_select_remain[i] = 0
        else:
            p_select_remain[i] = p_remain[(landuse_remain_flat[i]-1).astype(np.int8)]
    p_select_remain = p_select_remain/np.sum(p_select_remain)     # 抽中的概率调整为0-1
    
    index_change_selected = np.random.choice(landuse_2.size, size=int(np.ceil(N_Pix_change*Rate)), replace=False, p=p_select_change)
    index_change_selected = np.array(np.unravel_index(index_change_selected, landuse_2.shape)).T
    index_remain_selected = np.random.choice(landuse_2.size, size=int(np.ceil(N_Pix_remain*Rate)), replace=False, p=p_select_remain)
    index_remain_selected = np.array(np.unravel_index(index_remain_selected, landuse_2.shape)).T
    
    y_samps_change = ySampling_byIndex(landuse_2, index_change_selected)
    y_samps_remain = ySampling_byIndex(landuse_2, index_remain_selected)
    data.append(np.append(y_samps_change, y_samps_remain))
    
    # 读协变量数据，并采样
    for i in range(len(xData)):
        print("Sampling drivingFactor {0}".format(i+1))
        x = xData[i]        
        x = edgeExtension(x,nodata,r)
        x_samps_change = xSampling_byIndex(x, index_change_selected, winSize)
        x_samps_remain = xSampling_byIndex(x, index_remain_selected, winSize)
        data.append(np.append(x_samps_change, x_samps_remain, axis=0))
       
    # 删除含有NoData的样本
    data = dropNoData(data)
    
    ySamples = data[0]
    xSamples = np.zeros([data[1].shape[0],nframes,data[1].shape[1],data[1].shape[2],nvariables])   # nSamples,nframes,winSize,winSize,nDrivingFactors
    for f in range(nframes):
        for v in range(nvariables):
            xSamples[:,f,:,:,v] = data[f*nvariables+v+1]
 
    print(Counter(ySamples))
    return ySamples.astype(np.uint8), xSamples.astype(np.float32)

def sampling_point_PEAS(landuse_1, landuse_2, xData, Rate = 0.05, nodata=-9999):        # 适用于单点训练样本的采集，用于LR, RF, FCN
    winSize = 1
    r = int((winSize-1)/2)
    data = []
    landuse_1 = np.where(np.isnan(landuse_1),nodata,landuse_1)  # 确保没有nan，避免出现判断nan==nan的问题
    landuse_2 = np.where(np.isnan(landuse_2),nodata,landuse_2)
    index_change = np.argwhere(landuse_1!=landuse_2)        # 变化的像元下标, 大小为 (n*2) 的ndarray
    index_remain = np.argwhere((landuse_1==landuse_2) & (landuse_2!=nodata))    # 不变的像元下标, 大小为 (n*2) 的ndarray
    mapChange = np.where(landuse_1!=landuse_2, landuse_2, nodata)
    mapRemain = np.where(landuse_1==landuse_2, landuse_2, nodata)
    
    n_class = np.max(landuse_1)    # 统计土地利用共有几类
    N_Pix_change = index_change.shape[0]
    N_Pix_remain = index_remain.shape[0]
    
    n_Pix_class_change = np.zeros(n_class)  # 变化像元中各个类别的数量统计
    for i in range(n_class):
        n_Pix_class_change[i] = np.sum(mapChange == i + 1)   
    n_Pix_class_remain = np.zeros(n_class)  # 不变像元中各个类别的数量统计
    for i in range(n_class):
        n_Pix_class_remain[i] = np.sum(mapRemain == i + 1)
        
    p_change = np.array([N_Pix_change/n_Pix for n_Pix in n_Pix_class_change])     # 变化像元中各个地类被选中的概率
    p_remain = np.array([N_Pix_remain/n_Pix for n_Pix in n_Pix_class_remain])     # 不变像元中各个地类被选中的概率
    
    # 计算变化像元中各个位置被选中的概率
    p_select_change = np.zeros(landuse_2.size)
    landuse_change_flat = mapChange.flatten()
    for i in range(len(p_select_change)):
        if landuse_change_flat[i] == nodata:
            p_select_change[i] = 0
        else:
            p_select_change[i] = p_change[(landuse_change_flat[i]-1).astype(np.int8)]
    p_select_change = p_select_change/np.sum(p_select_change)     # 抽中的概率调整为0-1
    
    # 计算不变像元中各个位置被选中的概率
    p_select_remain = np.zeros(landuse_2.size)
    landuse_remain_flat = mapRemain.flatten()
    for i in range(len(p_select_remain)):
        if landuse_remain_flat[i] == nodata:
            p_select_remain[i] = 0
        else:
            p_select_remain[i] = p_remain[(landuse_remain_flat[i]-1).astype(np.int8)]
    p_select_remain = p_select_remain/np.sum(p_select_remain)     # 抽中的概率调整为0-1
    
    index_change_selected = np.random.choice(landuse_2.size, size=int(np.ceil(N_Pix_change*Rate)), replace=False, p=p_select_change)
    index_change_selected = np.array(np.unravel_index(index_change_selected, landuse_2.shape)).T
    index_remain_selected = np.random.choice(landuse_2.size, size=int(np.ceil(N_Pix_remain*Rate)), replace=False, p=p_select_remain)
    index_remain_selected = np.array(np.unravel_index(index_remain_selected, landuse_2.shape)).T
    
    y_samps_change = ySampling_byIndex(landuse_2, index_change_selected)
    y_samps_remain = ySampling_byIndex(landuse_2, index_remain_selected)
    data.append(np.append(y_samps_change, y_samps_remain))
    
    # 读协变量数据，并采样
    for i in range(len(xData)):
        print("Sampling drivingFactor {0}".format(i+1))
        x = xData[i]        
        x = edgeExtension(x,nodata,r)
        x_samps_change = ySampling_byIndex(x, index_change_selected)
        x_samps_remain = ySampling_byIndex(x, index_remain_selected)
        data.append(np.append(x_samps_change, x_samps_remain))
       
    # 删除含有NoData的样本
    flagNan = np.zeros(len(data[0]))
    for i_sample in range(len(data[0])):    # 第i号样本   
        for k_arg in range(len(data)):      # 第k号变量
            if np.isnan(data[k_arg][i_sample]): 
                flagNan[i_sample] = True
                break;
    indexNan = np.argwhere(flagNan==1)
    for k_arg in range(len(data)):      # 第k号变量，包括landuse
        data[k_arg] = np.delete(data[k_arg],indexNan.astype(int),axis=0)
    
    ySamples = data[0]
    xSamples = np.zeros([data[1].shape[0],len(data)-1])   # nSamples,nDrivingFactors
    for k in range(xSamples.shape[1]):
        xSamples[:,k] = data[k+1]    
    print('Original total sample amount: {0}'.format(ySamples.size))
    print(Counter(ySamples))    
    return ySamples.astype(np.uint8), xSamples.astype(np.float32)
    
    
    
# =======================================  废弃函数  ================================================
# def pyramidSpliting(dataList, fold = 1):
#     path_save = r'E:\graduate\LandscapeCA\exp\input\subRegion'
#     if not os.path.exists(path_save):
#         os.makedirs(path_save)
#     m,n = dataList[0].shape
#     divider = 2**fold
#     nSubRegion = divider * divider
#     rowStride = m//divider
#     colStride = n//divider    
#     for i in range(len(dataList)):
#         subRegion_1 = dataList[i][0:rowStride, 0:colStride]
#         subRegion_2 = dataList[i][0:rowStride, colStride:2*colStride]
#         subRegion_3 = dataList[i][rowStride:2*rowStride, 0:colStride]
#         subRegion_4 = dataList[i][rowStride:2*rowStride, colStride:2*colStride]
#         np.save(os.path.join(path_save,'1', str(i)+'.npy'),subRegion_1.astype(np.float32))
#         np.save(os.path.join(path_save,'2', str(i)+'.npy'),subRegion_2.astype(np.float32))
#         np.save(os.path.join(path_save,'3', str(i)+'.npy'),subRegion_3.astype(np.float32))
#         np.save(os.path.join(path_save,'4', str(i)+'.npy'),subRegion_4.astype(np.float32))

# def sampling_2dcnn(yData, xData, winSize, stride, nodata=-9999):     # dataList是一个列表，首个元素是initialMap，其余元素为经过归一化处理的drivingFactors
#     data = [] # 准备将采样结果摞在一起
#     # 读土地利用数据，并采样
#     landuse = yData
#     y_samps = ySampling(landuse, winSize, stride, nodata)
#     data.append(y_samps)
#     print('Training: landuse successfully sampled')

#     # 读协变量数据，并采样
#     for i in range(len(xData)):
#         x = xData[i]
#         print("Sampling drivingFactor {0}".format(i+1))
#         x_samps = xSampling(x, winSize, stride, nodata)
#         data.append(x_samps)

#     # 删除含有NoData的样本
#     data = dropNoData(data)
    
#     ySamples = data[0]
#     xSamples = np.zeros([data[1].shape[0],data[1].shape[1],data[1].shape[2],len(data)-1])   # nSamples,winSize,winSize,nDrivingFactors
#     for k in range(xSamples.shape[3]):
#         xSamples[:,:,:,k] = data[k+1]
#     # for k in range(xSamples.shape[3]):
#     #     xSamples[:,:,:,k] = normalize(xSamples[:,:,:,k])  
#     return ySamples.astype(np.uint8), xSamples.astype(np.float32)


# def sampling_3dcnn(yData, xData, winSize, stride, nframes, nodata = -9999):
#     nvariables = int(len(xData)/nframes) # 自变量数
#     data = [] # 准备将采样结果摞在一起
#     # 读土地利用数据，并采样
#     landuse = yData
#     y_samps = ySampling(landuse, winSize, stride, nodata)
#     data.append(y_samps)
#     print('Training: landuse successfully sampled')

#     # 读协变量数据，并采样
#     for i in range(len(xData)):
#         x = xData[i]
#         print("Sampling drivingFactor {0}".format(i+1))
#         x_samps = xSampling(x, winSize, stride, nodata)
#         data.append(x_samps)

#     # 删除含有NoData的样本
#     data = dropNoData(data)
    
#     ySamples = data[0]
#     xSamples = np.zeros([data[1].shape[0],nframes,data[1].shape[1],data[1].shape[2],nvariables])
#     for f in range(nframes):
#         for v in range(nvariables):
#             xSamples[:,f,:,:,v] = data[f*nvariables+v+1]

#     # for k in range(xSamples.shape[-1]):
#     #     xSamples[:,:,:,:,k] = normalize(xSamples[:,:,:,:,k])
    
#     return ySamples.astype(np.uint8), xSamples.astype(np.float32)
    
    
    
    
    






