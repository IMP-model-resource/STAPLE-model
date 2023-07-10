# -*- coding: utf-8 -*-
"""
Created on Tue Nov  2 12:44:42 2021

@author: Geng
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
import gc 
import copy
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
import utils_PG

def normalize(data, nodata=-9999):
    temp = np.where(data==nodata,np.nan,data)
    Max = np.nanmax(temp)
    Min = np.nanmin(temp)
    data_norm = np.where(data==nodata,nodata,(temp-Min)/(Max-Min))
    return data_norm

def ySampling (data,winSize,stride,nodata):
    m,n = np.shape(data)
    # r = int((winSize-1)/2)
    # temp = np.where(data==nodata,np.nan,data) 
    row_start = int((winSize-1)/2)
    col_start = int((winSize-1)/2)
    row_end = int(row_start+(((m-winSize)//stride)*stride))
    col_end = int(col_start+(((n-winSize)//stride)*stride))
    nSamples = int(((row_end-row_start)/stride+1)*((col_end-col_start)/stride+1))    
    y = np.zeros(nSamples)
    ii = 0
    for i in range(row_start,row_end+1,stride):     # 注意是行优先
        for j in range(col_start,col_end+1,stride):
            y[ii] = data[i,j]
            ii = ii + 1     
    return y

def xSampling(data,winSize,stride,nodata):
    m,n = np.shape(data)
    r = int((winSize-1)/2)
    temp = np.where(data==nodata,np.nan,data) # 训练时
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
    for i_sample in range(len(data[0])):    # 第i号样本,i有780369   
        for k_arg in range(1,len(data)):      # 第k号变量,k有9
            if True in np.isnan(data[k_arg][i_sample]): 
                flagNan[i_sample] = True
                break;
            
    indexNan = np.argwhere(flagNan==1)
    for k_arg in range(len(data)):      # 第k号变量,k有9
        data[k_arg] = np.delete(data[k_arg],indexNan.astype(int),axis=0)
    return data

def replaceNan(data): # 输入一个矩阵，把nan值替换为邻域均值
    temp = data
    indexs = np.argwhere(np.isnan(data))
    for index in indexs:
        temp[index[0],index[1]] = np.nanmean(data)
    return temp

def replaceNan_entire(data,nodata=-9999):    # 输入一个矩阵，把nan值替换为全局均值
    temp = np.where(data==nodata,np.nan,data) # 训练时
    temp = np.where(data==nodata,np.nanmean(temp),data)
    return temp    


def predict_2dcnn(xData, model, winSize, nPacking=25000, nodata=-9999): # nPacking一次采样能容纳的最多样本数, xData是一个list，每个元素是一个ndarray
    # 读土地利用数据，并采样
    # landuse = dataList[0]
    # print('Predicting: landuse successfully loaded')
    
    stride = 1                      # 滑动步长
    r = int((winSize-1)/2)          # 窗口半径
    nframes = 1                     # 帧数
    
    n_class = model.get_layer(index=-1).output_shape[-1]
    xList = []
    # 读协变量数据
    for i in range(len(xData)):
        x = utils_PG.edgeExtension(xData[i],nodata,r)
        x = replaceNan_entire(x)        # 遇到边界外有nodata时,用entire的函数不会Runtime Warning
        xList.append(x)
        print('Predicting: drivingFactor {0} successfully loaded'.format(str(i)))
        
    nvariables = int(len(xData)/nframes) # 自变量数
    m,n = np.shape(xList[0])
    r = int((winSize-1)/2)
    row_start = int((winSize-1)/2)
    col_start = int((winSize-1)/2)
    row_end = int(row_start+(((m-winSize)//stride)*stride))
    col_end = int(col_start+(((n-winSize)//stride)*stride))
    nSamples = int(((row_end-row_start)/stride+1)*((col_end-col_start)/stride+1))   # 应该采集的总样本数

    # y_samps = ySampling(landuse, winSize, stride, nodata)
    # while True in (y_samps==nodata):
    #     y_samps[np.argwhere(y_samps==nodata)] = 6
    #     # y_samps[np.argwhere(y_samps==nodata)] = y_samps[np.argwhere(y_samps==nodata)+1]
    # ySamples = y_samps

    pointerList = []
    for i in range(row_start,row_end+1,stride):     # 注意是行优先
        for j in range(col_start,col_end+1,stride):
            pointer = np.ravel_multi_index([i,j],xList[0].shape)
            pointerList.append(pointer)
            
    totalResult = np.zeros([nSamples,n_class])

    kSamples = 0
    while kSamples < nSamples:        
        packData = []
        for x in xList: # 对每一个自变量
            xPack = np.zeros([nPacking,winSize,winSize])
            kPacking = 0         
            while kPacking < nPacking:       
                pointer = np.unravel_index(pointerList[kSamples+kPacking], x.shape)
                i = pointer[0]
                j = pointer[1]
                xPack[kPacking] = x[i-r:i+r+1,j-r:j+r+1]
                kPacking += 1
                if (kSamples+kPacking==nSamples):
                    break
            packData.append(xPack)
            
        #ySamples = y_samps[kSamples:kSamples+kPacking]
        kSamples += kPacking
            
        xSamples = np.zeros([packData[1].shape[0],packData[1].shape[1],packData[1].shape[2],nvariables])
        for v in range(nvariables):
            xSamples[:,:,:,v] = packData[v]
                
        result = model.predict(xSamples)   
        totalResult[kSamples-kPacking:kSamples] = result[0:kPacking]
        print(" %.2f percent --- %d/%d samples processed "%(100*kSamples/nSamples,kSamples,nSamples))
        del xSamples
        del packData
        gc.collect()
        
    prob = np.zeros([n_class,m,n]) 
    prob[:,:,:] = nodata

    result2 = np.zeros([n_class,int((row_end-row_start)/stride+1),int((col_end-col_start)/stride+1)])
    for i in range(result2.shape[0]):
        for j in range(result2.shape[1]):
            for k in range(result2.shape[2]):
                result2[i,j,k] = totalResult[j*result2.shape[2]+k,i]
    prob[:,row_start:row_end+1,col_start:col_end+1] = result2
    
    temp = prob[:,r:-r,r:-r]
    for i in range(prob.shape[0]):
        temp[i] = np.where(xData[0]==nodata,nodata,temp[i])
    
    return temp


def predict_3dcnn(xData, model, winSize, nframes, nPacking=25000, nodata=-9999): # nPacking一次采样能容纳的最多样本数

    # 读土地利用数据，并采样
    # landuse = dataList[0]
    # print('Predicting: Landuse successfully loaded')
    
    r = int((winSize-1)/2)          # 窗口半径
    n_class = model.get_layer(index=-1).output_shape[-1]
    xList = []
    # 读协变量数据
    for i in range(len(xData)):
        x = utils_PG.edgeExtension(xData[i],nodata,r)
        x = replaceNan_entire(x)        # 遇到边界外有nodata时,用entire的函数不会Runtime Warning
        xList.append(x)
        print('Predicting: drivingFactor {0} successfully loaded'.format(str(i)))

    stride = 1                      # 滑动步长
    nvariables = int(len(xData)/nframes) # 自变量数
    m,n = np.shape(xList[0])
    row_start = int((winSize-1)/2)
    col_start = int((winSize-1)/2)
    row_end = int(row_start+(((m-winSize)//stride)*stride))
    col_end = int(col_start+(((n-winSize)//stride)*stride))
    nSamples = int(((row_end-row_start)/stride+1)*((col_end-col_start)/stride+1))   # 应该采集的总样本数

    # y_samps = ySampling(landuse, winSize, stride, nodata)
    # while True in (y_samps==nodata):
    #     y_samps[np.argwhere(y_samps==nodata)] = np.nan
    #     #y_samps[np.argwhere(y_samps==nodata)] = y_samps[np.argwhere(y_samps==nodata)+1]
    # ySamples = y_samps

    pointerList = []
    for i in range(row_start,row_end+1,stride):     # 注意是行优先
        for j in range(col_start,col_end+1,stride):
            pointer = np.ravel_multi_index([i,j],xList[0].shape)
            pointerList.append(pointer)

    totalResult = np.zeros([nSamples,n_class])
            
    kSamples = 0
    while kSamples < nSamples:        
        packData = []
        for x in xList: # 对每一个自变量
            xPack = np.zeros([nPacking,winSize,winSize])
            kPacking = 0         
            while kPacking < nPacking:       
                pointer = np.unravel_index(pointerList[kSamples+kPacking], x.shape)
                i = pointer[0]
                j = pointer[1]
                xPack[kPacking] = x[i-r:i+r+1,j-r:j+r+1]
                kPacking += 1
                if (kSamples+kPacking==nSamples):
                    break
            packData.append(xPack)
            
        #ySamples = y_samps[kSamples:kSamples+kPacking]
        kSamples += kPacking
            
        xSamples = np.zeros([packData[1].shape[0],nframes,packData[1].shape[1],packData[1].shape[2],nvariables])
        for f in range(nframes):
            for v in range(nvariables):
                xSamples[:,f,:,:,v] = packData[f*nvariables+v]
                
        result = model.predict(xSamples)   
        totalResult[kSamples-kPacking:kSamples] = result[0:kPacking]
        print(" %.2f percent --- %d/%d samples processed "%(100*kSamples/nSamples,kSamples,nSamples))
        del xSamples
        del packData
        gc.collect()
        
    prob = np.zeros([n_class,m,n]) 
    prob[:,:,:] = nodata

    result2 = np.zeros([n_class,int((row_end-row_start)/stride+1),int((col_end-col_start)/stride+1)])
    for i in range(result2.shape[0]):
        for j in range(result2.shape[1]):
            for k in range(result2.shape[2]):
                result2[i,j,k] = totalResult[j*result2.shape[2]+k,i]
    prob[:,row_start:row_end+1,col_start:col_end+1] = result2

    # 备份
    # tempProb = copy.deepcopy(prob)

    temp = prob[:,r:-r,r:-r]
    for i in range(prob.shape[0]):
        temp[i] = np.where(xData[0]==nodata,nodata,temp[i])

    return temp

    
def predict_lr(xData, model, nodata=-9999):
    winSize = 1
    stride = 1                      # 滑动步长
    r = int((winSize-1)/2)          # 窗口半径
    m,n = np.shape(xData[0])        # 研究区行列数
    
    nvariables = int(len(xData))    # 自变量数
    xSamples = np.zeros([xData[0].size,nvariables])     # 准备保存采样结果
    # 读协变量数据
    for i in range(len(xData)):
        x = utils_PG.edgeExtension(xData[i],nodata,r)
        x = replaceNan_entire(x)        # 遇到边界外有nodata时,用entire的函数不会Runtime Warning
        xSamples[:,i] = x.flatten()
        print('Predicting: drivingFactor {0} successfully loaded'.format(str(i)))
    
    result = model.predict_proba(xSamples)
    
    n_class = result.shape[1]    # 分类数
    prob = np.zeros([n_class,m,n])
    for i in range(prob.shape[0]):
        prob[i] = result[:,i].reshape(m,n)
        prob[i] = np.where(xData[0]==nodata,nodata,prob[i])
        
    return prob

def predict_rf(xData, model, nodata=-9999):
    winSize = 1
    stride = 1                      # 滑动步长
    r = int((winSize-1)/2)          # 窗口半径
    m,n = np.shape(xData[0])        # 研究区行列数
    
    nvariables = int(len(xData))    # 自变量数
    xSamples = np.zeros([xData[0].size,nvariables])     # 准备保存采样结果
    # 读协变量数据
    for i in range(len(xData)):
        x = utils_PG.edgeExtension(xData[i],nodata,r)
        x = replaceNan_entire(x)        # 遇到边界外有nodata时,用entire的函数不会Runtime Warning
        xSamples[:,i] = x.flatten()
        print('Predicting: drivingFactor {0} successfully loaded'.format(str(i)))
    
    result = model.predict_proba(xSamples)
    
    n_class = result.shape[1]    # 分类数
    prob = np.zeros([n_class,m,n])
    for i in range(prob.shape[0]):
        prob[i] = result[:,i].reshape(m,n)
        prob[i] = np.where(xData[0]==nodata,nodata,prob[i])
        
    return prob

def predict_fcn(xData, model, nodata=-9999):
    
    winSize = 1
    stride = 1                      # 滑动步长
    r = int((winSize-1)/2)          # 窗口半径
    m,n = np.shape(xData[0])        # 研究区行列数
    n_class = model.get_layer(index=-1).output_shape[-1]    # 分类数
    nvariables = int(len(xData))    # 自变量数
    xSamples = np.zeros([xData[0].size,nvariables])     # 准备保存采样结果
    # 读协变量数据
    for i in range(len(xData)):
        x = utils_PG.edgeExtension(xData[i],nodata,r)
        x = replaceNan_entire(x)        # 遇到边界外有nodata时,用entire的函数不会Runtime Warning
        xSamples[:,i] = x.flatten()
        print('Predicting: drivingFactor {0} successfully loaded'.format(str(i)))
    
    result = model.predict(xSamples)
    
    prob = np.zeros([n_class,m,n])
    for i in range(prob.shape[0]):
        prob[i] = result[:,i].reshape(m,n)
        prob[i] = np.where(xData[0]==nodata,nodata,prob[i])
        
    return prob
    
    
    
    
    
    
    