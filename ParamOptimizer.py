# -*- coding: utf-8 -*-
"""
Created on Thu Jan  6 13:18:40 2022

@author: Geng
"""

import numpy as np
import tensorflow as tf
import CreateModel
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D# Axes3D 包含的是实现3d绘图的各种方法
# import keras

def Gridsearch_2dcnn(ySamples, xSamples, n_class, lr_section, batchSize_section):
    valAccuracy_hst = np.zeros([len(lr_section),len(batchSize_section)])
    step = 0
    for i in range(len(lr_section)):
        for j in range(len(batchSize_section)):
            lr = lr_section[i]
            bs = batchSize_section[j]
            step = step + 1
            model = CreateModel.Create_2dcnn(inputShape=xSamples.shape[1:], output_nclass=n_class)
            print('==========searching step %d/%d: lr=%.4f bs=%d=========='%(step,len(lr_section)*len(batchSize_section),lr,bs))
            model.compile(loss='sparse_categorical_crossentropy',
                          optimizer=tf.keras.optimizers.Adam(learning_rate=lr), metrics=['accuracy'])
            history = model.fit(x=xSamples, y=ySamples, validation_split=0.25, epochs=10, batch_size=bs, verbose=2)
            valAccuracy_hst[i,j] = history.history['val_accuracy'][-1]
            del model
    i_best, j_best = np.unravel_index(np.argmax(valAccuracy_hst), valAccuracy_hst.shape)
    lr_best = lr_section[i_best]
    bs_best = batchSize_section[j_best]   
    
    return lr_best, bs_best, valAccuracy_hst


def Gridsearch_3dcnn(ySamples, xSamples, n_class, lr_section, batchSize_section):
    valAccuracy_hst = np.zeros([len(lr_section),len(batchSize_section)])
    step = 0
    for i in range(len(lr_section)):
        for j in range(len(batchSize_section)):
            lr = lr_section[i]
            bs = batchSize_section[j]
            step = step + 1
            model = CreateModel.Create_3dcnn(inputShape=xSamples.shape[1:], output_nclass=n_class)
            print('==========searching step %d/%d: lr=%.4f bs=%d=========='%(step,len(lr_section)*len(batchSize_section),lr,bs))
            model.compile(loss='sparse_categorical_crossentropy',
                          optimizer=tf.keras.optimizers.Adam(learning_rate=lr), metrics=['accuracy'])
            history = model.fit(x=xSamples, y=ySamples, validation_split=0.25, epochs=10, batch_size=bs, verbose=2)
            valAccuracy_hst[i,j] = history.history['val_accuracy'][-1]
            del model
    
    i_best, j_best = np.unravel_index(np.argmax(valAccuracy_hst), valAccuracy_hst.shape)
    lr_best = lr_section[i_best]
    bs_best = batchSize_section[j_best]   
    
    return lr_best, bs_best, valAccuracy_hst

def show(valAccuracy_hst, section_1, section_2):
    figure = plt.figure()   # 新建一个画布   
    ax = Axes3D(figure)     # 新建一个3d绘图对象   
    y = section_1   # 生成 x, y 的坐标集
    x = section_2    
    X, Y = np.meshgrid(x, y)    # 生成网格
    plt.xlabel("learning rate") # 定义x,y 轴名称
    plt.ylabel("batch size")
    ax.plot_surface(Y, X, valAccuracy_hst, cmap="rainbow")  # 设置间隔和颜色
    plt.show()

