# -*- coding: utf-8 -*-
"""
Created on Thu Jan  6 15:41:05 2022

@author: Geng
"""

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Conv3D, MaxPooling3D, Dropout, Dense, Flatten
from sklearn.ensemble import RandomForestClassifier
from sklearn import linear_model
from joblib import dump, load

def Create_2dcnn(inputShape, output_nclass):  # 创建3D-CNN架构
    model = Sequential()
    model.add(Conv2D(filters=10, kernel_size=(3,3), padding='same',input_shape=inputShape, activation='relu')) 
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Conv2D(filters=20, kernel_size=(3,3), padding='same',activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Flatten())
    model.add(Dense(100, activation='tanh'))
    model.add(Dropout(0.2))
    model.add(Dense(output_nclass, activation='softmax'))
    # 打印模型
    # print(model.summary())
    return model

def Create_3dcnn(inputShape, output_nclass):  # 创建3D-CNN架构
    model = Sequential()
    model.add(Conv3D(filters=10, kernel_size=(2,3,3), padding='same',input_shape=inputShape, activation='relu')) 
    model.add(MaxPooling3D(pool_size=(2,2,2)))
    model.add(Conv3D(filters=20, kernel_size=(2,3,3), padding='same',activation='relu'))
    model.add(MaxPooling3D(pool_size=(2,2,2)))
    model.add(Flatten())
    model.add(Dense(100, activation='tanh'))
    model.add(Dropout(0.2))
    model.add(Dense(output_nclass, activation='softmax'))
    # 打印模型
    # print(model.summary())
    return model

def Create_lr():
    return linear_model.LogisticRegression(solver='liblinear')

def Create_rf():
    return RandomForestClassifier(n_estimators=5, max_features=8)

def Create_fcn(input_nFeatures, output_nclass):  # 创建FCN架构
    model = Sequential()
    model.add(Dense(20, activation='tanh'))
    model.add(Dropout(0.2))
    model.add(Dense(20, activation='tanh'))
    model.add(Dropout(0.2))
    model.add(Dense(output_nclass, activation='softmax'))
    model.build(input_shape=[None, input_nFeatures])
    # 打印模型
    # print(model.summary())
    return model


    
    