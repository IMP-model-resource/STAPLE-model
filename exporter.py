# -*- coding: utf-8 -*-
"""
Created on Thu Mar 31 16:16:16 2022

@author: Geng
"""

import os
import numpy as np
import imageio

def saveSamples(ySamples, xSamples):
    #=========================================== 训练样本的写入位置 ===========================================
    saveSamplePath = r'E:\graduate\LandscapeCA\exp\output\PG_module\3dcnn\Samples_PEAS'
    #=========================================================================================================
    if not os.path.exists(saveSamplePath):
        os.makedirs(saveSamplePath)
    np.save(os.path.join(saveSamplePath, 'ySamples.npy'), ySamples.astype(np.uint8))
    np.save(os.path.join(saveSamplePath, 'xSamples.npy'), xSamples.astype(np.float32))
    
    print('Sampling Complete! Samples saved to '+saveSamplePath)

def saveModel(model):
    #=========================================== 训练样本的写入位置 ===========================================
    saveModelPath = r'E:\graduate\LandscapeCA\exp\output\PG_module\3dcnn\Model_3dcnn_PEAS'
    #=========================================================================================================
    if not os.path.exists(saveModelPath):
        os.makedirs(saveModelPath)
    model.save(saveModelPath, save_format='tf')     # 注意：这里的'path_to_saved_model'不再是模型名称，仅仅是一个文件夹，模型会保存在这个文件夹之下
    print('Model training complete! Trained Neural Network saved to '+saveModelPath)
    
def savePotentials(prob, ascMetaData):
    #=========================================== 训练样本的写入位置 ===========================================
    savePotentialPath = r'E:\graduate\LandscapeCA\exp\output\PG_module\3dcnn\Potential Maps PEAS'
    #=========================================================================================================
    if not os.path.exists(savePotentialPath):
        os.makedirs(savePotentialPath)
    for i in range(prob.shape[0]):
        # imageio.imwrite(os.path.join(savePotentialPath, 'potential_band_'+str(i+1)+'.tif'), prob[i])
        np.savetxt(os.path.join(savePotentialPath, 'potential_band_'+str(i+1)+'.txt'), prob[i], fmt='%f', 
                   delimiter=' ',header=ascMetaData,comments='')
    np.save(os.path.join(savePotentialPath,'potentials.npy'),prob)
    print('Potential Maps Calculation Complete! Potentials saved to '+savePotentialPath)

def saveSimulation(result, ascMetaData, alpha,beta,FoM):
    FoM = round(FoM,4)*100
    #=========================================== 模拟结果的写入位置 ===========================================
    saveSimulationPath = r'E:\graduate\LandscapeCA\exp\output\SA_module\CAPLE-from2010to2019_PEAS\simulation_test'     # 输出结果的保存路径
    #=========================================================================================================
    if not os.path.exists(saveSimulationPath):
        os.makedirs(saveSimulationPath)
    np.savetxt(os.path.join(saveSimulationPath,'v2_simulation_2019_urbDvl=5_a='+str(alpha)+'_b='+str(beta)+'_FoM='+str(FoM)+'.txt'),result,fmt='%d',delimiter=' ',header=ascMetaData,comments='')
    print('Simulation Complete! Simulation saved to '+saveSimulationPath)
    
def saveFoM(mapFoM,ascMetaData, alpha,beta,FoM):
    FoM = round(FoM,4)*100
    #=========================================== FoM各因子的写入位置 ===========================================
    saveFoMPath = r'E:\graduate\LandscapeCA\exp\output\SA_module\CAPLE-from2010to2019_PEAS\mapFoM_test'
    #==========================================================================================================
    if not os.path.exists(saveFoMPath):
        os.makedirs(saveFoMPath)
    np.savetxt(os.path.join(saveFoMPath,'v2_mapFoM_2019_urbDvl=5_a='+str(alpha)+'_b='+str(beta)+'_FoM='+str(FoM)+'.txt'),mapFoM,fmt='%d',delimiter=' ',header=ascMetaData,comments='')

# 查看中间变量  
# def saveL(L,ascMetaData, alpha,beta):
#     #=========================================== FoM各因子的写入位置 ===========================================
#     saveLPath = r'E:\graduate\LandscapeCA\exp\output\SA_module\LANDSCAPE_from2010to2019\simulation\beta\Lmap'
#     #==========================================================================================================
#     if not os.path.exists(saveLPath):
#         os.makedirs(saveLPath)
#     np.savetxt(os.path.join(saveLPath,'Lmap_a='+str(alpha)+'_b='+str(beta)+'.txt'),L,fmt='%f',delimiter=' ',header=ascMetaData,comments='')
    
# def savePL(pl,ascMetaData,  alpha,beta):
#     #=========================================== FoM各因子的写入位置 ===========================================
#     saveLPath = r'E:\graduate\LandscapeCA\exp\output\SA_module\LANDSCAPE_from2010to2019\simulation\beta\Lmap'
#     #==========================================================================================================
#     if not os.path.exists(saveLPath):
#         os.makedirs(saveLPath)
#     np.savetxt(os.path.join(saveLPath,'PLmap_a='+str(alpha)+'_b='+str(beta)+'.txt'),pl,fmt='%f',delimiter=' ',header=ascMetaData,comments='')
    
# def save_developed_selected_Map(developed_selected_Map,ascMetaData):
#     #=========================================== FoM各因子的写入位置 ===========================================
#     savePath = r'E:\graduate\LandscapeCA\exp\output\SA_module\LANDSCAPE\developed_selected_Map'
#     #==========================================================================================================
#     if not os.path.exists(savePath):
#         os.makedirs(savePath)
#     np.savetxt(os.path.join(savePath,'developed_selected_Map_b=1.txt'),developed_selected_Map,fmt='%f',delimiter=' ',header=ascMetaData,comments='')
    


