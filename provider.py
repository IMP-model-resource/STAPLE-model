# -*- coding: utf-8 -*-
"""
Created on Tue Mar 29 13:38:23 2022

@author: Geng
"""

import os
import sys
import numpy as np
import imageio
import random
import matplotlib.pyplot as plt
import copy
import utils_PG

def load_initialMap():
    nodata = -9999
    #=================================================== 路径参数 ===================================================
    path_land_map = r'E:\graduate\LandscapeCA\exp\input\landuse'
    filename_initialMap = 'CLCD_zhengzhou_2010.tif'
    #===============================================================================================================
    print("Loading initialMap")
    initialMap = imageio.imread(os.path.join(path_land_map,filename_initialMap))
    initialMap = np.where(initialMap==np.max(initialMap),nodata,initialMap)    # 修正tif格式的nodata Value
    return initialMap.astype(np.int32)

def load_groundTruth():
    nodata = -9999
    #=================================================== 路径参数 ===================================================
    path_land_map = r'E:\graduate\LandscapeCA\exp\input\landuse'
    filename_groundTruth = 'CLCD_zhengzhou_2019.tif'
    #===============================================================================================================
    print("Loading groundTruth")
    groundTruth = imageio.imread(os.path.join(path_land_map,filename_groundTruth)).astype(np.int8)
    groundTruth = np.where(groundTruth==np.max(groundTruth),nodata,groundTruth)    # 修正tif格式的nodata Value
    return groundTruth.astype(np.int32)

# def load_result():
#     nodata = -9999
#     #=================================================== 路径参数 ===================================================
#     path_land_map = r'E:\graduate\LandscapeCA\exp\output\SA_module\CAPLE-from2010to2019_noEntropy\simulation'
#     filename_result = 'sml_stcnn_noEntropy.tif'
#     #===============================================================================================================
#     print("Loading groundTruth")
#     result = imageio.imread(os.path.join(path_land_map,filename_result)).astype(np.int8)
#     result = np.where(result==np.max(result),nodata,result)    # 修正tif格式的nodata Value
#     return result.astype(np.int32)

def load_drivingFactors():
    nodata = -9999
    #=================================================== 路径参数 ===================================================
    path_root = r'E:\graduate\LandscapeCA\exp\input\drivingFactors\normal'
    # path_save = r'E:\graduate\LandscapeCA\exp\input\drivingFactors\normal'
    # if not os.path.exists(path_save):
    #     os.makedirs(path_save)
    varnames = [ 
                    'DEM_zhengzhou.npy',                        # 1
                    'Slope_zhengzhou.npy',                      # 2
                    'Precip_zhengzhou_2010.npy',                # 3
                    'NDVI_zhengzhou_2010.npy',                  # 4
                    'GDP_zhengzhou_2010.npy',                   # 5
                    'Pop_zhengzhou_2010.npy',                   # 6     
                    'nightLight_zhengzhou_2010.npy',            # 7
                    'Dis_countyCenter_zhengzhou.npy',           # 8
                    'Dis_townCenter_zhengzhou.npy',             # 9
                    'Dis_trunkRoad_zhengzhou_2010.npy',         # 10
                    'Dis_PrimaryRoad_zhengzhou_2010.npy',       # 11
                    'Dis_secondaryRoad_zhengzhou_2010.npy',     # 12
                    'Dis_tertiaryRoad_zhengzhou_2010.npy',      # 13
                    'Dis_subway_zhengzhou_2010.npy',            # 14 
                    'Dis_railway_zhengzhou_2010.npy',           # 15
                    
                    'DEM_zhengzhou.npy',                        # 1       
                    'Slope_zhengzhou.npy',                      # 2       
                    'Precip_zhengzhou_2012.npy',                # 3       
                    'NDVI_zhengzhou_2012.npy',                  # 4       
                    'GDP_zhengzhou_2012.npy',                   # 5 
                    'Pop_zhengzhou_2012.npy',                   # 6       
                    'nightLight_zhengzhou_2012.npy',            # 7       
                    'Dis_countyCenter_zhengzhou.npy',           # 8
                    'Dis_townCenter_zhengzhou.npy',             # 9       
                    'Dis_trunkRoad_zhengzhou_2012.npy',         # 10 
                    'Dis_PrimaryRoad_zhengzhou_2012.npy',       # 11      
                    'Dis_secondaryRoad_zhengzhou_2012.npy',     # 12
                    'Dis_tertiaryRoad_zhengzhou_2012.npy',      # 13      
                    'Dis_subway_zhengzhou_2012.npy',            # 14      
                    'Dis_railway_zhengzhou_2012.npy',           # 15 
                    
                    'DEM_zhengzhou.npy',                        # 1
                    'Slope_zhengzhou.npy',                      # 2
                    'Precip_zhengzhou_2014.npy',                # 3
                    'NDVI_zhengzhou_2014.npy',                  # 4
                    'GDP_zhengzhou_2014.npy',                   # 5
                    'Pop_zhengzhou_2014.npy',                   # 6     
                    'nightLight_zhengzhou_2014.npy',            # 7
                    'Dis_countyCenter_zhengzhou.npy',           # 8
                    'Dis_townCenter_zhengzhou.npy',             # 9
                    'Dis_trunkRoad_zhengzhou_2014.npy',         # 10
                    'Dis_PrimaryRoad_zhengzhou_2014.npy',       # 11
                    'Dis_secondaryRoad_zhengzhou_2014.npy',     # 12
                    'Dis_tertiaryRoad_zhengzhou_2014.npy',      # 13
                    'Dis_subway_zhengzhou_2014.npy',            # 14 
                    'Dis_railway_zhengzhou_2014.npy',           # 15
                    
                    'DEM_zhengzhou.npy',                        # 1
                    'Slope_zhengzhou.npy',                      # 2
                    'Precip_zhengzhou_2016.npy',                # 3
                    'NDVI_zhengzhou_2016.npy',                  # 4
                    'GDP_zhengzhou_2016.npy',                   # 5
                    'Pop_zhengzhou_2016.npy',                   # 6     
                    'nightLight_zhengzhou_2016.npy',            # 7
                    'Dis_countyCenter_zhengzhou.npy',           # 8
                    'Dis_townCenter_zhengzhou.npy',             # 9
                    'Dis_trunkRoad_zhengzhou_2016.npy',         # 10
                    'Dis_PrimaryRoad_zhengzhou_2016.npy',       # 11
                    'Dis_secondaryRoad_zhengzhou_2016.npy',     # 12
                    'Dis_tertiaryRoad_zhengzhou_2016.npy',      # 13
                    'Dis_subway_zhengzhou_2016.npy',            # 14 
                    'Dis_railway_zhengzhou_2016.npy',           # 15
                    
                    'DEM_zhengzhou.npy',                        # 1
                    'Slope_zhengzhou.npy',                      # 2
                    'Precip_zhengzhou_2018.npy',                # 3
                    'NDVI_zhengzhou_2018.npy',                  # 4
                    'GDP_zhengzhou_2018.npy',                   # 5
                    'Pop_zhengzhou_2018.npy',                   # 6     
                    'nightLight_zhengzhou_2018.npy',            # 7
                    'Dis_countyCenter_zhengzhou.npy',           # 8
                    'Dis_townCenter_zhengzhou.npy',             # 9
                    'Dis_trunkRoad_zhengzhou_2018.npy',         # 10
                    'Dis_PrimaryRoad_zhengzhou_2018.npy',       # 11
                    'Dis_secondaryRoad_zhengzhou_2018.npy',     # 12
                    'Dis_tertiaryRoad_zhengzhou_2018.npy',      # 13
                    'Dis_subway_zhengzhou_2018.npy',            # 14 
                    'Dis_railway_zhengzhou_2018.npy'            # 15                  
                  ]
    #===============================================================================================================
    drivingFactors = []     # 读取驱动变量
    # print("Normalizing driving factor")
    for var in varnames:
        # df = imageio.imread(os.path.join(path_root,var)).astype(np.float32)
        # df = np.where(df==np.min(df),nodata,df)
        df = np.load(os.path.join(path_root,var))
        print("Loading driving factors ({0})".format(var))
        # df_norm = utils_PG.normalizing(df)
        # np.save(os.path.join(path_save, var+'.npy'),df_norm.astype(np.float32))
        drivingFactors.append(df)
    return drivingFactors

def load_protectZone(n_landuse, landuseMap):
    protectZone = np.ones((n_landuse,landuseMap.shape[0],landuseMap.shape[1]))  # 保护区图，每个土地利用类型各一张，1能转换，0不能转换
    #=================================================== 路径参数 ===================================================
    path_protectZone = r'E:\graduate\LandscapeCA\exp\input\zoneProtect'
    # protectZone[5] = imageio.imread(os.path.join(path_protectZone,'wh2013_openWater_100m.tif'))
    #===============================================================================================================
    for i in range(protectZone.shape[0]):
        protectZone[i,:,:] = np.where(np.isnan(n_landuse),np.nan,protectZone[i,:,:])  # 将nodata的位置置为nan
    return protectZone

def load_developZone(n_landuse, landuseMap):
    develepZone = np.zeros((n_landuse,landuseMap.shape[0],landuseMap.shape[1])) # 开发区图，每个土地利用类型各一张，1为开发区
    develepZone[4] = imageio.imread(os.path.join(r'E:\graduate\LandscapeCA\exp\input\zoneDevelop','dvl_urban_zhengzhou.tif'))
    for i in range(develepZone.shape[0]):
        develepZone[i,:,:] = np.where(np.isnan(landuseMap),np.nan,develepZone[i,:,:])  # 将nodata的位置置为nan
    return develepZone

def load_banZone(n_landuse, landuseMap):
    banZone = np.zeros((n_landuse,landuseMap.shape[0],landuseMap.shape[1])) # 禁止开发区图，每个土地利用类型各一张，1为禁止开发区
    # banZone[4] = imageio.imread(os.path.join(r'E:\graduate\LandscapeCA\exp\input\zoneBan','ban_urban_zhengzhou.tif'))
    for i in range(banZone.shape[0]):
        banZone[i,:,:] = np.where(np.isnan(landuseMap),np.nan,banZone[i,:,:])  # 将nodata的位置置为nan
    return banZone
    
def load_metaData():
    #=================================================== 路径参数 ===================================================
    ascFile = r'E:\graduate\LandscapeCA\exp\input\landuse\clcd_zhengzhou_2010.txt'    # 用于提供asc文件的元数据
    #===============================================================================================================
    with open(ascFile, 'r') as f:
        lines = f.readlines()[0:6]
    header = ''
    for line in lines:
        header = header + line
    header = header[0:-1]
    return header

def load_potentialMaps(path_potential):
    # filename_potential = [os.path.join(path_potential,f) for f in os.listdir(path_potential) if f.endswith('.txt')]
    return np.load(os.path.join(path_potential,'potentials.npy'))















