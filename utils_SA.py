# -*- coding: utf-8 -*-
"""
Created on Wed Jan 12 13:15:07 2022

@author: Geng
"""

import numpy as np
import heapq

def normalization(data):    # 归一化函数
    _range = np.max(data) - np.min(data)
    return (data - np.min(data)) / _range

def potential_onelization(potentialMaps, nodataValue=-9999):  # 调整概率值，让同一位置上发展潜力之和为1
    # 输入为k*m*n的发展潜力矩阵，输出为调整后的k*m*n矩阵
    k,m,n = potentialMaps.shape
    new_potentialMaps = np.full_like(potentialMaps, np.nan)
    for i in range(k):
        new_potentialMaps[i,:,:] = potentialMaps[i,:,:]/np.sum(potentialMaps,axis=0)
    new_potentialMaps = np.where(potentialMaps==nodataValue, nodataValue, new_potentialMaps)
    return new_potentialMaps

def p_onelization(p):   # 调整概率值，让所有种子被选中的概率之和为1
    return p/np.nansum(p)
       
def get_Entropy(potentialMaps): # 根据发展潜力，求各个位置的系统熵
    # 输入为k*m*n的发展潜力ndarray，输出为m*n的发展潜力的系统熵ndarray
    k,m,n = potentialMaps.shape
    epsilon = 1e-5      # 加一个小值，防止出现divide by zero encountered in log
    entropyMap = np.zeros((m,n))
    for i in range(potentialMaps.shape[0]):
        temp = -potentialMaps[i,:,:]*np.log(potentialMaps[i,:,:] + epsilon)
        entropyMap += temp
    return entropyMap

def entropy2probability(entropyMap):    # 将系统熵转化为随机种子的播撒概率p
    # temp = 1 - np.exp(-entropyMap) # 撒在entropy高的地方
    temp = 10**(-entropyMap)   # 撒在entropy低的地方
    p = temp/np.nansum(temp)
    return np.where(np.isnan(p),0,p)

def isProtected(m,n,landuseMap,protectZone): # 判断m,n位置的土地利用能否转出
    k = landuseMap[m,n].astype(np.int8) - 1 # m,n 位置上的土地利用类型所对应的protectZone层号
    if protectZone[k,m,n] == 0:
        return True
    else:
        return False

def get_land_index(landuseMap, nodataValue=-9999):    # 从土地利用图中获取土地类型列表
    land_index = list(np.unique(landuseMap))    
    try:
        land_index.remove(nodataValue)
    except:
        None
    return land_index

def get_area(landuseMap, land_index):
    area = np.zeros(len(land_index))    # 各类土地面积    
    for i in range(len(land_index)):
        temp = np.sum(landuseMap==land_index[i])
        area[i] = temp
    return area.astype(np.int32)

def seeding(landuseMap,n_seed,p):
    p = p_onelization(p)
    # seeds_index_1d = np.random.choice(landuseMap.size, size=n_seed, replace=False, p=p.flatten()) # 撒种子，replace=False：不可重复选择
    seeds_index_1d = np.argsort(p.flatten())[::-1][0:n_seed]
    seeds_index_2d = np.zeros((seeds_index_1d.size,2))
    for i in range(seeds_index_1d.size):   # 将随机种子转回二维坐标
        seeds_index_2d[i,:] = np.unravel_index(seeds_index_1d[i], landuseMap.shape)
    return seeds_index_1d.astype(np.int32), seeds_index_2d.astype(np.int32)

def seeding_even(landuseMap,n_seed):
    p = np.where(np.isnan(landuseMap),0,1)
    p = p_onelization(p)
    seeds_index_1d = np.random.choice(landuseMap.size, size=n_seed, replace=False, p=p.flatten()) # 撒种子，replace=False：不可重复选择
    # seeds_index_1d = np.argsort(p.flatten())[::-1][0:n_seed]
    seeds_index_2d = np.zeros((seeds_index_1d.size,2))
    for i in range(seeds_index_1d.size):   # 将随机种子转回二维坐标
        seeds_index_2d[i,:] = np.unravel_index(seeds_index_1d[i], landuseMap.shape)
    return seeds_index_1d.astype(np.int32), seeds_index_2d.astype(np.int32)

def seed_surviving(seeds_index_1d, seeds_index_2d, potentialMaps, landuseMap_curr, protectZone):
    dead_seeds_index = []   # 一维list，保存死掉的种子在原始种子列表里的序号
    for i in range(seeds_index_2d.shape[0]):
        m,n = seeds_index_2d[i].astype(np.int32)
        maxPotentialType = np.argmax(potentialMaps[:,m,n]) + 1
        if ((landuseMap_curr[m,n]==maxPotentialType) or (isProtected(m,n,landuseMap_curr,protectZone))):  # 死亡条件：1.最大概率对应的类已经是当前类；2.当前类被保护
            dead_seeds_index.append(i)
    seeds_alive_index_1d = np.delete(seeds_index_1d,dead_seeds_index)
    seeds_alive_index_2d = np.delete(seeds_index_2d,dead_seeds_index,axis=0)
    return seeds_alive_index_1d.astype(np.int32), seeds_alive_index_2d.astype(np.int32)

def sortSeedsbyEntropy(seeds_alive_index_1d,seeds_alive_index_2d,entropyMap):
    seedsEntropy = np.zeros(seeds_alive_index_1d.size)
    for i in range(seedsEntropy.size):
        seedsEntropy[i] = entropyMap[seeds_alive_index_2d[i][0],seeds_alive_index_2d[i][1]]
    # sorted_index = np.argsort(-seedsEntropy) # 从大到小的索引值
    sorted_index = np.argsort(seedsEntropy) # 从小到大的索引值
    seeds_alive_index_1d_sorted = seeds_alive_index_1d[sorted_index]
    seeds_alive_index_2d_sorted = seeds_alive_index_2d[sorted_index,:]
    return seeds_alive_index_1d_sorted, seeds_alive_index_2d_sorted

# def updateSeeds(p,seeds_index_2d,winSize=3):
#     new_seeds_index_2d = np.zeros_like(seeds_index_2d)
#     r = int((winSize-1)/2)
#     p_padding = np.zeros([p.shape[0]+2*r,p.shape[1]+2*r])   # 生成补零的矩阵，防止边缘处种子更新出错
#     p_padding[r:-r,r:-r] = p
#     for k in range(seeds_index_2d.shape[0]):
#         i = seeds_index_2d[k,0]
#         j = seeds_index_2d[k,1]
#         i_padding = i + r   # 当前种子在padding概率矩阵上的行列号
#         j_padding = j + r
#         p_win = p_padding[i_padding-r:i_padding+r+1,j_padding-r:j_padding+r+1]  # 获得邻域上被选中的概率
#         p_win = p_onelization(p_win)
#         # index = np.random.choice(p_win.size, p=p_win.flatten()) # 依据p随机更新
#         index = np.argmax(p_win.flatten())
#         index_win = np.array(np.unravel_index(index, p_win.shape)) # 选择新种子，并计算新种子在邻域窗口内的index
#         index_data = np.array([i,j])-np.array([r,r])+index_win  # 计算新种子在原始p矩阵上的index
#         new_seeds_index_2d[k,0] = index_data[0]
#         new_seeds_index_2d[k,1] = index_data[1]
#     return new_seeds_index_2d

def updateSeeds(p,seeds_index_2d,winSize=3):
    dead_seeds_index = []   # 一维list，保存更新后出界的种子在原始种子列表里的序号
    new_seeds_index_2d = np.zeros_like(seeds_index_2d)
    r = int((winSize-1)/2)
    p_padding = np.zeros([p.shape[0]+2*r,p.shape[1]+2*r])   # 生成补零的矩阵，防止边缘处种子更新出错
    p_padding[r:-r,r:-r] = p
    for k in range(seeds_index_2d.shape[0]):
        # print(k)
        i = seeds_index_2d[k,0]
        j = seeds_index_2d[k,1]
        i_padding = i + r   # 当前种子在padding概率矩阵上的行列号
        j_padding = j + r
        p_win = p_padding[i_padding-r:i_padding+r+1,j_padding-r:j_padding+r+1]  # 获得邻域上被选中的概率
        # Entropy On
        p_win = p_onelization(p_win)
        # Entropy Off
        # p_win = p_onelization(p_win + 0.1*np.random.randint(0, 9, size=(p_win.shape[0], p_win.shape[1])))
        p_win = np.where(np.isnan(p_win),0,p_win)
        index = np.argmax(p_win.flatten())
        index_win = np.array(np.unravel_index(index, p_win.shape)) # 选择新种子，并计算新种子在邻域窗口内的index
        # print(index_win)
        index_data = np.array([i,j])-np.array([r,r])+index_win  # 计算新种子在原始p矩阵上的index
        new_seeds_index_2d[k,0] = index_data[0]
        new_seeds_index_2d[k,1] = index_data[1]
        if ( (index_data[0]<0) | (index_data[1]<0) | (index_data[0]>=p.shape[0]) | (index_data[1]>=p.shape[1]) ):
            dead_seeds_index.append(k)
    result = np.delete(new_seeds_index_2d, dead_seeds_index, axis=0)
    print('dead seeds index；{0}'.format(dead_seeds_index))
    return result

def getFoM(initialMap, groundTruth, simulation, nodata=-9999):
    Map_TH = (groundTruth != initialMap) & (simulation != initialMap) & (simulation == groundTruth)
    Map_PH = (groundTruth != initialMap) & (simulation != initialMap) & (simulation != groundTruth) & (~np.isnan(initialMap))
    Map_MS = (groundTruth != initialMap) & (simulation == initialMap)
    Map_FA = (groundTruth == initialMap) & (simulation != initialMap)
    Map_NS = (groundTruth == initialMap) & (simulation == initialMap)

    th = np.sum(Map_TH)
    ph = np.sum(Map_PH)
    ms = np.sum(Map_MS)
    fa = np.sum(Map_FA)
    ns = np.sum(Map_NS)
    FoM = th/(th+ph+ms+fa)
    
    mapFoM = np.full(Map_TH.shape, nodata)
    mapFoM = np.where(Map_TH==1,1,mapFoM)
    mapFoM = np.where(Map_PH==1,2,mapFoM)
    mapFoM = np.where(Map_MS==1,3,mapFoM)
    mapFoM = np.where(Map_FA==1,4,mapFoM)
    mapFoM = np.where(Map_NS==1,5,mapFoM)

    print('True Hits =', th)
    print('Partial Hits =', ph)
    print('Misses =', ms)
    print('False Alarms =', fa)
    print('Null Successes = ',ns)
    print('FoM =', FoM)
    
    return FoM, th, ph, ms, fa, ns, mapFoM
    
    

    
    
    