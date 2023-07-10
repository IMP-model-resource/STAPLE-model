# -*- coding: utf-8 -*-
"""
Created on Thu Mar 31 13:19:19 2022

@author: Geng
"""

import numpy as np
import utils_SA
import provider
import exporter
import gc
import copy
import os
import imageio
import sys
import time

gc.collect()
ascMetaData = provider.load_metaData()
time_start = time.time()

nodataValue = -9999
demands = np.array([482800, 54350, 8344, 10330, 196277, 7])     # 2019

# demands = np.array([434890, 54835, 7155, 9471, 245751, 6])      # Markov-2030
# demands = np.array([468315, 57283, 7155, 8996, 210430, 1])      # Regression-2030
# demands = np.array([444869, 58856, 6067, 8077, 234311, 0])      # Regression-2050

# demands = np.array([459396,57778,8564,8707,217663,0])           # Regression-2035-redo
# demands = np.array([443324, 58856, 7539, 8078, 234311, 0])      # Regression-2050-redo

transMatrix = np.array([[1,1,1,1,1,0],
                        [1,1,0,0,1,0],
                        [1,1,1,0,1,0],
                        [0,0,1,1,0,0],
                        [0,0,0,0,1,0],
                        [1,0,0,1,1,1]])
# 控制景观格局
r = 0.001  # 种子比例
alpha = 1   # alpha>1为infill, alpha<1为sprawl
beta = 0 # 正为聚集，负为分散
L_weight = [0,0,0,0,1,0]

# scenario 0: alpha = 1    beta = 0
# scenario 1: alpha = 0.1  beta = -0.005
# scenario 2: alpha = 0.1  beta = 0.01
# scenario 3: alpha = 10   beta = 0.01
# scenario 4: alpha = 10   beta = -0.005
path_potential = r'E:\graduate\LandscapeCA\exp\output\PG_module\3dcnn\Potential Maps LEAS' # 潜力面的保存路径，只到文件夹

######################################### 数据准备 #############################################
landuseMap = provider.load_initialMap()
land_index = utils_SA.get_land_index(landuseMap, nodataValue)  
landuseMap = np.where(landuseMap==nodataValue,np.nan,landuseMap.astype(np.int8))

n_landuse = len(land_index)    
potentialMaps = provider.load_potentialMaps(path_potential)

# 正式运行时，限制区和开发区应该由tif文件获得
protectZone = provider.load_protectZone(n_landuse, landuseMap)
developZone = provider.load_developZone(n_landuse, landuseMap)
banZone = provider.load_banZone(n_landuse, landuseMap)

for i in range(developZone.shape[0]):
    potentialMaps[i,:,:] = np.where(developZone[i,:,:]==1,potentialMaps[i,:,:]*5,potentialMaps[i,:,:])
    # for j in range(i+1,potentialMaps.shape[0]):
    #     potentialMaps[j,:,:] = np.where(develepZone[i,:,:]==1,0,potentialMaps[j,:,:])   # 如果某位置是地类i的开发区，则将该位置其它地类潜力置为0
    # 不一定用以上方法，也可以开发区类别置1以后，不影响其它类别的概率，从而在轮盘赌时进行更公平的竞争

# 为控制LUCC斑块的紧凑度，通过放缩发展概率来实现
for i in range(potentialMaps.shape[0]):
    potentialMaps[i,:,:] = np.where(potentialMaps[i,:,:]==nodataValue,nodataValue,potentialMaps[i,:,:] ** alpha)
    
potentialMaps = utils_SA.potential_onelization(potentialMaps)  # 保证各个位置发展潜力之和为1，得到考虑开发区后的潜力图

entropyMap = utils_SA.get_Entropy(potentialMaps)
# Entropy On
p = utils_SA.entropy2probability(entropyMap)    # 随机种子的播撒概率，p为二维矩阵
p = np.where(np.isnan(landuseMap),0,p)  # 确保nodata位置上的p严格为0
# Entopy Off
# p = np.where(np.isnan(landuseMap),0,1)
# p = np.where(np.isnan(landuseMap),0,1/np.sum(p))

# 禁止开发区相应地类概率置0, 放在此处不影响entropyMap计算
for i in range(banZone.shape[0]):
    potentialMaps[i,:,:] = np.where(banZone[i,:,:]==1, 0, potentialMaps[i,:,:])
    
# 为控制LUCC斑块的临近度, 通过控制种子点的选位来实现, 用L函数修饰p矩阵, 让远离既有斑块的p更大 (或更小)
# 大致确定旧斑块位置
if (beta == 0):     # beta=0时无放缩, L(d)=1, 为减小计算量直接给 pl=p
    pl = p
else:
    developed_selected_index_1d = np.random.choice(landuseMap.size, size=int(np.floor(landuseMap.size*0.001)), replace=False, 
                                                   p=utils_SA.p_onelization(np.where(landuseMap==nodataValue,0,1)).flatten())
    deleteIndex = []
    for i in range(len(developed_selected_index_1d)):
        if np.isnan(landuseMap.flatten()[int(developed_selected_index_1d[i])]):
            deleteIndex.append(i)
    developed_selected_index_1d = np.delete(developed_selected_index_1d, deleteIndex)
    developed_selected_index_2d = np.zeros((developed_selected_index_1d.size,2))
    developed_selected_Map = np.where(np.isnan(landuseMap),np.nan,0)
    for i in range(developed_selected_index_1d.size):   # 将选中像元转回二维坐标
        developed_selected_index_2d[i,:] = np.unravel_index(developed_selected_index_1d[i], landuseMap.shape)
        developed_selected_Map[int(developed_selected_index_2d[i,0]),int(developed_selected_index_2d[i,1])] = landuseMap[int(developed_selected_index_2d[i,0]),int(developed_selected_index_2d[i,1])]
    # 放缩
    L = np.zeros_like(p)     # 初始化L(d)函数
    for i in range(developed_selected_index_2d.shape[0]):
        ii = int(developed_selected_index_2d[i,0])
        jj = int(developed_selected_index_2d[i,1])
        iii,jjj = np.mgrid[0:p.shape[0],0:p.shape[1]]
        d = np.abs(iii-ii) + np.abs(jjj-jj)     # 所有点到当前点的距离矩阵
        landType = int(developed_selected_Map[ii,jj])
        # L = L + (L_weight[landType-1] * (2/(1+np.exp(-beta*d)) - 1))
        L = L + (L_weight[landType-1] * ( np.sign(beta)*np.exp(-np.sign(beta)*beta*d)) )
        
    L = np.where(np.isnan(landuseMap), np.nan, L)
    L = 50 * (L-np.nanmin(L))/(np.nanmax(L)-np.nanmin(L))
    L = np.where(np.isnan(landuseMap), 0, L)  
    pl = p * L
    pl = np.where(np.isnan(landuseMap),0,pl)  # 确保nodata位置上的p严格为0
    pl = utils_SA.p_onelization(pl)

# 检查 L 和 pl
# exporter.saveL(np.where(np.isnan(landuseMap), nodataValue, L), ascMetaData, alpha,beta)
# exporter.savePL(np.where(np.isnan(landuseMap),nodataValue, pl), ascMetaData, alpha,beta)


area_initial = utils_SA.get_area(landuseMap, land_index)
totalGap_initial = np.sum(np.absolute(area_initial - demands))
if (totalGap_initial==0):
    reachDemandFlag = 1
else:
    reachDemandFlag = 0

fstFlag = 1
n_seeds_alive = 0

Dt = np.ones(n_landuse) # 根据现实与需求的差距调整发展潜力自适应因子
area_curr = copy.deepcopy(area_initial)
GapArray_t_2 = area_curr - demands

iRound = 0    # 仅用来显式进行到第几轮 
landuseMap_curr = landuseMap 

try:
    while(reachDemandFlag == 0):
                
        n_seed = np.floor(r * np.count_nonzero(~np.isnan(landuseMap))).astype(np.int32) - n_seeds_alive  # 待播撒的随机种子个数  
        if fstFlag==1:
            seeds_index_1d_temp, seeds_index_2d_temp = utils_SA.seeding(landuseMap, n_seed, pl) # 撒种子
            print ("amount of new seeds: {0}".format(seeds_index_1d_temp.size))
            seeds_index_1d = seeds_index_1d_temp
            seeds_index_2d = seeds_index_2d_temp
            fstFlag = 0
        else:
            seeds_index_1d_temp, seeds_index_2d_temp = utils_SA.seeding(landuseMap, n_seed, pl) # 补齐种子
            print ("amount of new seeds: {0}".format(seeds_index_1d_temp.size))
            seeds_index_1d = np.hstack((seeds_index_1d,seeds_index_1d_temp))
            seeds_index_2d = np.vstack((seeds_index_2d,seeds_index_2d_temp))    
        # 更新被选中的概率，已经做过种子的点将不再被选中
        for ix in range(seeds_index_2d.shape[0]):   
            pl[seeds_index_2d[ix][0],seeds_index_2d[ix][1]] = 0
            p[seeds_index_2d[ix][0],seeds_index_2d[ix][1]] = 0
        pl = np.where(np.isnan(landuseMap),0,pl)  # 确保nodata位置上的p严格为0
        p = np.where(np.isnan(landuseMap),0,p)  # 确保nodata位置上的p严格为0
        pl = utils_SA.p_onelization(pl)
        p = utils_SA.p_onelization(p)
        
        GapArray_t_1 = utils_SA.get_area(landuseMap_curr, land_index) - demands
        for k in range(len(Dt)):
            if np.absolute(GapArray_t_1[k]) <= np.absolute(GapArray_t_2[k]) + 5: 
                Dt[k] = Dt[k]
            elif (GapArray_t_2[k] < 0) & (GapArray_t_1[k] < GapArray_t_2[k]):   # gap<0 且 越来越负，应该增大这一类的潜力
                Dt[k] = Dt[k] *(1 + (np.e**(GapArray_t_2[k]/GapArray_t_1[k]) - 1)) 
            elif (GapArray_t_2[k] > 0) & (GapArray_t_2[k] < GapArray_t_1[k]):   # gap>0 且 越来越正，应该减小这一类的潜力
                Dt[k] = Dt[k] *(1 + (np.e**-(GapArray_t_2[k]/GapArray_t_1[k]) - 1)) 
            elif (GapArray_t_2[k] * GapArray_t_1[k]) <= 0:  # Gap正在经过0, 接近收敛
                Dt[k] = 1e-10    # 限制转入
                # p = np.where(landuseMap_curr==k+1,np.min(p[p>0])*1e-100,p) # 限制转出
                protectZone[k,:,:] = 0
                # for ix in range(seeds_index_2d.shape[0]):
                #     if landuseMap_curr[seeds_index_2d[ix][0],seeds_index_2d[ix][1]] == k+1:
                #         np.delete(seeds_index_2d,ix,axis=0)
                # seeds_index_1d = np.ravel_multi_index([seeds_index_2d[:,0],seeds_index_2d[:,1]],dims=landuseMap.shape) 
               
        ################################################### 开启一轮分配 ##############################################
        iRound = iRound + 1
        print("=============== 第{0}轮分配 ===============".format(iRound))
        print ("total gap = {0}".format(np.sum(np.absolute(GapArray_t_1))))
        print ("type gap = {0}".format(GapArray_t_1))
        # 筛选出存活的种子
        seeds_alive_index_1d, seeds_alive_index_2d = utils_SA.seed_surviving(seeds_index_1d, seeds_index_2d, potentialMaps, landuseMap_curr, protectZone)
        n_seeds_alive = seeds_alive_index_1d.size
        # 将种子点按熵从小到大排序
        seeds_alive_index_1d, seeds_alive_index_2d = utils_SA.sortSeedsbyEntropy(seeds_alive_index_1d, seeds_alive_index_2d, entropyMap)

        seedsDropIndex = []  # 等待接收本轮分配被淘汰的种子行列号
        
        for k in range(n_seeds_alive):  # 对当前的第k个种子
            area_curr = utils_SA.get_area(landuseMap_curr, land_index)    # 当前土地面积，向量
            typeGap = area_curr - demands   # 各类面积与需求之差
            totalGap = np.sum(np.absolute(typeGap)) # 当前面积与需求只差，向量
            if totalGap <= totalGap_initial * 0.23: # 检查是否满足了需求
                reachDemandFlag = 1
                break
            i = seeds_alive_index_2d[k][0]  # 当前种子的行号
            j = seeds_alive_index_2d[k][1]  # 当前种子的列号
            roulette_potentials = utils_SA.p_onelization(potentialMaps[:,i,j]*Dt)  # 种子点上各类土地利用的发展潜力，也就是进行轮盘赌时各类土地利用被选中的概率
            if np.isnan(roulette_potentials).any():
                seedsDropIndex.append(k) # 检查当前的轮盘赌竞争概率是否合法（避免种子窗口滑动更新中的bug：在窗口内选中概率p全为0时也会更新到一个新位置，但这个新位置可能会出界）
                continue
            outType = landuseMap_curr[i,j].astype(np.int8)  # 种子点的当前类别，也就是转出类
            inType = np.random.choice(land_index, replace=True, p=roulette_potentials)

            if transMatrix[outType-1,inType-1] == 1:  
                landuseMap_curr[i,j] = inType   # 更新土地利用图
            

        ########################## 一轮种子分配结束 #######################
        seeds_alive_index_1d = np.delete(seeds_alive_index_1d,seedsDropIndex)
        seeds_alive_index_2d = np.delete(seeds_alive_index_2d,seedsDropIndex,axis=0)
        n_seeds_alive = seeds_alive_index_1d.size
        
        # 放弃的方法
        # 为控制LUCC斑块的临近度, 通过控制种子点的选位来实现, 即放缩上一轮活种子附近的播撒概率
        # if (beta == 0):     # beta=0时无放缩, L(d)=1, 为减小计算量直接给 pl=p
        #     pl = p
        # else:
        #     L = np.ones_like(p)     # 初始化L(d)函数
        #     for k in range(seeds_index_2d.shape[0]):
        #         ii = seeds_index_2d[k,0]
        #         jj = seeds_index_2d[k,1]
        #         i,j = np.mgrid[0:p.shape[0],0:p.shape[1]]
        #         d = np.abs(i-ii) + np.abs(j-jj)
        #         L = L * np.power(((2*delta)/(1+np.exp(-beta*d))-delta+1), 1/seeds_index_2d.shape[0])     
                
        #     pl = p * L
        #     pl = np.where(np.isnan(landuseMap),0,pl)  # 确保nodata位置上的p严格为0
        #     pl = utils_SA.p_onelization(pl)
           
        # 对存活的旧种子，更新其位置
        seeds_index_2d = utils_SA.updateSeeds(p,seeds_alive_index_2d,winSize=3)
        seeds_index_1d = np.ravel_multi_index([seeds_index_2d[:,0],seeds_index_2d[:,1]],dims=landuseMap.shape)
        
        GapArray_t_2 = copy.deepcopy(GapArray_t_1)


except:
    print("=====================================")
    print('WARNING: Incomplete allocation!')
    print("=====================================")


finally:
    # 精度检验
    landuseMap = provider.load_initialMap()
    land_index = utils_SA.get_land_index(landuseMap, nodataValue)  
    landuseMap = np.where(landuseMap==nodataValue,np.nan,landuseMap)
    groundTruth = provider.load_groundTruth()
    groundTruth = np.where(groundTruth==nodataValue,np.nan,groundTruth)
    # result = provider.load_result()
    result = np.where(np.isnan(landuseMap),nodataValue,landuseMap_curr)
    FoM, th, ph, ms, fa, ns, FoMs = utils_SA.getFoM(landuseMap, groundTruth, result)

    # 保存结果
    ascMetaData = provider.load_metaData()
    exporter.saveSimulation(result, ascMetaData, alpha,beta,FoM)  
    exporter.saveFoM(FoMs, ascMetaData, alpha,beta,FoM)
    
    
    time_end = time.time()
    print('=================== Program running time: %dmin - %ds ===================' 
          % ((time_end - time_start)//60,(time_end - time_start)-(time_end - time_start)//60*60))     
   
    
    
    
    
    
    