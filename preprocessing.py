# -*- coding: utf-8 -*-
"""
Created on Wed Jun  1 15:15:47 2022

@author: Geng
"""

import os
import sys 
import gc
import imageio
import numpy as np
import utils_PG
import provider

 varnames = [ 
                 # 'DEM_zhengzhou.tif',                        # 1
                 # 'Slope_zhengzhou.tif',                      # 2
                 # 'Precip_zhengzhou_2010.tif',                # 3
                 # 'NDVI_zhengzhou_2010.tif',                  # 4
                 # 'GDP_zhengzhou_2010.tif',                   # 5
                 # 'Pop_zhengzhou_2010.tif',                   # 6     
                 # 'nightLight_zhengzhou_2010.tif',            # 7
                 # 'Dis_countyCenter_zhengzhou.tif',           # 8
                 # 'Dis_townCenter_zhengzhou.tif',             # 9
                 # 'Dis_trunkRoad_zhengzhou_2010.tif',         # 10
                 # 'Dis_PrimaryRoad_zhengzhou_2010.tif',       # 11
                 # 'Dis_secondaryRoad_zhengzhou_2010.tif',     # 12
                 # 'Dis_tertiaryRoad_zhengzhou_2010.tif',      # 13
                 # 'Dis_subway_zhengzhou_2010.tif',            # 14 
                 # 'Dis_railway_zhengzhou_2010.tif',           # 15
                 
                 'DEM_zhengzhou.tif',                        # 1
                 'Slope_zhengzhou.tif',                      # 2
                 'Precip_zhengzhou_2012.tif',                # 3
                 'NDVI_zhengzhou_2012.tif',                  # 4
                 # 'GDP_zhengzhou_2012.tif',                   # 5
                 'Pop_zhengzhou_2012.tif',                   # 6     
                 'nightLight_zhengzhou_2012.tif',            # 7
                 'Dis_countyCenter_zhengzhou.tif',           # 8
                 'Dis_townCenter_zhengzhou.tif',             # 9
                 # 'Dis_trunkRoad_zhengzhou_2012.tif',         # 10
                 'Dis_PrimaryRoad_zhengzhou_2012.tif',       # 11
                 'Dis_secondaryRoad_zhengzhou_2012.tif',     # 12
                 'Dis_tertiaryRoad_zhengzhou_2012.tif',      # 13
                 'Dis_subway_zhengzhou_2012.tif',            # 14 
                 # 'Dis_railway_zhengzhou_2012.tif',           # 15
                 
                 'DEM_zhengzhou.tif',                        # 1
                 'Slope_zhengzhou.tif',                      # 2
                 'Precip_zhengzhou_2014.tif',                # 3
                 'NDVI_zhengzhou_2014.tif',                  # 4
                 # 'GDP_zhengzhou_2014.tif',                   # 5
                 'Pop_zhengzhou_2014.tif',                   # 6     
                 'nightLight_zhengzhou_2014.tif',            # 7
                 'Dis_countyCenter_zhengzhou.tif',           # 8
                 'Dis_townCenter_zhengzhou.tif',             # 9
                 # 'Dis_trunkRoad_zhengzhou_2014.tif',         # 10
                 'Dis_PrimaryRoad_zhengzhou_2014.tif',       # 11
                 'Dis_secondaryRoad_zhengzhou_2014.tif',     # 12
                 'Dis_tertiaryRoad_zhengzhou_2014.tif',      # 13
                 'Dis_subway_zhengzhou_2014.tif',            # 14 
                 # 'Dis_railway_zhengzhou_2014.tif',           # 15
                 
                 'DEM_zhengzhou.tif',                        # 1
                 'Slope_zhengzhou.tif',                      # 2
                 'Precip_zhengzhou_2016.tif',                # 3
                 'NDVI_zhengzhou_2016.tif',                  # 4
                 # 'GDP_zhengzhou_2016.tif',                   # 5
                 'Pop_zhengzhou_2016.tif',                   # 6     
                 'nightLight_zhengzhou_2016.tif',            # 7
                 'Dis_countyCenter_zhengzhou.tif',           # 8
                 'Dis_townCenter_zhengzhou.tif',             # 9
                 # 'Dis_trunkRoad_zhengzhou_2016.tif',         # 10
                 'Dis_PrimaryRoad_zhengzhou_2016.tif',       # 11
                 'Dis_secondaryRoad_zhengzhou_2016.tif',     # 12
                 'Dis_tertiaryRoad_zhengzhou_2016.tif',      # 13
                 'Dis_subway_zhengzhou_2016.tif',            # 14 
                 # 'Dis_railway_zhengzhou_2016.tif',           # 15
                 
                 'DEM_zhengzhou.tif',                        # 1
                 'Slope_zhengzhou.tif',                      # 2
                 'Precip_zhengzhou_2018.tif',                # 3
                 'NDVI_zhengzhou_2018.tif',                  # 4
                 # 'GDP_zhengzhou_2018.tif',                   # 5
                 'Pop_zhengzhou_2018.tif',                   # 6     
                 'nightLight_zhengzhou_2018.tif',            # 7
                 'Dis_countyCenter_zhengzhou.tif',           # 8
                 'Dis_townCenter_zhengzhou.tif',             # 9
                 # 'Dis_trunkRoad_zhengzhou_2018.tif',         # 10
                 'Dis_PrimaryRoad_zhengzhou_2018.tif',       # 11
                 'Dis_secondaryRoad_zhengzhou_2018.tif',     # 12
                 'Dis_tertiaryRoad_zhengzhou_2018.tif',      # 13
                 'Dis_subway_zhengzhou_2018.tif',            # 14 
                 # 'Dis_railway_zhengzhou_2018.tif',           # 15                  
               ]

nodata = -9999
drivingFactors = provider.load_drivingFactors()

print("Normalizing driving factor")
for df in drivingFactors:
    df_normal = utils_PG.normalizing(df, nodata)
    r'E:\graduate\LandscapeCA\exp\input\drivingFactors\normal'