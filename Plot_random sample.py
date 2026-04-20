# -*- coding: utf-8 -*-
"""
Created on Fri Sep 26 09:15:42 2025

@author: 59278
"""

import numpy as np
import FUNC_read_data as read
import FUNC_plot_image as plot

image_save="D:/images"
data_i ="D:/result/GEN_samples/Full_GEN_samples_2020_30.nc"
data_a ="D:/result/GEN_samples/IR_GEN_samples_2020_50.nc"

variables_name=['cloud_scenario','Radar_Reflectivity','Gen_result']
data={}
#globals()[file]={}
for name in variables_name:
    variable=read.read_source_data(data_a,name,dtype=np.float32)
    variable[variable<-30]=np.nan
    data[name]=variable
variable=read.read_source_data(data_i,'Gen_result',dtype=np.float32)
variable[variable<-30]=np.nan
data['Infrared chanel']=variable

#idx = np.random.choice(6401,5, replace=False)
idx = [5117,5022,3832,210,1948]
PLOT=plot.Comparison(image_save,resolution=1.1,samples=5)
data_name=[
           'cloud_class',
           'True_Reflectgrey',
           'CRR-LDM-Full_Reflectgrey',
           'CRR-LDM-IR_Reflectgrey',]
data_list=[data['cloud_scenario'][idx],
           data['Radar_Reflectivity'][idx],
           data['Gen_result'][idx],
           data['Infrared chanel'][idx],]
hight_layer=[None,None,None,None]
PLOT.images(data_list,data_name,hight_layer,save_name='samples')
