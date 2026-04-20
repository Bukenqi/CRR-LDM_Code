# -*- coding: utf-8 -*-
"""
Created on Tue Mar 17 14:42:30 2026

@author: Admin
"""

from tqdm import tqdm
import numpy as np
import FUNC_read_data as read
import FUNC_plot_image as plot
import FUNC_analyse_data as analyse
import properscoring
from scipy import stats

image_save="C:/Users/Admin/Desktop/新建文件夹 (2)"
CRR_LDM_Full ="D:/result/GEN_samples/Full_GEN_samples_2020_30.nc"
CRR_LDM_IR ="D:/result/GEN_samples/IR_GEN_samples_2020_30.nc"
Analyse=plot.Analyses(image_save,size=2)

def class_dbz(method,pred,true,cloud_scenario,boundary):
    windows=3
    count_out={}
    cloud_class=['Cirrus', 'Altostratus', 'Altocumulus', 'Stratus', 'Stratocumulus', 'Cumulus', 'Nimbostratus', 'Deep Convection']
    for name in cloud_class:
        mask=analyse.Class_Mask(cloud_scenario,name)
        #index=np.where(np.sum(mask,axis=(1,2))>0)[0]
        t = np.where(mask==1,true,np.nan)
        #print('t',np.nansum(t,axis=(1,2)))
        p = np.where(mask==1,pred,np.nan)
        #print('p',np.nansum(p,axis=(1,2)))
        class_c={}
        for mindbz in boundary:
            class_c[f'>={mindbz}']=analyse.Evaluation_method(method,p,t,mindbz=mindbz,windows=windows)
        count_out[name]=class_c
    return count_out
#%%
'''读取数据'''
#variables_name=read.read_variable_name(chanel_all)
variables_name=['cloud_scenario','Radar_Reflectivity','Gen_latent','Gen_result']
data={}
#globals()[file]={}
for name in variables_name:
    variable=read.read_source_data(CRR_LDM_Full,name,dtype=np.float32)
    data[name]=variable
    print(name,np.nanmin(data[name]),np.nanmax(data[name]))
del variable,variables_name
data['Gen_latent_Full'] = data.pop('Gen_latent')
data['Gen_Reflectivity_Full'] = data.pop('Gen_result')

variables_name=['Gen_latent','Gen_result']
#globals()[file]={}
for name in variables_name:
    variable=read.read_source_data(CRR_LDM_IR,name,dtype=np.float32)
    data[name]=variable
    print(name,np.nanmin(data[name]),np.nanmax(data[name]))
del variable,variables_name
data['Gen_latent_IR'] = data.pop('Gen_latent')
data['Gen_Reflectivity_IR'] = data.pop('Gen_result')
#%%%
'''计算SSIM'''
data['SSIM_Full'] = analyse.Evaluation_sort('SSIM',data['Radar_Reflectivity'],data['Gen_Reflectivity_Full'])['All']
data['SSIM_IR'] = analyse.Evaluation_sort('SSIM',data['Radar_Reflectivity'],data['Gen_Reflectivity_IR'])['All']
data_list=[data]
data_name=['CRR-LDM-Full','CRR-LDM-IR']
key_name=['SSIM_Full','SSIM_IR']
label={"image_class":"Dens", 
       "x_title":'SSIM',
       "y_title":"Density",
       "image_title":None,}
Analyse.images(data_list,data_name,key_name,label,'SSIM Dens')
del data_list
for key in key_name:
    SSIM_mean,SSIM_var = analyse.mean_var(data[key])
    SSIM_IQR = stats.iqr(data[key])
    SSIM_Kurtosis=stats.kurtosis(data[key])+3
    SSIM_Skewness=stats.skew(data[key])
    SSIM_Q1 = np.percentile(data[key], 25)
    SSIM_Q3 = np.percentile(data[key], 75)
    print(f'{key}','mean','var','Kurtosis','Skewness','Q1','Q3')
    print('SSIM',SSIM_mean,SSIM_var,SSIM_Kurtosis,SSIM_Skewness,SSIM_Q1,SSIM_Q3)

#%%%
'''计算crps'''
for key in ['Full','IR']:
    true = data['Radar_Reflectivity']
    pred = np.full((true.shape[0],true.shape[1],true.shape[2], 30), np.nan)
    for i in tqdm(range(30)):
        reflectivity=read.read_source_data(f"D:/result/{key}_GEN_samples_2020_{i}.nc",'Gen_result',dtype=np.float32)
        #print(np.nanmin(reflectivity),np.nanmax(reflectivity))
        pred[:,:,:,i]=reflectivity
    crps_map = np.full(true.shape, np.nan)
    for i in tqdm(range(len(true))):
        crps_map[i,:,:] = properscoring.crps_ensemble(true[i], pred[i], axis=-1)
    del pred,true
    data[f'CRPS_{key}']=np.nanmean(crps_map,axis=(1,2))

'''统计指标'''
#data['CRPS_a'] = np.nanmean(all_crps,axis=(1,2))
#data['CRPS_i'] = np.nanmean(infrared_crps,axis=(1,2))
print('指标','mean','var','Kurtosis','Skewness','Q1','Q3')
CRPS_mean,CRPS_var = analyse.mean_var(data['CRPS_a'])
#CRPS_IQR = stats.iqr(data['CRPS'])
CRPS_Kurtosis=stats.kurtosis(data['CRPS_a'])+3
CRPS_Skewness=stats.skew(data['CRPS_a'])
CRPS_Q1 = np.percentile(data['CRPS_a'], 25)
CRPS_Q3 = np.percentile(data['CRPS_a'], 75)
print('CRPS_a',CRPS_mean,CRPS_var,CRPS_Kurtosis,CRPS_Skewness,CRPS_Q1,CRPS_Q3)
CRPS_mean,CRPS_var = analyse.mean_var(data['CRPS_i'])
#CRPS_IQR = stats.iqr(data['CRPS'])
CRPS_Kurtosis=stats.kurtosis(data['CRPS_i'])+3
CRPS_Skewness=stats.skew(data['CRPS_i'])
CRPS_Q1 = np.percentile(data['CRPS_i'], 25)
CRPS_Q3 = np.percentile(data['CRPS_i'], 75)
print('CRPS_i',CRPS_mean,CRPS_var,CRPS_Kurtosis,CRPS_Skewness,CRPS_Q1,CRPS_Q3)

data_list=[data]
data_name=['CRR-LDM-Full','CRR-LDM-IR']
key_name=['CRPS_a','CRPS_i']
label={"image_class":"Dens", 
       "x_title":'CRPS',
       "y_title":"Density",
       "image_title":None,}
Analyse.images(data_list,data_name,key_name,label,'CRPS Dens')
del data_list
#%%%
'''计算FSS,TS在不同回波强度密度分布'''
boundary=[-25,-20,-15,-10,-5,0,5,10,15]
FSS_chanel_all = analyse.Evaluation_sort('FSS',data['Radar_Reflectivity'],data['Gen_Reflectivity_Full'],data['cloud_scenario'],boundary)
TS_chanel_all = analyse.Evaluation_sort('TS',data['Radar_Reflectivity'],data['Gen_Reflectivity_Full'],data['cloud_scenario'],boundary)
FSS_chanel_infrared = analyse.Evaluation_sort('FSS',data['Radar_Reflectivity'],data['Gen_Reflectivity_IR'],data['cloud_scenario'],boundary)
TS_chanel_infrared = analyse.Evaluation_sort('TS',data['Radar_Reflectivity'],data['Gen_Reflectivity_IR'],data['cloud_scenario'],boundary)

data_list=[FSS_chanel_all,FSS_chanel_infrared]
data_name=['CRR-LDM-Full','CRR-LDM-IR']
varible_name=['>=-25','>=-20','>=-15','>=-10','>=-5','>=0','>=5','>=10','>=15']
label={"image_class":"Boxs",
       "x_title":"Reflectivity factor thresholds(dbz)",
       "y_title":"FSS",
       "image_title":None,}
Analyse.images(data_list,data_name,varible_name,label,' FSS Boxs2')

data_list=[TS_chanel_all,TS_chanel_infrared]
data_name=['CRR-LDM-Full','CRR-LDM-IR']
varible_name=['>=-25','>=-20','>=-15','>=-10','>=-5','>=0','>=5','>=10','>=15']
label={"image_class":"Boxs",
       "x_title":"Reflectivity factor thresholds(dbz)",
       "y_title":"TS",
       "image_title":None,}
Analyse.images(data_list,data_name,varible_name,label,' TS Boxs2')

#%%
'''计算FSS,TS在不同云类型箱线图'''
FSS_chanel_all_class =class_dbz('FSS',data['Radar_Reflectivity'],data['Gen_Reflectivity_Full'],data['cloud_scenario'],boundary)
FSS_infrared_class =class_dbz('FSS',data['Radar_Reflectivity'],data['Gen_Reflectivity_IR'],data['cloud_scenario'],boundary)
cloud_class=['Nimbostratus', 'Deep Convection']
for name in cloud_class:
    data_list=[FSS_chanel_all_class[name],FSS_infrared_class[name]]
    data_name=['CRR-LDM-Full','CRR-LDM-IR']
    varible_name=['>=-25','>=-20','>=-15','>=-10','>=-5','>=0','>=5','>=10','>=15']
    label={"image_class":"Boxs",
           "x_title":"Reflectivity factor thresholds(dbz)",
           "y_title":"FSS",
           "image_title":name,}
    Analyse.images(data_list,data_name,varible_name,label,name)