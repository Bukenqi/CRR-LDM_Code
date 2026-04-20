# -*- coding: utf-8 -*-
"""
Created on Fri Jul  5 11:05:17 2024

@author: 59278
"""
import numpy as np
from tqdm import tqdm
import FUNC_read_data as read
from scipy import stats
from scipy.ndimage import uniform_filter
from skimage.metrics import structural_similarity as compare_ssim

def mean_var(arr):
    mean_value = np.mean(arr)
    variance_value = np.var(arr)
    return mean_value,variance_value

def generate_scenes(gen, modis_vars, modis_mask, noise_dim=64, rng_seed=None,
    zero_noise=False, noise_scale=1.0):
    batch_size = modis_vars.shape[0]
    if zero_noise:
        noise = np.zeros((batch_size, noise_dim), dtype=np.float32)
    else:
        prng = np.random.RandomState(rng_seed)
        noise = prng.normal(scale=noise_scale, size=(batch_size, noise_dim))
    scene_gen = gen.predict([noise, modis_vars, modis_mask])
    return scene_gen

def Count_remse(gen,real):
    gen1= np.nan_to_num(gen, nan=-35)
    real1= np.nan_to_num(real, nan=-35)
    squared_diff = (gen1-real1) ** 2
    mean_squared_diff = np.mean(squared_diff,axis=(1,2))
    rmse = np.sqrt(mean_squared_diff)
    return rmse

def Count_FSS(true,pred,cloud_class=None,boundary=None):
    Windows_size=3
    FSS_out= {}
    Trues=read.scale_Reflect(true)[...,0]/2+0.5
    preds=read.scale_Reflect(pred)[...,0]/2+0.5
    print('计算全部的FSS')
    FSS=[]
    for i in tqdm(range(len(preds))):
        S_f = uniform_filter(Trues[i], size=Windows_size, mode="constant", cval=0)
        S_o = uniform_filter(preds[i], size=Windows_size, mode="constant", cval=0)
        FSS.append(1-np.nansum((S_f-S_o)**2)/(np.nansum(S_f**2)+np.nansum(S_o**2)))
    FSS=np.array(FSS)
    FSS_out['FSS']=FSS#[~np.isnan(FSS)]
    
    if cloud_class is not None:
        print('计算不同类型云的FSS')
        class_label=['Cirrus', 'Altostratus', 'Altocumulus', 'Stratus', 'Stratocumulus', 'Cumulus', 'Nimbostratus', 'Deep Convection']
        for name in class_label:
            mask=Class_Mask(cloud_class,name)
            t=Trues*mask
            p=preds*mask
            FSS=[]
            for i in tqdm(range(len(t))):
                S_f = uniform_filter(t[i], size=Windows_size, mode="constant", cval=0)
                S_o = uniform_filter(p[i], size=Windows_size, mode="constant", cval=0)
                FSS.append(1-np.nansum((S_f-S_o)**2)/(np.nansum(S_f**2)+np.nansum(S_o**2)))
            FSS=np.array(FSS)
            FSS_out[name]=FSS#[~np.isnan(FSS)]
    
    if boundary is not None:
        print('计算不同dbz的FSS')
        for mindbz in boundary:
            p = preds*np.where(pred >= mindbz,1,0)
            t = Trues*np.where(true >= mindbz,1,0)
            FSS=[]
            for i in tqdm(range(len(t))):
                S_f = uniform_filter(t[i], size=Windows_size, mode="constant", cval=0)
                S_o = uniform_filter(p[i], size=Windows_size, mode="constant", cval=0)
                FSS.append(1-np.nansum((S_f-S_o)**2)/(np.nansum(S_f**2)+np.nansum(S_o**2)))
            FSS=np.array(FSS)
            FSS_out[f'>={mindbz}']=FSS#[~np.isnan(FSS)]
    return FSS_out

def Evaluation_method(name,pred,true,mindbz=-35,windows=3,axis=(1,2)):
    mask_pred = np.where(pred >= mindbz,1,0)
    mask_true = np.where(true >= mindbz,1,0)
    TP = np.sum((mask_true == 1) & (mask_pred == 1),axis=axis)#实际有云，预测有云
    FN = np.sum((mask_true == 1) & (mask_pred == 0),axis=axis)#实际有云，预测没云
    FP = np.sum((mask_true == 0) & (mask_pred == 1),axis=axis)#实际没云，预测有云
    TN = np.sum((mask_true == 0) & (mask_pred == 0),axis=axis)#实际没云，预测没云
    if name =='TS' or name =='CSI':
        score = TP/(TP+FP+FN)
    elif name == 'POD':
        score = TP/(TP+FN)
    elif name =='FAR':
        score = FP/(TP+FP)
    elif name =='HSS':
        score = 2*(TP*TN-FN*FP)/(FN*FN+FP*FP+2*TP*TN+(FN+FP)*(TP+TN))
    elif name=='FSS':
        S_f = uniform_filter(mask_pred, size=(1,windows,windows), mode="constant", cval=0.0)
        S_o = uniform_filter(mask_true, size=(1,windows,windows), mode="constant", cval=0.0)
        score = 1-np.mean((S_f-S_o)**2,axis=axis)/(np.mean(S_f**2,axis=axis)+np.mean(S_o**2,axis=axis))
    elif name=='SSIM':
        Trues = mask_true*read.scale_Reflect(true)[...,0]/2+0.5
        Preds = mask_pred*read.scale_Reflect(pred)[...,0]/2+0.5
        score = np.empty(len(Trues))
        for i in tqdm(range(len(Trues))):
            score[i]=compare_ssim(Trues[i],Preds[i], win_size=windows, data_range=1)
    return score

def Evaluation_sort(method,pred,true,cloud_scenario=None,boundary=None):
    windows=3
    count_out={}
    if cloud_scenario is not None:
        print(f'计算不同类型云的{method}')
        cloud_class=['Cirrus', 'Altostratus', 'Altocumulus', 'Stratus', 'Stratocumulus', 'Cumulus', 'Nimbostratus', 'Deep Convection']
        for name in cloud_class:
            mask=Class_Mask(cloud_scenario,name)
            #index=np.where(np.sum(mask,axis=(1,2))>0)[0]
            t = np.where(mask==1,true,np.nan)
            #print('t',np.nansum(t,axis=(1,2)))
            p = np.where(mask==1,pred,np.nan)
            #print('p',np.nansum(p,axis=(1,2)))
            count_out[name]=Evaluation_method(method,p,t,windows=windows)#[~np.isnan(score)]
    if boundary is not None:
        print(f'计算不同dbz的{method}')
        for mindbz in boundary:
            count_out[f'>={mindbz}']=Evaluation_method(method,pred,true,mindbz=mindbz,windows=windows)
    print(f'计算整个云体{method}')
    count_out['All']=Evaluation_method(method,pred,true,windows=windows)
    return count_out

def Count_SSIM(true,pred,cloud_class=None,boundary=None):
    Windows_size=3
    SSIM_out= {}
    Trues=read.scale_Reflect(true)[...,0]/2+0.5
    preds=read.scale_Reflect(pred)[...,0]/2+0.5
    print('计算全部的ssim')
    SSIM=[]
    for i in tqdm(range(len(preds))):
        SSIM.append(compare_ssim(Trues[i],preds[i], win_size=Windows_size, data_range=1))
    SSIM = np.array(SSIM)
    SSIM_out['ssim']=SSIM#[~np.isnan(SSIM)]
    
    if cloud_class is not None:
        print('计算不同类型云的ssim')
        class_label=['Cirrus', 'Altostratus', 'Altocumulus', 'Stratus', 'Stratocumulus', 'Cumulus', 'Nimbostratus', 'Deep Convection']
        for name in class_label:
            mask=Class_Mask(cloud_class,name)
            t=Trues*mask
            p=preds*mask
            SSIM=[]
            for i in tqdm(range(len(t))):
                SSIM.append(compare_ssim(t[i],p[i], win_size=Windows_size, data_range=1))
            SSIM = np.array(SSIM)
            SSIM_out[name]=SSIM#[~np.isnan(SSIM)]
    if boundary is not None:
        print('计算不同dbz的ssim')
        for mindbz in boundary:
            p = preds*np.where(pred >= mindbz,1,0)
            t = Trues*np.where(true >= mindbz,1,0)
            SSIM=[]
            for i in tqdm(range(len(t))):
                SSIM.append(compare_ssim(t[i],p[i], win_size=Windows_size, data_range=1))
            SSIM = np.array(SSIM)
            SSIM_out[f'>={mindbz}']=SSIM#[~np.isnan(SSIM)]
    return SSIM_out

def calculate_hdi(data, credible_mass=0.7):
    # 将数据转换为numpy数组
    data = np.array(data).flatten()
    # 计算核密度估计
    kde = stats.gaussian_kde(data)
    # 创建评估范围（数据最小值到最大值，扩展10%）
    data_min, data_max = np.min(data), np.max(data)
    range_expansion = 0.1 * (data_max - data_min)
    x_values = np.linspace(data_min - range_expansion, 
                          data_max + range_expansion, 
                          1000)
    density_values = kde(x_values)
    
    # 找到密度最大值及其对应的x值
    max_density_idx = np.argmax(density_values)
    max_density = density_values[max_density_idx]
    max_value = x_values[max_density_idx]
    
    # 计算最高密度区间(HDI)
    # 首先对密度值进行排序并找到阈值
    sorted_density = np.sort(density_values)[::-1]  # 降序排列
    hdi_height = np.interp(credible_mass, 
                          np.cumsum(sorted_density) / np.sum(sorted_density), 
                          sorted_density)
    # 找到密度大于阈值的x值范围
    above_threshold = density_values >= hdi_height
    indices_above = np.where(above_threshold)[0]
    if len(indices_above) == 0:
        # 如果没有点高于阈值，返回整个范围
        hdi_low, hdi_high = x_values[0], x_values[-1]
    else:
        # 找到连续区域
        intervals = []
        start = indices_above[0]
        for i in range(1, len(indices_above)):
            if indices_above[i] != indices_above[i-1] + 1:
                intervals.append((start, indices_above[i-1]))
                start = indices_above[i]
        intervals.append((start, indices_above[-1]))
        # 选择包含最高密度的区间（通常是第一个）
        if intervals:
            hdi_low = x_values[intervals[0][0]]
            hdi_high = x_values[intervals[0][1]]
        else:
            hdi_low, hdi_high = x_values[0], x_values[-1]
    return max_density,max_value,hdi_low,hdi_high

def iqr(data,percent_star,percent_end):
    quartile1, quartile3 = np.percentile(data, [percent_star,percent_end])
    #iqr = quartile3 - quartile1
    #lower_bound = quartile1 - (1 * iqr)
    #upper_bound = quartile3 + (1 * iqr)
    #return np.where((data < lower_bound) | (data > upper_bound))
    #index=np.where(np.logical_and(quartile1 < data, data < quartile3))[0]
    return np.where(np.logical_and(quartile1 < data, data < quartile3))[0]

def cloudsat_to_var(Reflectivity):
    if Reflectivity.ndim==4:
        Reflect=np.squeeze(Reflectivity, axis=-1)
    else:
        Reflect=Reflectivity
    CPR_Z=10**(Reflect/10)
    CPR_lwp=np.nansum(CPR_Z,axis=1)
    CPR_re = np.nanmax(CPR_Z,axis=1)
    
    mask = np.empty_like(Reflect)
    for i in range(Reflect.shape[1]):
        mask[:, i, :] = 64 - i
    mask=mask*24+24
    Cloudsat_heigh=np.where(~np.isnan(Reflect),mask,0)
    CPR_heigh = np.nanmax(Cloudsat_heigh,axis=1)
    CPR_lwp=np.nan_to_num(CPR_lwp, nan=0)
    CPR_re=np.nan_to_num(CPR_re, nan=0)
    CPR_heigh=np.nan_to_num(CPR_heigh, nan=0)
    return CPR_heigh,CPR_re,CPR_lwp

def Class_Mask(cloud_class,Class_name):
    CLASS={'Cirrus':1,
           'Altostratus':2,
           'Altocumulus':3,
           'Stratus':4,
           'Stratocumulus':5,
           'Cumulus':6,
           'Nimbostratus':7,
           'Deep Convection':8}
    mask = np.where(cloud_class == CLASS[Class_name], 1, 0)
    return mask