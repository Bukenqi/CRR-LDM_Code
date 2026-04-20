# -*- coding: utf-8 -*-
"""
Created on Tue Mar 17 16:45:00 2026

@author: Admin
"""

import os
import numpy as np
import FUNC_plot_image as plot
import FUNC_read_data as read
from scipy import signal
#file_path="D:/result/3D_RAGASA/Gen_RAGASA_20250924_0220_2km.nc"
file_path="D:/result/3D_RAGASA/IR/Gen_RAGASA_20250924_0220_2km_infrared.nc"
image_save="D:/images"

#%%
"""SWAN的每一层"""
def alpha_lonlat(kernel,layer,save_path,name):
    os.makedirs(save_path, exist_ok=True)
    alpha=np.where(np.isnan(layer),0,1)
    mask=signal.convolve2d(alpha, kernel, mode='same')
    mask[(mask==15) & (mask==20)]=25
    lonlat=lon_lat[:,((mask!=25) & (mask!=0))]
    alpha=np.pad(alpha,pad_width=((0, 201), (0, 0)),mode='constant',constant_values=0)
    #alpha=np.where(alpha==0,0.3,1)
    np.save(f'{save_path}/alpha_{name}.npy', alpha[::-1,:])
    np.save(f'{save_path}/lonlat_{name}.npy', lonlat)
    return lonlat

file="D:/data/SWAN/2025092402/Z_OTHE_RADAMOSAIC_20250924021800.bin.bz2"
save_path="D:/result/RAGASA_SWAN_202509240218"
image_save="D:/images"
save_name='SWAN'
area={'minlon':105,
     'maxlon':120,
     'minlat':15,
     'maxlat':30}
swan=read.read_swan(file)
#swan.save(save_path,save_name,area=area)
dbz,lon,lat,layers,maxdbz,times,time_interval=swan.read(use_area=area)
#dbz[dbz<=0]=np.nan

PLOT=plot.Geographical(image_save+'/RAGASA_SWAN_layers_'+times,area,line=90)
lon_lat = np.stack(np.meshgrid(lon,lat))
layers_list = layers.tolist()
#PLOT.images([dbz[11]],lon_lat,['Layer 6km_color'],f'SWAN{times}_6km',max_min=[0,70])

kernel = np.ones((5,5))
for i in layers_list:
    indes=layers_list.index(i)
    layer=dbz[indes]
    lonlat_indes=alpha_lonlat(kernel,layer,save_path,str(i))
    #data_name=[f'SWAN {times[:4]}-{times[4:6]}-{times[6:8]} {times[8:10]}:{times[10:12]}UTC {i/1000}km_color']
    data_list=[layer,None]
    Lon_Lat=[lon_lat,lonlat_indes]
    data_name=[f'Layer {i/1000}km_sband',f'Layer {i/1000}km_Path']
    PLOT.images(data_list,Lon_Lat,data_name,f'SWAN{times}_{i/1000}km')
    print(f'{i}层已完成画图')

#%%
"""3DRF的每一层"""
area={'minlon':105,
     'maxlon':120,
     'minlat':15,
     'maxlat':30}
filename=file_path.split('/')[-1].split('_')
day = filename[2]
time = filename[3]
PLOT=plot.Geographical(image_save+'/RAGASA_Gen3D_layers_IR_'+day+time,area,line=90)
dbz=read.read_source_data(file_path,"out",dtype=np.float32)
dbz[dbz<-25]=np.nan
lon=np.arange(area['minlon'],area['maxlon'],0.01)
lat=np.arange(area['minlat'],area['maxlat'],0.01)
lon_lat = np.stack(np.meshgrid(lon,lat))
fload=dbz[:,-1500:,500:2000]
npy_path='D:/result/RAGASA_SWAN_202509240218'
swan_layers=np.array([500,1000,1500,2000,2500,3000,3500,4000,4500,5000,5500,6000,7000,8000,9000,10000,12000,14000,15500,17000,19000])
for layer in range(dbz.shape[0]):
    high = (layer*240 + 480)
    idx = np.flatnonzero(np.abs(swan_layers-high) <= 120)
    if idx.size == 1:
        swan_layer=swan_layers[idx][0]
        print(layer,high,swan_layer)
        mask=np.where(fload[layer]<=-10,0.2,1)
        mask[550:850,600:1100]=1
        alpha=np.load(f'{npy_path}/alpha_{swan_layer}.npy')
        alpha=alpha*mask
        lonlat=np.load(f'{npy_path}/lonlat_{swan_layer}.npy')
        data_list=[fload[layer],None]
        lonlat_list=[lon_lat,lonlat]
        data_name=[f'CRR-LDM-IR Layer {high/1000:.2f}km_wband',f'CRR-LDM-IR Layer {high/1000:.2f}km_Path']
        PLOT.images(data_list,lonlat_list,data_name,f'Himawari{day}{time}_{high/1000:.2f}km_IR',alpha=alpha)
    else:
        print(layer,high)
    #data_name=[f'CRR-LDM-IR Gen3D {day[:4]}-{day[4:6]}-{day[6:]} {time[:2]}:{time[2:]}UTC {high:.2f}km_color']
    data_name=[f'Layer {high/1000:.2f}km_wband']
    PLOT.images([fload[layer,:,:]],[lon_lat],data_name,f'Himawari{day}{time}_{high/1000:.2f}km')
    print(f'{high/1000}km已完成画图')
