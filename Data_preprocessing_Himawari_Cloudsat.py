# -*- coding: utf-8 -*-
"""
Created on Tue Jun 10 16:26:16 2025

@author: 59278
"""

import FUNC_read_data as read
import os
from tqdm import tqdm
import numpy as np
#import geopandas as gpd
import transbigdata as tbd
import xarray as xr
import pandas as pd

#from scipy.signal import convolve
highmin =-10
highmax =1000
sea = 1
land = 1
resolution=2
cloudsat_long=128
Himawari_long=cloudsat_long//resolution
#window=np.ones(Himawari_long)/Himawari_long
years=[2019,2020]
'''
load_path='D:/data'
save_path="C:/Users/59278/Desktop"
save_name='Himawari_cloudsat_128_cloud'
himawari_path='/data/yangl/himawari/2km'
'''
load_path='/public/data'
save_path="/public/home/Xiongqq/Data_Train"
save_name='Himawari_cloudsat_128_cloud_2km'
himawari_path='/data/yangl/himawari/2km'

Himawari_varible=['albedo_01','albedo_02','albedo_03','albedo_04','albedo_05','albedo_06',
                  'tbb_07','tbb_08','tbb_09','tbb_10','tbb_11','tbb_12','tbb_13','tbb_14',
                  'tbb_15','tbb_16','latitude','longitude','SOZ','SOA','SAZ','SAA']
cloudsat_varible=['cloud_scenario','Radar_Reflectivity','land_sea_mask']

class match:
    def __init__(self,resolution=1):
        self.resolution = resolution
        self.Himawari_area={'minlat':-60,'maxlat':60.02,'minlon':80,'maxlon':180}
        self.params = tbd.area_to_params([self.Himawari_area['minlon'],
                                          self.Himawari_area['minlat'],
                                          self.Himawari_area['maxlon'],
                                          self.Himawari_area['maxlat']],
                                         accuracy=1112*resolution)
        #params = tbd.area_to_params([80,-60.02,180,60],accuracy=2224)
    def position(self,c_cood):
        c_cood = c_cood[np.arange((self.resolution//2)-1,len(c_cood)+(self.resolution//2)-1,self.resolution)]
        aindex_h=np.empty(c_cood.shape)
        aindex_h[:,0],aindex_h[:,1] = tbd.GPS_to_grid(c_cood[:,0], c_cood[:,1],self.params)
        #print(np.min(aindex_h),np.max(aindex_h))
        #len(np.unique(aindex_h,axis=0))
        aindex_h[:,0][aindex_h[:,0] < 0] += (360/0.01)/self.resolution
        aindex_h[:,1]=np.abs(aindex_h[:,1]-((self.Himawari_area['maxlat']-self.Himawari_area['minlat'])/0.01)/self.resolution+1)
        return aindex_h.astype(np.int64)
Match=match(2)

'''
def expand_ones(arr):
    index = np.where(arr==1)
    for i in range(25):
        a_r = (index[0], index[1] + i)
        arr[a_r]=1
        a_l = (index[0], index[1] - i)
        arr[a_l]=1
    return arr

file_h="D:/data/Himawari_2km/NC_H08_20160401_0150_R21_FLDK.06001_06001.nc"
file_c="D:/data/2b-geoprof/2016/2016001172904_51489_CS_2B-GEOPROF_GRANULE_P1_R05_E06_F00.hdf"
file_s="D:/data/2b-cldclass/2016/2016001172904_51489_CS_2B-CLDCLASS_GRANULE_P1_R05_E06_F00.hdf"

file_h="D:/data/Himawari_2km/NC_H09_20230930_2350_R21_FLDK.06001_06001.nc"
file_c="D:/data/2b-geoprof/2016/2016001172904_51489_CS_2B-GEOPROF_GRANULE_P1_R05_E06_F00.hdf"
file_s="D:/data/2b-cldclass/2016/2016001172904_51489_CS_2B-CLDCLASS_GRANULE_P1_R05_E06_F00.hdf"
'''

for y in years:
    data={}
    for name in Himawari_varible+cloudsat_varible:
        if name == 'Radar_Reflectivity' or name == 'cloud_scenario':
            data.update({name: np.empty((0, cloudsat_long, 64))})
        else:
            data.update({name: np.empty((0, Himawari_long))})

    file_cloudsat = load_path+'/2b-geoprof/{}'.format(y)
    file_class = load_path+'/2b-cldclass/{}'.format(y)
    
    for cloudsat_name in tqdm(os.listdir(file_cloudsat)):
        #cloudsat_name=os.listdir(file_cloudsat)[-13]
        file_c = file_cloudsat+'/'+cloudsat_name
        file_sp = cloudsat_name.split('_')
        year=int(file_sp[0][:4])
        day=int(file_sp[0][4:7])
        himawari_file=read.date_conversation(year,day)
        if not os.path.isdir(himawari_path+'/'+himawari_file[:6]):
            tqdm.write(f'{himawari_file[:6]}不存在')
            continue
        file_sp[3]='2B-CLDCLASS'
        file_s ='_'.join(file_sp)
        file_s = file_class+'/'+file_s
        try:
            cld_class = read.read_source_data(file_s,'cloud_scenario',dtype=np.int32)
            land_sea = read.read_source_Vdata(file_s,'Navigation_land_sea_flag')
        except:
            tqdm.write(f"{cloudsat_name}:cloud_class数据不匹配")
            continue
        tqdm.write(f'{file_sp[0]}对应{himawari_file}')
        cloudsat_Lon = read.read_source_Vdata(file_c, 'Longitude')
        cloudsat_Lat = read.read_source_Vdata(file_c, 'Latitude')
        cloudsat_varb =np.stack((cloudsat_Lon,cloudsat_Lat), axis=-1)
        cloudsat_index = np.where(((((cloudsat_varb[:,0] >= 80) & (cloudsat_varb[:,0] <= 180))
                                 |((cloudsat_varb[:,0] >= -180)&(cloudsat_varb[:,0] <= -160)))
                                 &(cloudsat_varb[:,1] >= -60) & (cloudsat_varb[:,1] <= 60)))[0]
        if len(cloudsat_index) % 2 != 0:
            cloudsat_index=cloudsat_index[:-1]
        if len(cloudsat_index)<cloudsat_long:
            continue
        '''读取cloudsat相关数据,并根据高度进行筛选'''
        Reflectivity,_ = read.read_cloudsat(file_c)
        Reflectivity = Reflectivity[cloudsat_index,:]
        Reflectivity,C1 = read.Filt_Sample(Reflectivity,cloudsat_long,[0.25,1])
        
        '''读取高度数据,并根据高度进行筛选'''
        cloudsat_High = read.read_source_Vdata(file_c, 'DEM_elevation')[cloudsat_index]
        cloudsat_High_index = np.where((cloudsat_High >= highmin) & (cloudsat_High <= highmax),1,np.nan)
        _,C2 = read.Filt_Sample(cloudsat_High_index,cloudsat_long,[0.9, 1])
        
        '''云分类,并根据高度进行筛选''' 
        cld_class = cld_class[cloudsat_index,38:102]&0b0001100000011110
        cld_class = (cld_class>>1)-1024
        cld_class = read.Slice_Sample(cld_class,cloudsat_long)

        '''陆地海洋'''
        land_sea = np.where(land_sea[cloudsat_index] ==2,sea,land)
        land_sea,C3 = read.Filt_Sample(land_sea,cloudsat_long,[0.8, 1])
        
        '''如果cloudsat，高度，陆地海岸都满足条件'''
        if np.all(np.isnan(C1*C2*C3)):
            #print(np.isnan(C1*C2*C3))
            continue
        index_filter = np.where(~np.isnan(C1*C2*C3))[0]
        
        '''找到符合条件的cloudsat样本'''
        sample_Reflectivity = Reflectivity[index_filter]
        sample_cld_class = cld_class[index_filter]
        sample_land_sea = land_sea[index_filter]
        
        '''匹配Himawari和cloudsat,生成的经纬度样本'''
        cloudsat_position = cloudsat_varb[cloudsat_index]
        Himawari_index = Match.position(cloudsat_position)
        Himawari_index = np.repeat(Himawari_index, 2, axis=0)
        index_cloudsat_Himawari = np.column_stack((cloudsat_index,Himawari_index))
        sample_lon_lat = read.Slice_Sample(index_cloudsat_Himawari,cloudsat_long)[index_filter]
        
        '''匹配Himawari和cloudsat,生成时间样本'''
        time_cloudsat = read.read_time(file_c)[cloudsat_index]
        time_index = read.Slice_Sample(time_cloudsat,cloudsat_long,Array=False)
        sample_time=[]
        for i in index_filter:
            time = read.round_datetime(time_index[i]).strftime('%Y%m%d_%H%M')
            counts = time.value_counts()
            sample_time.append((counts[counts == counts.max()].index)[0])
        Himawari_file = np.array(sample_time)
        #Himawari_file = np.unique(Himawari_file).tolist()
        for file in np.unique(Himawari_file).tolist():
            text = file.split('_')[0]
            index_use = np.where(Himawari_file == file)[0]
            file_h = f'{himawari_path}/{text[:6]}/{text[6:8]}/NC_H08_{file}_R21_FLDK.06001_06001.nc'
            #file_h="D:/data/Test_data/Himawari_2km/NC_H09_20240905_0600_R21_FLDK.06001_06001.nc"
            try:
                varibles_name=read.read_variable_name(file_h)
            except:
                tqdm.write(f'无{file}')
                continue
            if not set(Himawari_varible).issubset(set(varibles_name)):
                continue
            Himawari = xr.open_dataset(file_h)  
            SOZ = Himawari['SOZ'].astype('float64').values
            h_mask = np.where(SOZ>=70,np.nan,1)
            if np.all(np.isnan(h_mask)):
                continue
            SOZ = SOZ*h_mask
            index_use_H = sample_lon_lat[index_use,::2,1:3]
            index_use_H[index_use_H==6001]=6000
            sample_SOZ = SOZ[index_use_H[:,:,1],index_use_H[:,:,0]]
            mask = ~np.isnan(sample_SOZ).all(axis=1)
            if ~mask.all():
                continue
            index_final=np.where(mask)[0]
            #print(np.nanmin(sample_SOZ,axis=1),mask)
            '''
            data['Radar_Reflectivity'] = sample_Reflectivity[index_use][index_final]
            data['cloud_scenario'] = sample_cld_class[index_use][index_final]
            data['land_sea_mask'] = sample_land_sea[index_use][index_final,::2]
            '''
            data['Radar_Reflectivity'] = np.concatenate((data['Radar_Reflectivity'],sample_Reflectivity[index_use][index_final]), axis=0)
            data['cloud_scenario'] = np.concatenate((data['cloud_scenario'],sample_cld_class[index_use][index_final]), axis=0)
            data['land_sea_mask'] = np.concatenate((data['land_sea_mask'],sample_land_sea[index_use][index_final,::2]), axis=0)
            for name_h in Himawari_varible:
                if name_h == 'SOZ':
                    varible = sample_SOZ[index_final]
                else:
                    var_himawari = Himawari[name_h].astype('float64').values
                    if name_h=='longitude':
                        varible = var_himawari[index_use_H[index_final,:,0]]
                    elif name_h=='latitude':
                        varible = var_himawari[index_use_H[index_final,:,1]]
                    else:
                        varible = var_himawari[index_use_H[index_final,:,1],index_use_H[index_final,:,0]]
                data[name_h] = np.concatenate((data[name_h],varible), axis=0)
            Himawari.close()  # 显式关闭文件
            del Himawari      # 删除引用以便垃圾回收
        tqdm.write(f"已完成样本数量为{len(data[Himawari_varible[-1]])}")
        '''保存数据'''
    tqdm.write(f"样本数量为{len(data['longitude'])}")
    read.save_data(data,save_path,f'{y}_'+save_name)