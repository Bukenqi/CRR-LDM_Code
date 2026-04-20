# -*- coding: utf-8 -*-
"""
Created on Tue Mar 17 16:39:51 2026

@author: Admin
"""

import FUNC_read_data as read
import FUNC_plot_image as plot
import FUNC_plot_3Dscene as scene
import numpy as np
import gc

#%%
"""画风云4B真彩卫星云图"""
image_save="C:/Users/Admin/Desktop/RAGASA"
area={'minlon':100,'maxlon':130,
      'minlat':10,'maxlat':30}
PLOT=plot.Geographical(image_save,area,line=90)
path="C:/Users/Admin/Desktop/FY4B_20250924_0215.nc"
# print(path)
B=read.read_source_data(path,'Channel01',dtype=np.float32)
G=read.read_source_data(path,'Channel02',dtype=np.float32)
R=read.read_source_data(path,'Channel03',dtype=np.float32)
RGB=np.dstack((R, G, B))
lon=read.read_source_data(path,'lon',dtype=np.float32)
lat=read.read_source_data(path,'lat',dtype=np.float32)
lon_lat = [np.stack((lon,lat))]
data_name=['FY-4b Real Color 2025-09-24 02:15_RealColor']
PLOT.images([RGB],lon_lat,data_name,'RAGASA_20250924_0215')

#%%
"""3DRF 渲染图的卫星云图"""
typhoon_name="RAGASA"
save_path=f"C:/Users/Admin/Desktop/{typhoon_name}"
file_path="D:/result/3D_RAGASA/Gen_RAGASA_20250924_0220_2km.nc"
# 生成地图纹理的函数
line_spacing=90
#area = {'minlon': 110,'maxlon': 150.02,'minlat': 20,'maxlat': 40.02,'minhigh':480,'maxhigh':15600}
area = {'minlon': 100,'maxlon': 130.02,'minlat': 10,'maxlat': 30.02,'minhigh':480,'maxhigh':15600}
#area = {'minlon': 114,'maxlon': 126,'minlat': 18,'maxlat': 30,'minhigh':480,'maxhigh':15600}
resolution ='high'
#
length=(area['maxlon'] - area['minlon'])/0.01
width=(area['maxlat'] - area['minlat'])/0.01

back_size={'length':length,'width':width}
focal_point={'lon':length/2,'lat':width/2}
#读取数据
reflectivity = read.read_source_data(file_path, 'out', dtype=np.float32)
#创建一个三维体
world_actor = scene.create_world_actor(line_spacing,save_path,area)
# 创建三维体或地球
volume_actor = scene.create_volume_actor(reflectivity)
del reflectivity
gc.collect()
# 创建渲染器和文本
renderer, text_actor = scene.createa_render(volume_actor, world_actor)
del volume_actor
gc.collect()
#确定视角
str_list=file_path.split('/')[-1].split('_')
day = str_list[2]
time = str_list[3]
Cam = scene.CameraManager(save_path,take_snapshot=True)
render_window = scene.setup_window(renderer,back_size,OffScreen=True)
camera_views = [{"name": "angle_view",
                "camera_position": (1501,-3200,2000),
                "focal_point": (1501,1001, 0),
                "view_up": (0, 0.4, 0.92)},
                {"name": "top_view",
                "camera_position": (focal_point['lon'],focal_point['lat'],3750),#(x, y, z)
                "focal_point": (focal_point['lon'],focal_point['lat'],0),
                "view_up": (0, 1, 0)}]
for view in camera_views:
    # 添加文件名后缀确保唯一性
    save_name=f"{typhoon_name}_{view['name']}_{day}_{time} CRR-LDM-Full"
    render_window = Cam.configure_camera(renderer,render_window,view,save_name)
    scene.add_title_to_image(f"{save_path}/{save_name}.png",
                             f"{save_path}/{save_name}.png",
                             #f"3D {typhoon_name} {day[:4]}-{day[4:6]}-{day[6:]} {time[:2]}:{time[2:]} CRR-LDM-Full",
                             f"3D Reflectivity {day[:4]}-{day[4:6]}-{day[6:]} {time[:2]}:{time[2:]} CRR-LDM-Full",
                             title_size=110)