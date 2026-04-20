# -*- coding: utf-8 -*-
"""
Created on Mon Apr  1 13:52:30 2024

@author: 59278
"""
import os
import gc
from mpl_toolkits.basemap import Basemap
import numpy as np
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
#from matplotlib.colors import ListedColormap
from matplotlib import cm, colors
import pandas as pd
from scipy.interpolate import interp1d
'''
def stretch(data):
    data = np.nan_to_num(data* 255) 
    x = [0, 30, 60, 120, 190, 255]
    y = [0, 110, 160, 210, 240, 255]
    interp = interp1d(x, y, bounds_error=False, fill_value=255)
    result = interp(data).astype(int)
    return np.clip(result/255,0,1)

'''
def stretch(data,only_cloud=False):
    x = [0, 30, 60, 120, 190, 255]
    y = [0, 110, 160, 210, 240, 255]
    interp = interp1d(x, y, bounds_error=False, fill_value=255)
    RGB = np.where(data>0.015,data,0)* 255
    RGB=interp(RGB).astype(int)/255
    alpha = np.all(RGB == 0, axis=2)
    alpha = np.where(alpha,0.0,1.0)[..., np.newaxis]
    if only_cloud:
        diff = np.max(RGB, axis=2) - np.min(RGB, axis=2)
        #    阈值：差值小于 0.05（约 12/255）视为灰色。可根据图片情况微调。
        cloud_alpha = np.where(diff < 0.05,1.0,0.0)
        alpha=alpha*cloud_alpha
    RGBA = np.concatenate([RGB, alpha], axis=2)
    return RGBA

'''画2维颜色映射图-带经纬度-地图映射'''
class Geographical:
    def __init__(self,image_save,area,line=20,color='black',Earthmaps=None,EarthBack=None):
        self.image_save = image_save
        os.makedirs(image_save, exist_ok=True)
        self.Earthmaps = Earthmaps
        self.EarthBack = EarthBack
        self.area=area
        self.line=line
        self.color=color
        #self.size=1.4
        self.size=1.2

    def backdrop(self):
        #创建地图投影
        if self.Earthmaps is None:
            m = Basemap(projection='cyl',#等经纬度投影
                        llcrnrlon=self.area['minlon'],   # 左下角经度
                        llcrnrlat=self.area['minlat'],   # 左下角纬度
                        urcrnrlon=self.area['maxlon'],   # 右上角经度
                        urcrnrlat=self.area['maxlat'],   # 右上角纬度
                        resolution='i')
        elif self.Earthmaps == 'Earth':#地球投影
            # 使用全球Orthographic投影
            m = Basemap(projection='ortho',
                        lat_0=(self.area['maxlat']+self.area['minlat'])/2,#中心位置纬度
                        lon_0=(self.area['maxlon']+self.area['minlon'])/2,#中心位置经度
                        resolution='i')
        else:
            print('没有这类投影')
        
        #设置背景图（网格或蓝色大理石）
        if self.EarthBack == None:
            m.fillcontinents(color='lightgray', lake_color='white')#填充大陆，大陆是灰色，湖泊是白色
            m.drawmapboundary(fill_color='white')#填充海洋，海洋是蓝色'lightblue'
        elif self.EarthBack == 'BlueMarble':
            m.bluemarble(scale=0.5)#蓝色大理石背景
        elif self.EarthBack == 'ShadedRelief':
            m.shadedrelief(scale=0.5)# 使用地形阴影背景
        elif self.EarthBack == 'ETOPO':
            m.etopo(scale=0.5)# 使用ETOPO地形背景
        else:
            print('无填充，纯白色背景')
        # 绘制地图国界线和陆地海洋线
        m.drawcountries(color=self.color, linewidth=1)
        m.drawcoastlines(color=self.color, linewidth=1)
        # 添加经纬度线
        if self.Earthmaps == 'Earth':
            m.drawparallels(np.arange(-90, 91, self.line), color=self.color, linewidth=self.size,fontsize=15*self.size)
            m.drawmeridians(np.arange(-180, 181, self.line), color=self.color, linewidth=self.size,fontsize=15*self.size)
        else:
            m.drawparallels(np.arange(-90,91,self.line), labels=[1,0,0,0], color=self.color, linewidth=self.size,fontsize=15*self.size)
            m.drawmeridians(np.arange(-180,181,self.line), labels=[0,0,0,1], color=self.color, linewidth=self.size,fontsize=15*self.size)
        return m
    
    def Path(self,m,lon_grid,lat_grid):
        x, y = m(lon_grid,lat_grid)
        m.scatter(x, y, marker='.', color='red',s=2*self.size,label='Sample Locations')
        return m

    def Geographic_Grid(self,m,lon_grid,lat_grid):
        if len(lon_grid.shape)==1:
            lon_grid, lat_grid = np.meshgrid(lon_grid, lat_grid)
        x, y = m(lon_grid, lat_grid)
        return m,x,y

    def Color(self,m,data,lon_grid,lat_grid,MinMax=None,alpha=None,color=None,revers=False):
        if color is None:
            cmap='rainbow'#viridis
        elif color=='user':
            colors_list = [(0,0,0,0),"cyan","green", "yellow", "orange", "red","purple"]
            #colors_list = [(0, 0, 0, 0),"blue","skyblue","cyan","limegreen","yellow","orange", "red","purple"]
            #colors_list = ["skyblue","cyan","limegreen","lime","darkgreen","yellow","goldenrod","darkorange","red","firebrick","darkred","magenta","darkviolet"]#"mediumpurple"
            #colors_list = [(1, 1, 1, 0),"skyblue","blue","limegreen","lime","darkgreen","yellow","goldenrod","darkorange","red","firebrick","darkred","magenta","darkviolet","mediumpurple"]
            # 创建颜色映射
            cmap = colors.LinearSegmentedColormap.from_list("custom1_white_yellow", colors_list)
            if revers:
                cmap=cmap.reversed()
        else:
            cmap=color
        if MinMax is None:
            MinMax=[None,None]
        m,x,y = self.Geographic_Grid(m,lon_grid,lat_grid)
        cs = m.pcolormesh(x,y,data,cmap=cmap,vmin=MinMax[0],vmax=MinMax[1],shading='auto',alpha=alpha,snap=True)
        # 添加颜色条
        cbar = plt.colorbar(cs, orientation='vertical', pad=0.01 ,aspect=20, shrink=0.82)
        cbar.ax.tick_params(labelsize=20*self.size)
        return m

    def Sband(self,m,data,lon_grid,lat_grid,MinMax=None,alpha=None,color=None,revers=False):
        if color is None:
            colors_list=["#00000000", # 0,透明
                        "#0000FF",  # 5 - 蓝色
                        "#1E90FF",  # 10 - 浅蓝色/道奇蓝
                        "#64E7EB",  # 15 - 浅青色/青蓝色
                        "#00FF00",  # 20 - 绿色
                        "#ADFF2F",  # 25 - 黄绿色
                        "#FFFF00",  # 30 - 黄色
                        "#FFA500",  # 35 - 橙色
                        "#FF8C00",  # 40 - 深橙色
                        "#FF4500",  # 45 - 橙红色
                        "#FF0000",  # 50 - 红色
                        "#FF00FF",  # 55 - 粉红色/洋红色
                        "#8A2BE2",  # 60 - 蓝紫色
                        "#9932CC",  # 65 - 深紫色
                        "#800080"   # 70 - 紫色
                            ]
        else:
            colors_list=color
        if MinMax is None:
            bounds=[0,5,10,15,20,25,30,35,40,45,50,55,60,65,70]
        else:
            bounds=np.linspace(MinMax[0],MinMax[1],len(colors_list))
            
        cmap = colors.ListedColormap(colors_list)
        if revers:
            cmap=cmap.reversed()
        norm = colors.BoundaryNorm(bounds, cmap.N)
        m,x,y = self.Geographic_Grid(m,lon_grid,lat_grid)
        cs = m.pcolormesh(x,y,data,cmap=cmap,norm=norm,shading='auto',alpha=alpha,snap=True,rasterized=True)
        # 添加颜色条
        cbar = plt.colorbar(cs, orientation='vertical', pad=0.01 ,aspect=20, shrink=0.82)
        cbar.ax.tick_params(labelsize=20*self.size)
        return m
        
    def Wband(self,m,data,lon_grid,lat_grid,MinMax=None,alpha=None,color=None,revers=False):
        if color is None:
            '''            
            colors_list = [
                # 深蓝 → 亮蓝（拉开层次）
                "#0B1F8A",  # -25 深蓝（低亮度，稳）
                "#0F2AA0",  # -22
                "#1435B6",  # -20
                "#1B43CB",  # -18
                "#2554DD",  # -16
                "#3A6AE6",  # -14
                "#5481EE",  # -12
                # 蓝 → 青（明显过渡）
                "#6FB8F4",  # -10 蓝青（提高区分度）
                # 青色区（重点优化：拉开差异）
                "#58CFEA",  # -8 青蓝
                "#4FE3DF",  # -6 明青
                "#57F0D5",  # -4 浅青绿
                "#63F7C8",  # -2 青绿
                "#66FAD7",  # 0  保持你原来的基准色
                # 绿色 → 高亮
                "#4AF09C",  # 2  
                "#3DF27A",  # 4
                "#2DF458",  # 6
                "#00FF00",  # 8
                # 黄红区（基本合理，小优化平滑）
                "#ADFF2F",    # 10 黄绿色
                "#FFFF00",    # 12 黄色
                "#FFA500",    # 14 橙色
                "#FF0000",    # 16 红色（保持不动）
                "#FF1A1A",    # 18 稍深红（自然过渡）
                "#CC0000",    # 20 深红（高值端）
            ]
            '''
            colors_list = [
                # 深蓝 → 亮蓝（拉开层次）
                "#0B1F8A",  # -25 深蓝（低亮度，稳）
                "#0B1F8A",  # -22 深蓝（低亮度，稳）
                "#0B1F8A",  # -20 深蓝（低亮度，稳）
                "#0F2AA0",  # -18
                "#1435B6",  # -16
                "#1B43CB",  # -14
                "#2554DD",  # -12
                "#3A6AE6",  # -10
                "#5481EE",  # -8
                # 蓝 → 青（明显过渡）
                "#6FB8F4",  # -6 蓝青（提高区分度）
                # 青色区（重点优化：拉开差异）
                "#58CFEA",  # -4 青蓝
                "#4FE3DF",  # -2 明青
                "#57F0D5",  # 0 浅青绿
                "#63F7C8",  # 2 青绿
                #"#66FAD7",  # 0  保持你原来的基准色
                # 绿色 → 高亮
                "#4AF09C",  # 4  
                #"#3DF27A",  # 4
                "#2DF458",  # 6
                "#00FF00",  # 8
                # 黄红区（基本合理，小优化平滑）
                "#ADFF2F",    # 10 黄绿色
                "#FFFF00",    # 12 黄色
                "#FFA500",    # 14 橙色
                "#FF0000",    # 16 红色（保持不动）
                "#FF1A1A",    # 18 稍深红（自然过渡）
                "#CC0000",    # 20 深红（高值端）
            ]
        else:
            colors_list=color
        if MinMax is None:
            #bounds=[#-25,-22,-18,-12,-8,-4,
            #        -10,-8,-4,0,4,8,10,12,13.7,16,18,20]#
            bounds=[-25,-22,-20,-18,-16,-14,-12,-10,-8,-6,-4,-2,0,2,4,6,8,10,12,13.7,16,18,20]#
        else:
            bounds=np.linspace(MinMax[0],MinMax[1],len(colors_list))
        cmap = colors.ListedColormap(colors_list)
        if revers:
            cmap=cmap.reversed()
        norm = colors.BoundaryNorm(bounds, cmap.N)
        m,x,y = self.Geographic_Grid(m,lon_grid,lat_grid)
        cs = m.pcolormesh(x,y,data,cmap=cmap,norm=norm,shading='auto',alpha=alpha,snap=True)
        # 添加颜色条
        cbar = plt.colorbar(cs, orientation='vertical', pad=0.01 ,aspect=20, shrink=0.82)
        cbar.ax.tick_params(labelsize=20*self.size)
        return m
    
    def Mask(self,m,data,lon_grid,lat_grid,):
        # 创建离散颜色映射
        cmap = colors.ListedColormap([(0.9, 0.9, 0.9, 0.85)])  # 半透明白色
        cmap.set_bad(color='none', alpha=0)  # nan区域完全透明
        m,x,y = self.Geographic_Grid(m,lon_grid,lat_grid)
        m.pcolormesh(x,y,data,cmap=cmap,shading='auto',edgecolors='none')#'flat'
        return m
    '''
    def Truecolor(self,m,data,lon_grid,lat_grid):
        RGB = stretch(data)
        Color = RGB.reshape(-1, 3)
        m,x,y = self.Geographic_Grid(m,lon_grid,lat_grid)
        m.pcolormesh(x, y,RGB,color=Color,shading='nearest', latlon=False)
        return m
    '''
    def Truecolor(self,m,data,lon_grid,lat_grid):
        RGBA = stretch(data)
        m,x,y = self.Geographic_Grid(m,lon_grid,lat_grid)
        m.pcolormesh(x, y,RGBA,shading='nearest',latlon=False)
        return m

    #用lon_grid, lat_grid = np.meshgrid(data['longitude'], data['latitude'])
    def images(self,data_list,lon_lat,data_name,image_name,MinMax=None,alpha=None,color=None,revers=False):
        s=(self.area['maxlat']-self.area['minlat'])/(self.area['maxlon']-self.area['minlon'])
        #fig,ax= plt.figure(nrows=1, ncols=1,figsize=(20, 20*s))
        fig=plt.figure(figsize=(20, 20*s))
        ax = self.backdrop()
        for data,lonlat,name in zip(data_list,lon_lat,data_name):
            if '_Path' in name:
                ax = self.Path(ax,lonlat[0],lonlat[1])
                flag='_Path'
            if '_color' in name:
                ax = self.Color(ax,data,lonlat[0],lonlat[1],MinMax=MinMax,alpha=alpha,color=color,revers=revers)
                flag='_colormap'
            if '_sband' in name:
                ax = self.Sband(ax,data,lonlat[0],lonlat[1],MinMax=MinMax,alpha=alpha,color=color,revers=revers)
                flag='_SBand'
            if '_wband' in name:
                ax = self.Wband(ax,data,lonlat[0],lonlat[1],MinMax=MinMax,alpha=alpha,color=color,revers=revers)
                flag='_WBand'
            if '_mask' in name:
                ax = self.mask(ax,data,lonlat[0],lonlat[1])
            if "_TrueColor" in name:
                ax = self.Truecolor(ax,data,lonlat[0],lonlat[1])
                flag='_TrueColor'
        title = name.rsplit('_', 1)[0]
        plt.title(title, fontsize=30*self.size,fontweight='bold')
        plt.savefig(f'{self.image_save}/{image_name}{flag}.png', bbox_inches='tight', dpi=300, format='png')
        plt.close()  # 关闭图形释放内存
        del fig, ax     # 删除对象引用
        gc.collect()    # 强制垃圾回收
        print(f"{self.image_save}/{image_name}{flag} saved and cleared memory")

class Analyses:
    def __init__(self,image_save,size=1):
        self.image_save = image_save
        self.colors = ['lightpink','skyblue','lightgreen','lightsalmon','khaki','lightcoral',
                       'lightblue','lightsteelblue','thistle','navajowhite']
        os.makedirs(image_save, exist_ok=True)
        self.size = size

    def Boxs(self,ax,df,n_vars,xlabel):
        ax = sns.boxplot(# 数据参数
                         data=df,x='x',y='y',hue='variable',#分组变量
                         #order=xlabel,#指定x轴类别的顺序
                         palette = self.colors[:n_vars],# 样式与颜色
                         saturation=0.9,#颜色饱和度（0-1）
                         fill=True,#是否填充箱体颜色
                         linecolor='black',#箱体线条颜色
                         linewidth=1,#箱体轮廓线宽度
                         width=0.5,#箱体宽度（0-1）
                         gap=0.2,#组内箱体间距（Seaborn ≥0.12）
                         dodge=True,#是否并排显示分组箱体
                         whis=[5, 95],  # 使用5%和95%分位数作为须线端点
                         showmeans=True,#是否显示均值标记
                         meanprops={"marker": "o","markerfacecolor": "red","markeredgecolor": "black","markersize": "8"},
                         showfliers=True,#是否显示离群点
                         fliersize=1,#离群点标记大小
                         flierprops={"marker": "o","markerfacecolor": "gray","markeredgecolor": "black","alpha": 0.5},
                         showbox=True,#是否显示箱体
                         boxprops={"alpha": 0.8},
                         showcaps=True,#是否显示须线端点
                         capprops={"color": "black", "linewidth": 1.5},
                         medianprops={"color": "black", "linewidth": 2},#设置中位线
                         whiskerprops={"color": "black", "linewidth": 1.5},#设置须线
                         # 其他
                         orient='v',#方向："v"垂直箱线图，"h"水平箱线图
                         ax=ax)
        '''设置图例'''
        ax.legend(loc='upper right',bbox_to_anchor=(1.0,1.0),fontsize=17*self.size,frameon=False)#loc='upper left',bbox_to_anchor=(1.13,1)
        #ax.legend(loc='upper left',fontsize=17*self.size,frameon=False)
        ax.grid(axis='y', linestyle='--', alpha=1)
        # 设置x轴刻度标签
        labels = [item.get_text() for item in ax.get_xticklabels()]
        xlabels = [label.replace(' ', '\n') for label in labels]
        ax.set_xticklabels(xlabels)
        #设置绘图区边框线宽
        for spine in ax.spines.values():
            spine.set_linewidth(self.size)
        return ax
    
    def Line(self,ax,df,n_vars,xlabel):
        # 使用 seaborn lineplot 绘制折线图
        ax = sns.lineplot(data=df,x='x',y='y',hue='variable',
            palette=self.colors[:n_vars],linewidth=3*self.size,
            marker='o',markersize=5*self.size,ax=ax)
        # 设置x轴刻度标签
        ax.set_xticks(np.linspace(0,np.max(xlabel), 10+1))
        ax.set_xticklabels(np.linspace(0,np.max(xlabel), 10+1))
        ax.set_xscale('log')
        # 添加网格
        ax.grid(True, linestyle='--', alpha=1,linewidth=0.6*self.size)
        # 添加图例
        ax.legend(loc='best', fontsize=17*self.size, frameon=False)
        #设置绘图区边框线宽
        for spine in ax.spines.values():
            spine.set_linewidth(self.size)
        return ax

    def Dens(self, ax, df, n_vars):
        #print("Unique variables:", df['variable'].unique())
        # 绘制密度图
        ax = sns.kdeplot(data=df, x='y', hue='variable', fill=True, alpha=0.6,
                         palette=self.colors[:n_vars], linewidth=self.size, ax=ax, common_norm=False)
        ax.grid(axis='x', linestyle='--', alpha=1,linewidth=0.6*self.size)
        # 获取唯一变量和调色板
        unique_vars = df['variable'].unique()
        # 排序，以保证顺序一致（如果需要）
        palette=self.colors[:n_vars]
        # 为每个变量创建一个图例句柄（矩形块，表示填充颜色）
        handles = []
        for i, var in enumerate(unique_vars):
            # 创建一个矩形补丁（Patch），颜色为调色板中对应的颜色，并设置透明度
            handle = matplotlib.patches.Patch(color=palette[i], alpha=0.6, label=var)
            handles.append(handle)
        labels = unique_vars.tolist()
        #ax.legend(handles, labels, loc='upper left', bbox_to_anchor=(1.0, 1.0), fontsize=17 * self.size, frameon=False)
        ax.legend(handles, labels, loc='upper left', fontsize=17 * self.size, frameon=False)
        #设置绘图区边框线宽
        for spine in ax.spines.values():
            spine.set_linewidth(self.size)
        return ax

    def Hist(self,ax,df,n_vars):
        number_ticks=6
        ax = sns.histplot(data=df,x='y',hue='variable',bins=number_ticks,
                        element='bars', multiple='dodge',  # 并排显示
                        palette=self.colors[:n_vars],
                        edgecolor='white',linewidth=1,binrange=(0, 3),ax=ax)
        ax.grid(axis='y', linestyle='--', alpha=1)
        ax.legend_.set_title(None)
        ax.set_xticks(ticks=np.linspace(0, 3, number_ticks+1))
        #设置绘图区边框线宽
        for spine in ax.spines.values():
            spine.set_linewidth(self.size)
        return ax
    
    def images(self,data_list,data_name,key_name,label,image_name):
        datas=[]
        if len(data_list)==1:
            for d in data_list:
                row_data = []
                for name,keys in zip(data_name,key_name):
                    for value in d[keys]:
                        row_data.append((name, value))
                datas.append(pd.DataFrame(row_data, columns=['variable','y']))
        else:
            for d,n in zip(data_list,data_name):
                name=n.split('_')[0]
                row_data = []
                for keys in key_name:
                    for value in d[keys]:
                        row_data.append((name, keys, value))
                datas.append(pd.DataFrame(row_data, columns=['variable', 'x', 'y']))
                '''
                datas.append(pd.DataFrame([(name,keys,value) for keys in key_name for value in d[keys]],
                                          columns=['variable', 'x', 'y']))
                '''
        df = pd.concat(datas).reset_index(drop=True)
        n_vars = len(df['variable'].unique())
        fig, ax = plt.subplots(figsize=(16, 9))
        sns.set_context("notebook", font_scale=1.7)
        if label["image_class"] == 'Boxs':
            ax = self.Boxs(ax,df,n_vars,key_name)
        elif label["image_class"] == 'Line':
            ax = self.Line(ax,df,n_vars,key_name)
        elif label["image_class"] == 'Dens':
            ax = self.Dens(ax,df,n_vars)
        elif label["image_class"] == 'Hist':
            ax = self.Hist(ax,df,n_vars)
        #统一设置x轴和y轴的刻度字体大小
        ax.tick_params(axis='both', labelsize=17*self.size, width=5*self.size)
        #设置标题
        ax.set_title(label["image_title"], fontsize=25*self.size)  # 修正这里
        ax.set_xlabel(label["x_title"], fontsize=20*self.size)     # 修正这里
        ax.set_ylabel(label["y_title"], fontsize=20*self.size)       # 修正这里  
        plt.tight_layout()
        plt.savefig(f"{self.image_save}/{image_name}.png", bbox_inches='tight', dpi=300, format='png')
        plt.close()  # 关闭图形释放内存
        del fig, ax     # 删除对象引用
        gc.collect()    # 强制垃圾回收
        print(f"Saved {image_name} and cleared memory")

def plot_overlop(result,real,title,image_name,image_save):
    gen_mask = np.where(~np.isnan(result), 1, 2)
    real_mask = np.where(~np.isnan(real), 10,20)
    overlop=(gen_mask+real_mask).astype(float)
    overlop[overlop==22]= np.nan
    overlop[overlop==11]= 1
    overlop[overlop==12]= 2
    overlop[overlop==21]= 3
    # 创建一个包含16个子图的图形，2行8列的布局
    fig, axes = plt.subplots(nrows=1, ncols=10, figsize=(31,3))
    # 定义三色映射的颜色
    colors = ['red','darkgreen','blue']
    cmap = matplotlib.colors.ListedColormap(colors)
    # 遍历每个子图并绘制色图
    for i, ax in enumerate(axes.flat):
        ax.imshow(overlop[i],extent=[0, 64, 0, 64],cmap=cmap)
        # 设置横轴刻度
        ax.set_xticks(np.linspace(0, 64, 3))
        ax.set_xticklabels([f'{int(tick)}km' for tick in ax.get_xticks()])
        # 设置纵轴刻度
        if i ==0:
            # 在每行的最左端显示一个纵轴
            ax.yaxis.set_ticks_position('left')
            ax.yaxis.set_label_position('left')
            #ax.yaxis.set_title()
            ax.set_yticks(np.linspace(0, 64, 6))
            ax.set_yticklabels([f'{(tick * 250 + 900) / 1000:.1f}km' for tick in ax.get_yticks()])
        else:
            ax.set_yticks([])  # 隐藏其他行的纵轴刻度 
    # 在左侧添加一个标题
    axes[0].set_ylabel("{}".format(title), labelpad=3, fontdict={'size': 16})#labelpad用于控制标题与轴的距离,fontdict是调整字体大小
    # 添加共用的色标卡
    cbar_ax = fig.add_axes([0.91, 0.1, 0.01, 0.8]) # 调整色标卡的位置
    label=['TP', 'FN', 'FP']
    block_height = 1 / len(colors)
    for i in range(len(colors)):
        color = colors[i]
        rect = mpatches.Rectangle((0, i*block_height), 1, block_height, color=color)
        cbar_ax.add_patch(rect)
        cbar_ax.text(1.2, i*block_height+block_height/2, label[i], va='center', fontsize=16)
    cbar_ax.axis('off')
    # 调整子图布局，减小间距
    plt.subplots_adjust(wspace=0.03, hspace=0.05,left=0.1, right=0.9, top=0.9, bottom=0.1)
    # 保存图片（请提供适当的名称）
    plt.savefig("{}/{}_overlop.png".format(image_save,image_name), bbox_inches='tight', dpi=300, format='png')
    #plt.savefig("C:/Users/59278/Desktop/{}_overlop.png".format(image_name), bbox_inches='tight', dpi=300, format='png')
    plt.show()

class Comparison:
    def __init__(self,image_save,resolution=1,samples=10):
        self.image_save = image_save
        os.makedirs(image_save, exist_ok=True)
        self.resolution = resolution
        self.samples = samples
        self.size=1.5
        
    def Variable(self, data,axs,maxmin):
        #_,l = data.shape
        l = len(data)
        x = np.arange(0, l)
        axs.plot(x, data, color='black')
        axs.set_yticks(np.linspace(maxmin[1], maxmin[0],6))
        return axs,l
    
    def Double_Variable(self, data,axs,maxmin):
        #_,l = data.shape
        _,l = data.shape
        x = np.arange(0, l)
        axs.plot(x, data[0],color='black')
        axs.set_yticks(np.linspace(maxmin[1], maxmin[0],6))
        axs1 = axs.twinx()
        axs1.plot(x,data[1],color='red')
        axs1.set_yticks(np.linspace(maxmin[3], maxmin[2],6))
        return axs,axs1,l
    
    def Reflectivity(self,data,axs,maxmin=None,layers=None):
        h,l=data.shape
        #print(maxmin)
        if maxmin is None or (20>maxmin[0] and maxmin[0]>5):
            v_min = -35
            v_max = 20
            ticks = np.arange(v_max, v_min - 1, -10)
        elif 70>maxmin[0] and maxmin[0]>20:
            v_max = 70
            v_min = 0
            ticks = np.arange(v_max, v_min-1, -10)
        else:
            v_max = maxmin[0]
            v_min = maxmin[1]
            ticks = np.linspace(v_max, v_min, 5)
        #print(v_min,v_max)
        colors_list = ["blue","skyblue","cyan","limegreen","yellow","orange", "red","purple"] 
        #col = [# (1, 1, 1, 0)完全透明,# 蓝色（低降水）,# 天蓝色,# 青色,# 浅绿,# 黄色,# 橙色,# 红色（高降水）,# 深紫（极端值）]
        vir_white = colors.LinearSegmentedColormap.from_list("custom_white_yellow", colors_list)
        #vir_white = colors.ListedColormap([[1.,1.,1.]]+cm.viridis.colors)
        dBZ_norm = colors.Normalize(vmin=v_min, vmax=v_max)# 设置色标卡的范围和分度值
        im = axs.imshow(data,extent=[0,l,0,h],aspect='auto',cmap=vir_white, norm=dBZ_norm)#interpolation='nearest',
        if layers is None:
            star=480
            layers=np.arange(star, star+64*240, 240)
        else:
            star=layers[0]
        high = np.linspace(star,np.max(layers),5)
        diff = np.abs(high[:, np.newaxis] - layers)
        y_positions = np.argmin(diff, axis=1)[1:]
        y_ticks = layers[y_positions]
        # 2. 设置刻度位置和标签
        axs.set_yticks(y_positions)# 设置层索引位置
        axs.set_yticklabels([f'{tick/1000:.1f}km' for tick in y_ticks]) # 显示实际高度值
        return axs,im,ticks,l
    
    def Reflectivity_grey(self,data,axs,maxmin=None,layers=None):
        h,l=data.shape
        #print(maxmin)
        if maxmin is None or (20>maxmin[0] and maxmin[0]>5):
            v_min = -35
            v_max = 20
            ticks = np.arange(v_max, v_min - 1, -10)
        elif 70>maxmin[0] and maxmin[0]>20:
            v_max = 70
            v_min = 0
            ticks = np.arange(v_max, v_min-1, -10)
        else:
            v_max = maxmin[0]
            v_min = maxmin[1]
            ticks = np.linspace(v_max, v_min, 5)
        #print(v_min,v_max)
        vir_white = colors.ListedColormap([[1.,1.,1.]]+cm.viridis.colors)
        dBZ_norm = colors.Normalize(vmin=v_min, vmax=v_max)# 设置色标卡的范围和分度值
        im = axs.imshow(data,extent=[0,l,0,h],aspect='auto',cmap=vir_white, norm=dBZ_norm)#interpolation='nearest',
        if layers is None:
            star=480
            layers=np.arange(star, star+64*240, 240)
        else:
            star=layers[0]
        high = np.linspace(star,np.max(layers),5)
        diff = np.abs(high[:, np.newaxis] - layers)
        y_positions = np.argmin(diff, axis=1)[1:]
        y_ticks = layers[y_positions]
        # 2. 设置刻度位置和标签
        axs.set_yticks(y_positions)# 设置层索引位置
        axs.set_yticklabels([f'{tick/1000:.1f}km' for tick in y_ticks]) # 显示实际高度值
        return axs,im,ticks,l
    
    def Cloud_Class(self,data,axs):
        h,l=data.shape
        data[data == 0] = np.nan
        mycvals = [1, 2, 3, 4, 5, 6, 7, 8]
        mycolors = ["#ff0000", "#ff6347", "#ffff00", "#00ff00", "#008000", "#00ffff", "#0000ff", "#9a0eea"]
        label = ['Cirrus', 'Altostratus', 'Altocumulus', 'Stratus', 'Stratocumulus', 'Cumulus', 'Nimbostratus', 'Deep Convection']
        norm1 = plt.Normalize(min(mycvals), max(mycvals))
        mycmap = colors.ListedColormap(mycolors)
        axs.imshow(data,extent=[0,l,0,h],aspect='auto',cmap=mycmap, norm=norm1)
        axs.set_yticks(np.linspace(0,h,5)[1:])
        axs.set_yticklabels([f'{(tick * 240 + 480)/1000:.1f}km' for tick in axs.get_yticks()])
        return axs,mycolors,label,l
    
    def images(self,data_list,data_name,hight_layer=None,indexs=None,sampletitle=None,save_name='1'):
        row = len(data_list)
        if self.samples<=8:
            #sizes=10//self.samples
            sizes=2
        else:
            sizes=1
        fig, axs = plt.subplots(nrows=row, ncols=self.samples, figsize=(int(self.samples*sizes*3+1),row*3.5))
        if self.samples>1:
            axs = axs.reshape(row,self.samples)
        if hight_layer is None:
            hight_layer=[None] * len(data_list)
        for i,data,name,layers in zip(range(row),data_list,data_name,hight_layer):
            low=((row-(i+1))/row)*0.8+0.1
            high=(0.8-(row-1)*0.005)/row
            if isinstance(data,list):
                maxmin = [np.nanmax(data[0]),np.nanmin(data[0]),
                          np.nanmax(data[1]),np.nanmin(data[1])]
                data=np.stack(data, axis=1)
            else:
                if np.isnan(data).all():
                    maxmin = None 
                else:
                    maxmin = [np.nanmax(data),np.nanmin(data)]
            #print(data.shape,maxmin)
            if indexs is not None:
                data=data[indexs]
            #print(data.shape)
            for j in range(self.samples):
                if '_line' in name:
                    title=name.rsplit('_', 1)[0]
                    axs[i,j],l = self.Variable(data[j],axs[i,j],maxmin)
                elif len(name)==2:
                    title = name[0].rsplit('_', 1)[0]
                    title1 = name[1].rsplit('_', 1)[0]
                    axs[i,j],axs1,l = self.Double_Variable(data[j],axs[i,j],maxmin)
                elif '_Reflectivity' in name:
                    title=name.rsplit('_', 1)[0]
                    axs[i,j],im,ticks,l= self.Reflectivity(data[j],axs[i,j],maxmin=maxmin,layers=layers)
                    cbar_ax = fig.add_axes([0.91,low,0.01,high])
                    #print(low,high)
                    cbar = plt.colorbar(im, cax=cbar_ax,orientation='vertical',ticks=ticks, aspect=20)#
                    cbar.ax.tick_params(labelsize=16*self.size)
                elif '_Reflectgrey' in name:
                    title=name.rsplit('_', 1)[0]
                    axs[i,j],im,ticks,l= self.Reflectivity_grey(data[j],axs[i,j],maxmin=maxmin,layers=layers)
                    cbar_ax = fig.add_axes([0.91,low,0.01,high])
                    #print(low,high)
                    cbar = plt.colorbar(im, cax=cbar_ax,orientation='vertical',ticks=ticks, aspect=20)#
                    cbar.ax.tick_params(labelsize=16*self.size)
                elif 'cloud_class' in name:
                    title='Cloud Class'
                    axs[i,j],mycolors,label,l= self.Cloud_Class(data[j],axs[i,j])
                    cbar_ax = fig.add_axes([0.91,low,0.01,high])
                    #print(low,high)
                    block_height = 1 / len(mycolors)
                    for k in range(len(mycolors)):
                        color = mycolors[k]
                        rect = mpatches.Rectangle((0, k * block_height), 1, block_height, color=color)
                        cbar_ax.add_patch(rect)
                        cbar_ax.text(1.2, k * block_height + block_height / 2, label[k], va='center',fontsize=16*self.size)
                    cbar_ax.axis('off')
                else:
                    print('名称错误')
                for spine in axs[i,j].spines.values():
                    spine.set_linewidth(self.size)#设置小图的线宽
                if j==0:
                    axs[i,j].yaxis.set_ticks_position('left')
                    axs[i,j].yaxis.set_label_position('left')
                    axs[i,j].tick_params(axis='y', colors='black')
                    axs[i,j].set_ylabel(title,color='black',labelpad=3,fontdict={'size': 20*self.size})
                    axs[i,j].yaxis.set_tick_params(labelsize=16*self.size)
                else:
                    axs[i,j].yaxis.set_visible(False)
                    
                if len(name)==2:
                    if j==self.samples-1:
                        axs1.yaxis.set_ticks_position('right')
                        axs1.yaxis.set_label_position('right')
                        axs1.tick_params(axis='y', colors='red')
                        axs1.set_ylabel(title1,color='red',labelpad=3,fontdict={'size': 20*self.size})
                        #axs1[i,j].yaxis.set_tick_params(labelsize=16*self.size)
                        axs1.yaxis.set_tick_params(labelsize=16*self.size)
                    else:
                        axs1.yaxis.set_visible(False)
    
                if i==row-1:
                    axs[i,j].set_xticks(np.linspace(0,l,3)[1:])
                    axs[i,j].set_xticklabels([f'{int(tick * self.resolution)}km' for tick in axs[i,j].get_xticks()])
                    axs[i,j].xaxis.set_tick_params(labelsize=16*self.size)
                else:
                    axs[i,j].set_xticks([])
                if i==0:
                    if sampletitle is None:
                        axs[i,j].set_title(f'Sample{j+1}',fontsize=25*self.size)
                    else:
                        axs[i,j].set_title(f'{sampletitle[j]}',fontsize=25*self.size)
        if row!=1:
            image_name=f'{save_name}_comparison'
        elif len(name)==2:
            image_name=f'{save_name}_{title}_{title1}_line'
        else:
            image_name=f'{save_name}_{name}'
        plt.subplots_adjust(wspace=0.05, hspace=0.15,left=0.1, right=0.908, top=0.9, bottom=0.1)
        plt.savefig(f"{self.image_save}/{image_name}.png", bbox_inches='tight', dpi=300, format='png')
        plt.close()  # 关闭图形释放内存
        del fig, axs,data,name,layers,data_list    # 删除对象引用
        gc.collect()    # 强制垃圾回收
        print(f"Saved {image_name} and cleared memory")  
        
        
'''      
def set_axis_style(ax):
    """设置坐标轴样式"""
    # 设置边框线宽
    for spine in ax.spines.values():
        spine.set_linewidth(self.axis_linewidth)
    
    # 设置刻度线宽和长度
    ax.tick_params(axis='both', which='major', 
                  width=self.tick_width, 
                  length=self.tick_width * 4,  # 刻度线长度与宽度成比例
                  labelsize=self.tick_label_size)
    
    # 设置刻度标签字体大小
    ax.xaxis.set_tick_params(labelsize=self.tick_label_size)
    ax.yaxis.set_tick_params(labelsize=self.tick_label_size)
'''