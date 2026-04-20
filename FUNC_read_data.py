# -*- coding: utf-8 -*-
"""
Created on Mon Apr  1 14:20:34 2024

@author: 59278
"""
import os
import numpy as np
from pyhdf import HDF,SD,VS,V
from netCDF4 import Dataset
import xarray as xr
import datetime
import pandas as pd
import transbigdata as tbd
import bz2
import struct
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import imageio
import platform
import re
import subprocess

"""筛选数据"""
def extract_time_from_path(path):
    filename = os.path.basename(path)  # 提取文件名
    match = re.search(r'(\d{8})_(\d{4})', filename)
    if match:
        return int(match.group(2))  # 返回时间部分
    return 0

def extract_numbers(filename):
    numbers = re.findall(r'\d+', filename)
    return tuple(map(int, numbers))

def search_files(data_path, keys=None, file_type=None):
    """
    搜索指定路径下符合要求的文件。

    参数:
        data_path: 文件夹路径
        keys: 关键词列表（可选），文件名必须包含列表中所有关键词（子串匹配，区分大小写）
        file_type: 文件扩展名（可选），文件名必须为‘.格式’形式，如 '.nc'、'.png' 等

    返回:
        按时间排序的文件路径列表（Path 对象）
    """
    base_path = Path(data_path)
    if not base_path.is_dir():
        print('文件目录不存在')
        return []

    # 确定扩展名模式
    if file_type is None:
        pattern = '*'
    elif '.' in file_type:
        pattern = f'*{file_type}'
    else:
        print("文件格式输入错误！ 不是‘.格式'的形式")
        return []
    all_files = list(base_path.glob(pattern))
    if keys is None:
        files_path = all_files
    elif isinstance(keys, (list, tuple)):
        files_path = [f for f in all_files if all(k in f.name for k in keys)]
    else:
        print("‘关键词’输入格式错误！不是‘列表’或‘元组’ ")
        return []
    return sorted(files_path, key=lambda p: extract_numbers(p.name))

'''读取HDF或者nc的变量名称''' 
def read_variable_name(file_path):
    if file_path.lower().endswith(".hdf"):
        file = SD.SD(file_path, SD.SDC.READ)
        variables_name = file.datasets()  # 显示文件中子文件的列表
        return variables_name
        file.end()
    elif file_path.lower().endswith(".nc"):
        file = Dataset(file_path, "r")
        variables_name = list(file.variables.keys())
        return variables_name
        file.close()
    else:
        return '非HDF,NC文件'

'''读取HDF或者nc的变量数据''' 
def read_source_data(file_path,name,dtype=np.float64):
    if file_path.lower().endswith(".nc"):
        file = Dataset(file_path, "r")
        variable_values = np.array(file.variables[name][:])
        #variable_values=variable_values.filled(fill_value=np.nan).astype(np.float64)
        file.close()
        return variable_values.astype(dtype)
    elif file_path.lower().endswith(".hdf"):
        file = SD.SD(file_path, SD.SDC.READ)
        variable_values = file.select(name)
        variable_values = variable_values.get()
        variable_values = variable_values.astype(dtype)
        if np.any(variable_values == -9999):
            variable_values = np.where(variable_values == -9999, np.nan, variable_values)
        if np.any(variable_values == -999):
            variable_values = np.where(variable_values == -999, np.nan, variable_values)
        file.end()
        return variable_values
    else:
        return '非HDF,NC文件'

'''读取Vdata数据''' 
def read_source_Vdata(file_path,name):
    file=HDF.HDF(file_path, HDF.HC.READ)
    vdata = file.vstart()
    VD_object = vdata.attach(name)
    vInfo = VD_object.inquire()
    variable_values = VD_object.read(nRec=vInfo[0])#nRec为行数
    variable_values= np.squeeze(variable_values)
    if np.any(variable_values == -9999):
        variable_values[variable_values == -9999] = 0
    # 关闭数据集
    VD_object.detach()
    # 关闭文件
    file.close()
    return variable_values

'''读取cloudsat数据'''
def read_cloudsat(file_path):
    file = SD.SD(file_path, SD.SDC.READ)
    
    Radar_Reflectivity = file.select('Radar_Reflectivity')
    Radar_Reflectivity = Radar_Reflectivity.get()[:,38:102]
    Radar_Reflectivity = Radar_Reflectivity.astype(np.float64)
    #Radar_Reflectivity[Radar_Reflectivity == -8888] = np.nan
    Radar_Reflectivity = Radar_Reflectivity/100
    Radar_Reflectivity[Radar_Reflectivity <-35] = np.nan
    Radar_Reflectivity[Radar_Reflectivity > 20] = 20
    
    CPR_Cloud_mask = file.select('CPR_Cloud_mask')
    CPR_Cloud_mask = CPR_Cloud_mask.get()[:,38:102]
    CPR_Cloud_mask = CPR_Cloud_mask.astype(np.float64)
    CPR_Cloud_mask[CPR_Cloud_mask == -9] = np.nan
    CPR_Cloud_mask[CPR_Cloud_mask == 5] = np.nan
    CPR_Cloud_mask[CPR_Cloud_mask == 0] = np.nan
    CPR_Cloud_mask[~np.isnan(CPR_Cloud_mask)] = 1
    
    Reflectivity= np.multiply(Radar_Reflectivity,CPR_Cloud_mask)
    
    file.end()
    return Reflectivity,CPR_Cloud_mask

'''处理cloudsat数据'''
def Make_Reflectivity(Reflectivity,Cloud_mask):
    Radar_Reflectivity = Reflectivity/100
    Radar_Reflectivity[Radar_Reflectivity <-35] = np.nan
    Radar_Reflectivity[Radar_Reflectivity > 20] = 20
    CPR_Cloud_mask = Cloud_mask.astype(np.float64)
    CPR_Cloud_mask[CPR_Cloud_mask == -9] = np.nan
    CPR_Cloud_mask[CPR_Cloud_mask == 5] = np.nan
    CPR_Cloud_mask[CPR_Cloud_mask == 0] = np.nan
    CPR_Cloud_mask[~np.isnan(CPR_Cloud_mask)] = 1
    Reflect= np.multiply(Radar_Reflectivity,CPR_Cloud_mask)
    return Reflect,CPR_Cloud_mask

'''读取Himawari数据'''
def read_Himawari(file_path,name):
    file = Dataset(file_path, "r")
    variable_values = np.array(file.variables[name][:])
    variable_values[variable_values == -327.68] = 0
    variable_values[variable_values == -655.36] = 0
    variable_values[variable_values == -999] = 0
    variable_values[variable_values == -128] = 0
    variable_values[variable_values == -99] = 0
    variable_values[variable_values == -1] = 0
    file.close()
    return variable_values#[:960]

class read_swan:
    def __init__(self, filename):
        self.filename = filename
        f = bz2.BZ2File(filename, "rb")
        self.ZonName = struct.unpack("12s", f.read(12))
        self.DataName = struct.unpack("38s", f.read(38))  # 数据说明(例如 2008年5月19日雷达三维拼图)38个字节
        self.Flag = struct.unpack("8s", f.read(8))  # 文件标志，"swan"
        self.Version = struct.unpack("8s", f.read(8))  # 数据版本号，"1.0"
        self.year = struct.unpack("H", f.read(2))
        self.month = struct.unpack("H", f.read(2))
        self.day = struct.unpack("H", f.read(2))
        self.hour = struct.unpack("H", f.read(2))
        self.minute = struct.unpack("H", f.read(2))
        self.interval = struct.unpack("H", f.read(2))
        self.XNumGrids = struct.unpack("H", f.read(2))[0]
        self.YNumGrids = struct.unpack("H", f.read(2))[0]
        self.ZNumGrids = struct.unpack("H", f.read(2))[0]
        self.RadarCount = struct.unpack("i", f.read(4))  # 拼图雷达数 四个字节
        self.StartLon = struct.unpack("f", f.read(4))[0] # 网格开始经度（左上角） 四个字节
        self.StartLat = struct.unpack("f", f.read(4))[0]  # 网格开始纬度（左上角） 四个字节
        self.CenterLon = struct.unpack("f", f.read(4))[0]  # 网格中心经度 四个字节
        self.CenterLat = struct.unpack("f", f.read(4))[0]  # 网格中心纬度 四个字节
        self.XReso = struct.unpack("f", f.read(4))[0]  # 经度方向分辨率 四个字节
        self.YReso = struct.unpack("f", f.read(4))[0]  # 纬度方向分辨率 四个字节
        self.ZhighGrids = struct.unpack("40f", f.read(40 * 4))
        self.RadarStationName = []
        for i in range(20):
            self.RadarStationName.append(struct.unpack("16s", f.read(16)))
        self.RadarLongitude = struct.unpack("20f", f.read(20 * 4))
        self.RadarLatitude = struct.unpack("20f", f.read(20 * 4))
        self.RadarAltitude = struct.unpack("20f", f.read(20 * 4))
        self.MosaicFlag = struct.unpack("20B", f.read(20))
        f.read(172)
        tempdata = np.frombuffer(f.read(self.ZNumGrids * self.YNumGrids * self.XNumGrids), dtype="B")
        self.data = np.array(tempdata, dtype=float)
        self.data.shape = self.ZNumGrids, self.YNumGrids, self.XNumGrids
        del tempdata
        
    def make_data(self,):
        lon1 = self.StartLon + (self.XNumGrids - 1) * self.XReso
        lat1 = self.StartLat - (self.YNumGrids - 1) * self.YReso
        grid_lon = np.linspace(self.StartLon, lon1, self.XNumGrids)
        grid_lat = np.linspace(self.StartLat, lat1, self.YNumGrids)
        self.data[self.data==0]=np.nan
        dbz = (self.data - 66) / 2
        dbz[dbz <= -32.5] = 0 # 雷达观测无值
        lev = np.array([500,1000,1500,2000,2500,3000,3500,4000,4500,5000,5500,6000,7000,8000,9000,10000,12000,14000,15500,17000,19000])
        return dbz,grid_lon,grid_lat,lev
       
    def read(self,use_area=None):
        area={'minlat':17,
              'maxlat':55,
              'minlon':70,
              'maxlon':137,
              'baselayer':500,
              'toplayer':19000}
        times=str(self.year[0])+str(self.month[0]).zfill(2)+str(self.day[0]).zfill(2)+str(self.hour[0]).zfill(2)+str(self.minute[0]).zfill(2)+'00'
        time_interval = str(self.interval[0]).zfill(2)
        dbz,grid_lon,grid_lat,layers=self.make_data()
        if use_area is not None:
            for key, value in use_area.items():
                area[key]=value
            index_lat=np.where((area['maxlat']>=grid_lat)&(grid_lat>=area['minlat']))[0]
            index_lon=np.where((area['maxlon']>=grid_lon)&(grid_lon>=area['minlon']))[0]
            index_lay=np.where((area['toplayer']>=layers)&(layers>=area['baselayer']))[0]
            dbz = dbz[np.ix_(index_lay, index_lat, index_lon)]
            grid_lon=grid_lon[index_lon]
            grid_lat=grid_lat[index_lat]
            layers = layers[index_lay]
        maxdbz = np.nanmax(dbz,axis=0)
        return dbz,grid_lon,grid_lat,layers,maxdbz,times,time_interval
    
    def save(self,save_path,save_name,area=None):
        data={}
        data['dbz'],data['grid_lon'],data['grid_lat'],data['layers'],data['maxdbz'],data['times'],data['time_interval']=self.read(use_area=area)
        save_data(data, save_path, save_name)

def read_time(file_path):
    '''读取每个数据点对应的时间.
    datetime=True: 返回所有数据点的日期时间组成的DatetimeIndex.
    datetime=False: 返回所有数据点相对于第一个点所经过的秒数组成的numpy数组.
    '''
    second = np.array(read_source_Vdata(file_path,'Profile_time'))
    TAI = read_source_Vdata(file_path,'TAI_start').item()
    start = pd.to_datetime('1993-01-01') + pd.Timedelta(seconds=TAI)
    offsets = pd.to_timedelta(second, unit='s')
    time = pd.date_range(start=start, end=start, periods=offsets.size)
    time = time + offsets
    return time

'''时间四舍五入'''
def round_datetime(datetime_series: pd.Series) -> pd.Series:
    """向量化四舍五入实现"""
    total_seconds = (
        datetime_series.hour * 3600 
        + datetime_series.minute * 60 
        + datetime_series.second)
    rounded_seconds = (total_seconds + 300) // 600 * 600
    return pd.to_datetime(datetime_series.normalize()+ pd.to_timedelta(rounded_seconds, unit='s'))

'''日期转化为第几天'''
def convert_date_to_day(year, month, day):
    date = datetime.datetime(year, month, day)
    day_of_year = (date - datetime.datetime(date.year, 1, 1)).days + 1
    return day_of_year

'''第几天转化为日期'''
def date_conversation(year,day):
    #输入的字符串类型的年和日转换为整型
    year=int(year)
    day=int(day)
    #first_day：此年的第一天
    #类型：datetime
    first_day=datetime.datetime(year,1,1)
    #用一年的第一天+天数-1，即可得到我们期望的日期
    #-1是因为当年的第一天也算一天
    wanted_day=first_day+datetime.timedelta(day-1)
    #返回需要的字符串形式的日期
    wanted_day=datetime.datetime.strftime(wanted_day,'%Y%m%d')
    return wanted_day

'''样本切片'''
def Slice_Sample(data,cloudsat_long,Array=True):
    sample_number=data.shape[0]//cloudsat_long
    data_slic = data[:sample_number*cloudsat_long]
    if Array:
        data_split = np.array(np.array_split(data_slic, sample_number, axis=0))
    else:
        data_split = np.array_split(data_slic, sample_number, axis=0)
    return data_split

'''筛选根据样本数据量筛选样本'''
def Filt_Sample(data,cloudsat_long,rate_range):
    data = Slice_Sample(data,cloudsat_long)
    if np.ndim(data)==3:
        #data = np.transpose(data, (0,2,1))
        data_count = np.sum(~np.isnan(data), axis=(1, 2))
        k=64
    else:
        data_count = np.sum(~np.isnan(data), axis=(1))
        k=1
    data_index = np.where(((cloudsat_long * k * rate_range[1]) >= data_count) & (data_count >= (cloudsat_long * k * rate_range[0])), 1, np.nan)
    return data,data_index

'''经纬度与葵花8匹配'''
def match_position(c_cood):
    c_cood = c_cood[np.arange(2,len(c_cood)+2,5)]
    params = tbd.area_to_params([80,-60,180,60], accuracy=5560)
    aindex_h=np.empty((c_cood.shape[0],2))
    aindex_h[:,1],aindex_h[:,0] = tbd.GPS_to_grid(c_cood[:,0], c_cood[:,1], params)
    #print(np.min(aindex_h),np.max(aindex_h))
    #len(np.unique(aindex_h[:0],axis=0))
    aindex_h[:,1][aindex_h[:,1] < 0] += 7200
    aindex_h[:,0]=np.abs(aindex_h[:,0]-2400)
    aindex_h = aindex_h.astype(np.int64)
    return aindex_h

'''平滑样本'''
def smoothness(var,windowsize,name):
    var1 = rescale_variable(var,name)
    re = np.zeros_like(var)
    array  = np.abs(np.abs(np.arange(windowsize+1)[-windowsize:]-windowsize//2-1)-windowsize//2-1)
    window = array/np.sum(array)
    '''
    window = np.ones(int(windowsize))/float(windowsize)
    '''
    for i in range(len(var1)):
        re[i] = np.convolve(var1[i], window, 'same')
    #var2 = moving_average(var1, 5)
    out = Normal_Variable(re,name)
    return out

def rescale_variable(var,name):
    if name=='Cloud_Optical_Thickness':
        var = np.exp((1.13*var+2.20))*100
    elif name=='Cloud_top_pressure_1km':
        var = (265*var+532)*10
    elif name=='Cloud_Effective_Radius':
        var = np.exp((0.542*var+3.06))*100
    elif name=='Cloud_Water_Path':
        var = np.exp((1.11*var+0.184))*100
    else:
        var=var
    return var

def Normal_Variable(var,name):
    var = np.where(var == 0, np.nan, var)
    #if name.startswith('albedo_0'):
        #var = (var*2)-1
    if name=='Cloud_Optical_Thickness':
        var = (np.log(var*0.01)-2.2)/1.13
    elif name=='Cloud_top_pressure_1km':
        var = ((var*0.1)-532)/265
    elif name=='Cloud_Effective_Radius':
        var = (np.log(var*0.01)-3.06)/0.542
    elif name=='Cloud_Water_Path':
        var = (np.log(var*0.01)-0.184)/1.11
    elif name=='albedo_01' or name=='albedo_02' or name=='albedo_03' or name=='albedo_04':
        var[var>1]=1
        var=var*6-3
    elif name=='albedo_05':
        var=var*9-3
    elif name=='albedo_06':
        var=var*10-2.5
    elif name=='tbb_07':
        var = ((var-235)/118)*7-3
    elif name=='tbb_08':
        var = ((var-187)/110)*9-3
    elif name=='tbb_09':
        var = ((var-185)/84)*6-3
    elif name=='tbb_10':
        var = ((var-185)/89)*6-3
    elif name=='tbb_11':
        var = ((var-184)/114)*6-3
    elif name=='tbb_12':
        var = ((var-205)/73)*6-3
    elif name=='tbb_13':
        var = ((var-184)/117)*6-3
    elif name=='tbb_14':
        var = ((var-183)/117)*6-3
    elif name=='tbb_15':
        var = ((var-183)/112)*6-3
    elif name=='tbb_16':
        var = ((var-185)/95)*6-3
    elif name=='SOZ' or name=='SAZ':
        var = (np.cos(np.radians(var)))*6-3
    else:
        var=var
    var[var==0]=5.65760608270041e-08
    var[np.isnan(var)]=0
    return var

def read_varible(file_path,var_name,random=None,trans=False,Normal=False):
    data={}
    samples=len(read_source_data(file_path,var_name[0]))
    if random is not None:
        indices = np.random.choice(samples, size=random, replace=False)
    for name in var_name:
        if name=='Cloud_Mask_1km':
            varible=read_source_data(file_path,name,dtype=np.int64)
        else:
            varible=read_source_data(file_path,name,dtype=np.float64)
        if varible.ndim==3 and trans:
            varible=np.transpose(varible,(0,2,1))
        try:
            varible=varible[indices]
        except:
            varible=varible
        if Normal:
            data[name]=Normal_Variable(varible,name)
        else:
            data[name]=varible
    ncfile=file_path.split('/')[-1]
    print(f'{ncfile}:样本{samples},读取{len(data[name])}')
    #input_varible = np.stack(list(data.values()),axis=2)
    return data

def scale_varible(varible,var_name):
    data={}
    for name in var_name:
        data[name]=Normal_Variable(varible[name],name)
        print(name,np.min(data[name]),np.max(data[name]))
    input_varible = np.stack(list(data.values()),axis=2)
    return input_varible

'''标准化cloudsat'''
def scale_Reflect(data):
    Reflectivity=2*(data+35)/55-1
    Reflectivity=np.expand_dims(Reflectivity, axis=-1)
    return np.nan_to_num(Reflectivity, nan=-1)

'''解码真实图片'''
def rescale_real(scene, Z_range=(-35,20), missing_max=-35):
    if not isinstance(scene, np.ndarray):
        scene=scene.numpy()
    sc=np.where(scene==-1,np.nan,scene)
    sc = Z_range[0] + (sc+1)/2.0*(Z_range[1]-Z_range[0])
    #sc[sc <= missing_max] = np.nan
    return sc

'''解码生成图片'''
def rescale_gen(scene, Z_range=(-35,20), missing_max=-35):
    if not isinstance(scene, np.ndarray):
        sc=scene.numpy()
    else:
        sc=scene
    sc = Z_range[0] + (sc+1)/2.0 * (Z_range[1]-Z_range[0])
    #sc[sc <= missing_max] = np.nan
    return sc

def crop_borders(image,crop_ratio=None):
    """自动裁剪图像周围的空白区域"""
    if crop_ratio is None:
        img_array = np.array(image.convert('RGB'))
        #height, width, _ = img_array.shape
        # 检测背景色（取四个角的平均颜色）
        corners = [
            img_array[0, 0], img_array[0, -1], 
            img_array[-1, 0], img_array[-1, -1]]
        bg_color = np.mean(corners, axis=0).astype(int)
        # 创建掩码检测非背景区域
        mask = np.any(np.abs(img_array - bg_color) > 10, axis=2)
        # 找到边界
        rows = np.any(mask, axis=1)
        cols = np.any(mask, axis=0)
        top, bottom = np.where(rows)[0][[0, -1]]
        left, right = np.where(cols)[0][[0, -1]]
    else:
        # 按比例裁剪逻辑
        if len(crop_ratio) != 4:
            raise ValueError("crop_ratio 参数必须是包含4个数值的列表")
        top_ratio,bottom_ratio,left_ratio,right_ratio= crop_ratio
        # 验证比例值范围
        if not (0 <= left_ratio < right_ratio <= 1 and 0 <= top_ratio < bottom_ratio <= 1):
            raise ValueError("裁剪比例必须在0到1之间，且左/上比例小于右/下比例")
        width, height = image.size
        # 计算裁剪坐标
        left = int(width * left_ratio)
        right = int(width * right_ratio)
        top = int(height * top_ratio)
        bottom = int(height * bottom_ratio)
    # 确保裁剪区域至少有一个像素
    if left >= right or top >= bottom:
        raise ValueError("裁剪比例计算后得到的区域无效")
    return image.crop((left, top, right, bottom))

def add_title_to_image(input_image_path,output_image_path,title,title_size=30,cut_ratio=None,height=None,width=None,bold=True):
    '''
    cut_ratio=[上，下，左，右]比例
    '''
    title_margin=5
    # 打开原始图片
    img = Image.open(input_image_path)
    # 剪裁掉四周空白部分
    img_cropped = crop_borders(img,crop_ratio=cut_ratio)
    # 创建一个新的图像，向上添加空白部分
    if height is None:
        new_height = img_cropped.height + title_size + title_margin + 15
    else:
        new_height = height + title_size + title_margin + 15
    if new_height<img_cropped.height:
        return print("图片输入高度太低了")
    if width is None: 
        new_width = img_cropped.width
    else:
        new_width = width
    if new_width<img_cropped.width:
        return print("图片输入宽度太窄了")
    new_img = Image.new("RGB", (new_width, new_height), (255, 255, 255))
    # 将剪裁后的图片粘贴到新图像的下方
    new_img.paste(img_cropped, (int(new_width/2-img_cropped.width/2), new_height-img_cropped.height))
    # 添加标题
    draw = ImageDraw.Draw(new_img)
    # 获取系统类型
    system = platform.system().lower()
    font = None
    try:
        # Windows 系统
        if system == "windows":
            if bold:
                # 尝试加载 Arial 粗体
                try:
                    font = ImageFont.truetype("arialbd.ttf", title_size)
                except:
                    # 使用常规 Arial
                    font = ImageFont.truetype("arial.ttf", title_size)
            else:
                font = ImageFont.truetype("arial.ttf", title_size)
        # Linux 系统
        elif system == "linux":
            try:
                if bold:
                    font = ImageFont.truetype("DejaVuSans-Bold.ttf", title_size)
                else:
                    font = ImageFont.truetype("DejaVuSans.ttf", title_size)
            except:
                # 如果 DejaVu 不可用，尝试 FreeSans
                if bold:
                    font = ImageFont.truetype("FreeSansBold.otf", title_size)
                else:
                    font = ImageFont.truetype("FreeSans.otf", title_size)
        # 其他系统 (macOS)
        else:
            try:
                if bold:
                    font = ImageFont.truetype("Arial Bold.ttf", title_size)
                else:
                    font = ImageFont.truetype("Arial.ttf", title_size)
            except:
                # 尝试 Helvetica
                if bold:
                    font = ImageFont.truetype("Helvetica-Bold.ttf", title_size)
                else:
                    font = ImageFont.truetype("Helvetica.ttf", title_size)
    except Exception as e:
        print(f"字体加载错误: {e}")
        font = ImageFont.load_default()
    # 如果仍未设置字体，使用默认字体
    if font is None:
        font = ImageFont.load_default()
    # 计算标题位置
    bbox = draw.textbbox((0, 0), title, font=font)
    title_width = bbox[2] - bbox[0]
    title_height = bbox[3] - bbox[1]
    # 居中位置
    title_position = ((new_img.width - title_width) // 2, 
                      (title_margin + (title_size - title_height) // 2))
    # 绘制标题
    if bold and (font.path is None or ("bold" not in font.path.lower() and "bd" not in font.path.lower())):
        # 手动加粗效果
        for dx, dy in [(0, 0), (1, 0), (0, 1), (1, 1)]:
            draw.text((title_position[0] + dx, title_position[1] + dy), 
                      title, fill="black", font=font)
    else:
        draw.text(title_position, title, fill="black", font=font)
    # 保存新图片
    new_img.save(output_image_path)
    return print(f'{output_image_path}已完成')
    
    
def Generate_GIF(image_path,save_name,fps,keywords=None,file_type=None):
    # 设置输出文件扩展名
    image_files=search_files(image_path,keywords,file_type)
    if len(image_files)==0:
        print("未找到符合条件的图片！")
        return
    # 读取所有图片
    images = []
    for img_path in image_files:
        try:
            img = imageio.imread(img_path)
            images.append(img)
        except Exception as e:
            print(f"无法处理 {img_path}: {str(e)}")
    num_images=len(images)
    if num_images==0:
        print("没有可用的图片数据！")
        return
    imageio.mimsave(f'{image_path}/{save_name}.gif',images,duration=1/fps,loop=0)  # 0 = 无限循环
    print(f"成功生成 GIF:{image_path}/{save_name}.gif, {num_images}帧, {1/fps}ms")
    del images
    return


'''
def save_data(data, save_path, save_name):
    ds = xr.Dataset()
    for key, value in data.items():
        if isinstance(value, np.ndarray):
            dim_list=[]
            for dim in list(range(1, len(value.shape) + 1)):
                dim_list.append(f'{key}_dim{dim}')
            ds[key] = (dim_list,value)
        else:
            ds[key] = value
    # 构建文件路径
    file_path = f"{save_path}/{save_name}.nc"
    os.makedirs(save_path, exist_ok=True)
    # 保存为 NetCDF 文件
    ds.to_netcdf(file_path)
'''

'''保存数据'''
def save_data(data,save_path,save_name):
    '''
    这个函数支持两种'data'的输入形式
    第一种：
    data = {'time':'2025-09-24 04:00:00 UTC',
            'shape': {'High':High, 'Lat':Lat, 'Lon':Lon},#必须输入
            'reflectivity': {'value': reflectivity ,#必须输入
                             'dims':['High','Lat','Lon'],#必须输入
                             'long_name':None,
                             'description':'Equivalent W-band',
                             'units':'dbz',},
            }
    第二种：
    data = {'time':'2025-09-24 04:00:00 UTC',
            'reflectivity':  np.random.rand(128,64),
            'wind': np.random.rand(512,256,64),
            }
    '''
    if 'shape' in data:
        ds = xr.Dataset(coords = data['shape'])
        for key, value in data.items():
            if key == 'shape':
                print(f'跳过{key}')
                continue
            if not isinstance(value, dict):
                ds[key] = value
            else:
                ds[key] = (value['dims'], value['value'])
                del value['dims'],value['value']
                for title,state in value.items():
                    if state is None:
                        continue
                    ds[key].attrs[title]=state
    else:
        ds = xr.Dataset()
        for key, value in data.items():
            if isinstance(value, np.ndarray):
                # 生成带变量名的唯一维度名，例如 reflectivity_dim0, reflectivity_dim1
                dim_names = [f'{key}_dim{i}' for i in range(value.ndim)]
                ds[key] = (dim_names, value)
            else:
                ds[key] = value
    file_path = f"{save_path}/{save_name}.nc"
    os.makedirs(save_path, exist_ok=True)
    ds.to_netcdf(file_path)
    print(f"{file_path}已保存")
    return file_path

'''传输数据'''
def transfer_data(local_file, remote_host, remote_path,
                  remote_user, password, remote_port=22):
    """
    把'当前服务器'的文件传输到'目标服务器'
    local_file = '/work/home/acmh4zm9q3/Data_3D/Danas'   #本地服务器：文件的完整路径
    remote_path = '/data4/xiongqq/Gen_3D_samples/Danas'  #目标服务器：文件夹路径
    remote_host = '202.195.237.131'                      # 目标服务器：ID
    remote_port = 22222                                  # 目标服务器：端口
    remote_user = 'xiongqq'                              # 目标服务器：用户名
    remote_password = 'Y592787276zw'                     # 目标服务器：用户密码
    """
    remote_dest = f"{remote_user}@{remote_host}:{remote_path}/"
    cmd = ["sshpass", "-p", password,
           "scp", "-P", str(remote_port), "-r", local_file, remote_dest]
    try:
        subprocess.run(cmd,check=True)#capture_output=True, text=True,
        print(f"文件已传输到 {remote_user}@{remote_host}:{remote_path}")
    except subprocess.CalledProcessError as e:
        print(f"传输失败：{e.stderr}")
        raise
    try:
        os.remove(local_file)
        print(f"本地文件已删除：{local_file}")
    except OSError as e:
        print(f"删除本地文件失败：{e}")