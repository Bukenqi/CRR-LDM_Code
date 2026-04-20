# -*- coding: utf-8 -*-
"""
Created on Wed Aug  6 21:56:30 2025

@author: 59278
"""


import numpy as np
import vtk
from vtk.util import numpy_support
import os
import urllib.request
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
from PIL import Image, ImageDraw, ImageFont
import platform
import gc

# 下载NASA蓝色大理石图像
def download_nasa_blue_marble(path,resolution):
    if resolution =='high':
        url = "https://eoimages.gsfc.nasa.gov/images/imagerecords/73000/73909/world.topo.bathy.200412.3x21600x10800.png"
        filename = f"{path}/source_word_map_hight.png"
    elif resolution =='low':
        url = "https://eoimages.gsfc.nasa.gov/images/imagerecords/73000/73909/world.topo.bathy.200412.3x5400x2700.png"
        filename = f"{path}/source_word_map_low.png"
    if not os.path.exists(filename):
        print("下载NASA蓝色大理石图像.....")
        print("下载时间长！等着！别着急！")
        urllib.request.urlretrieve(url, filename)
        print("下载完成")
    return filename

def generate_basemap(line_spacing,output_path,name,resolution ='low',area=None):
    os.makedirs(output_path, exist_ok=True)
    if area is None:
        area = {'minlon': -180,'maxlon': 180,'minlat': -90,'maxlat': 90}
    if resolution =='high':
        rate=100
        Image.MAX_IMAGE_PIXELS = 233280000
    elif resolution =='low':
        rate=20
    ground_map_path = os.path.dirname(os.path.abspath(__file__))
    word_image_path = download_nasa_blue_marble(ground_map_path,resolution)
    word_image = Image.open(word_image_path)
    word_image = word_image.resize((360*rate, 180*rate), Image.LANCZOS)
    x1,y1=(area['minlon']+180)*rate,np.abs(area['maxlat']-90)*rate#左上角
    x2,y2=(area['maxlon']+180)*rate,np.abs(area['minlat']-90)*rate#右下角
    word_image = word_image.crop((x1, y1, x2, y2))
    
    lat_span = area['maxlat'] - area['minlat']
    lon_span = area['maxlon'] - area['minlon']
    fig = plt.figure(figsize=(lon_span,lat_span), dpi=rate)
    ax = fig.add_axes([0, 0, 1, 1])
    m = Basemap(projection='cyl',
                llcrnrlon=area['minlon'],llcrnrlat=area['minlat'],
                urcrnrlon=area['maxlon'],urcrnrlat=area['maxlat'],
                resolution='i',ax=ax)
    m.imshow(word_image, origin='upper', extent=[area['minlon'], area['maxlon'], area['minlat'], area['maxlat']])
    # 绘制地图元素
    #line_width = max(2, lat_span / 60)
    line_width = 2
    m.drawcountries(color='black', linewidth=line_width)
    m.drawcoastlines(color='white', linewidth=line_width)#'black''white''lightgray'
    latitudes = list(np.arange(area['minlat'], area['maxlat'] + 1, line_spacing))
    longitudes = list(np.arange(area['minlon'], area['maxlon'] + 1, line_spacing))
    m.drawparallels(latitudes,color='white',linewidth=line_width,dashes=[4, 2])
    m.drawmeridians(longitudes,color='white',linewidth=line_width,dashes=[4, 2])
    save_path=f'{output_path}/{name}_{resolution}.png'
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
    plt.close(fig) 
    return save_path


def vectorized_geographic_to_cartesian(lons, lats, alts):
    """向量化地理坐标到笛卡尔坐标的转换"""
    # 地球半径 (单位:米)
    EARTH_RADIUS = 6371000.0
    # 将角度转换为弧度
    lon_rad = np.radians(lons)
    lat_rad = np.radians(lats)
    # 计算笛卡尔坐标
    r = EARTH_RADIUS + alts
    x_rad = r*np.cos(lat_rad) * np.cos(lon_rad)
    y_rad = r*np.cos(lat_rad) * np.sin(lon_rad)
    z_rad = r*np.sin(lat_rad)
    return x_rad, y_rad, z_rad

def create_world_actor(line_spacing,output_path,area=None):
    print("\n===== 创建背景Actor'=====")
    print('第一步：创建地图背景')
    if area is None:
        print(' 1.创建个球')
        image_path=generate_basemap(line_spacing,output_path,'base_map',resolution ='low',area=area)
        world = vtk.vtkTexturedSphereSource()
        world.SetThetaResolution(300)
        world.SetPhiResolution(300)
        world.SetRadius(6371000)  # 使用实际地球半径(米)
        # 创建变换
        transform = vtk.vtkTransform()
        transform.RotateZ(180)  # 沿Y轴旋转180度
    else:
        print(' 1.制作地理平面')
        image_path=generate_basemap(line_spacing,output_path,'base_map',resolution ='high',area=area)
        lat = int((area['maxlat'] - area['minlat']) / 0.01)-1
        lon = int((area['maxlon'] - area['minlon']) / 0.01)-1
        world = vtk.vtkPlaneSource()
        world.SetOrigin(0, 0, 0)  # 左下角
        world.SetPoint1(lon, 0, 0)  # 右下角
        world.SetPoint2(0, lat, 0)  # 左上角
    print(' 2.纹理映射到背景')
    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputConnection(world.GetOutputPort())
    
    print('第二步：读取地图图片...')
    reader = vtk.vtkPNGReader()#创建 PNG 图像读取器对象
    reader.SetFileName(image_path)#设置要读取的图像文件路径
    reader.Update()#执行实际的图像读取操作
    
    print('第三步：实例化VTK纹理对象...')
    texture = vtk.vtkTexture()#实例化 VTK 纹理对象，用于将2D图像映射到3D几何体表面
    texture.SetInputConnection(reader.GetOutputPort())#将纹理连接到读取器的输出
    texture.InterpolateOn()#当纹理被拉伸或缩小时，使用双线性插值平滑图像
    texture.RepeatOff()#当纹理坐标超出 [0,1] 范围时，不重复平铺纹理
    texture.EdgeClampOn()#当纹理坐标超出边界时，延伸边缘像素颜色
    
    print('第四步：创建Actor，将上边做好的添加入Actor...')
    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    actor.SetTexture(texture)
    actor.GetProperty().SetAmbient(0.5)#设置环境光系数
    actor.GetProperty().SetDiffuse(0.9)#设置漫反射系数
    actor.GetProperty().SetSpecular(0.5)#设置镜面反射系数
    actor.GetProperty().SetSpecularPower(20)#设置高光锐度
    # 将变换应用于actor
    if area is None:
        actor.SetUserTransform(transform)
    print("=======================\n")
    return actor

def data_process(mask,area):
    scale_x, scale_y ,scale_z = 1,1,5# 例如，如果希望z方向间距为5，则设为5.0
    if area is None:
        point_size=0.5
        # x和y方向间隔为1
        # 获取有效点索引并缩放
        high, lats, lons = np.where(mask)
        lons = lons * scale_x
        lats = lats * scale_y
        high = high * scale_z
        print(" 1.已完成三维网格点坐标")
    else:
        point_size=350
        # 3. 创建坐标数组
        lons = np.linspace(area['minlon'],area['maxlon'], mask.shape[2], dtype=np.float32)
        lats = np.linspace(area['minlat'],area['maxlat'], mask.shape[1], dtype=np.float32)
        high = np.linspace(area['minhigh'],area['maxhigh'],mask.shape[0], dtype=np.float32)
        high,lats,lons = np.meshgrid(high,lats,lons,indexing='ij')
        lons = lons[mask]*scale_x
        lats = lats[mask]*scale_y
        high = high[mask]*scale_z
        lons,lats,high = vectorized_geographic_to_cartesian(lons,lats,high)
        print(" 1.已转化计算笛卡尔坐标")
    points_coord = np.column_stack((lons,lats,high)).astype(np.float32)
    print(" 2.组合坐标点完成")
    del lons,lats,high
    gc.collect()
    return points_coord,point_size

def create_color_lookup(color_scheme="gray"):
    lookup_table = vtk.vtkLookupTable()
    lookup_table.SetNumberOfTableValues(256)
    lookup_table.SetRange(-32, 20)  # 反射率的典型范围
    if color_scheme == "gray":
        # 灰度方案
        gray = -(np.linspace(0.38, 1, 256)-1)**2+1#控制颜色
        #alph_range = np.linspace(0.05, 1, 256)**3#控制透明度
        alpha = np.linspace(0, 1, 256)**2.5#控制透明度
        light=list(range(256))#控制透亮度
        for i in range(256):  # 将 i 从 0 到 255 进行循环np.linspace(0, 1, 256)
            # 设置 RGB 和透明度
            lookup_table.SetTableValue(light[i], gray[i], gray[i], gray[i], alpha[i])
    
    elif color_scheme == "spectral":
        # 光谱色方案
        colors = [(0, 0, 0),
                (0.0, 0.0, 1.0),      # 纯蓝
                (0.0, 0.33, 0.67),    # 蓝绿色调
                (0.0, 0.67, 0.33),    # 青绿色调
                (0.0, 1.0, 0.0),      # 纯绿
                (0.33, 0.67, 0.0),    # 黄绿色调
                (0.67, 0.33, 0.0),    # 橙黄色调
                (1.0, 1.0, 0.0),      # 纯黄
                (1.0, 0.75, 0.0),     # 金黄
                (1.0, 0.5, 0.0),      # 橙色
                (1.0, 0.33, 0.0),     # 红橙色
                (1.0, 0.17, 0.0),     # 橘红
                (1.0, 0.0, 0.0),      # 纯红
                (0.9, 0.0, 0.0),      # 暗红
                (0.7, 0.0, 0.35),     # 红紫色 (修改点1: 红色减少，蓝色分量引入)
                (0.4, 0.0, 0.4),      # 标准紫色 (修改点2: 红色与蓝色分量平衡)
                ]
        # 创建颜色插值
        num_colors = len(colors)
        color_positions = np.linspace(0, 1, num_colors)
        alpha = np.linspace(0.12, 0.5, 256) ** 2.5
        
        for i in range(256):
            pos = i / 255.0
            # 找到对应的颜色区间
            for j in range(num_colors - 1):
                if color_positions[j] <= pos <= color_positions[j + 1]:
                    t = (pos - color_positions[j]) / (color_positions[j + 1] - color_positions[j])
                    r = colors[j][0] + t * (colors[j + 1][0] - colors[j][0])
                    g = colors[j][1] + t * (colors[j + 1][1] - colors[j][1])
                    b = colors[j][2] + t * (colors[j + 1][2] - colors[j][2])
                    break
            else:
                r, g, b = colors[-1]
            #lookup_table.SetTableValue(i, r, g, b, 0.1)
            lookup_table.SetTableValue(i, r, g, b, alpha[i])
    
    else:
        raise ValueError(f"不支持的色彩方案: {color_scheme}")
    lookup_table.Build()
    return lookup_table

def Create_render_method(poly_data, point_size, method='PointGaussian'):
    mapper = vtk.vtkPointGaussianMapper()
    mapper.SetInputData(poly_data)
    mapper.SetScaleFactor(point_size)
    mapper.SetEmissive(False)
    mapper.SetScalarVisibility(True)
    mapper.SetScalarRange(-32,20)
    return mapper


def create_volume_actor(reflectivity,area=None):
    print("\n===== 创建数据Actor =====")
    z,y,x = reflectivity.shape
    print(f"原始数据维度: {z},{y},{x}")
    print(f"NaN值比例: {np.isnan(reflectivity).mean() * 100:.2f}%")
    print('第一步：开始处理数据...')
    #创建掩码：过滤NaN值
    print(" 1.创建非NaN值掩码")
    mask = ~np.isnan(reflectivity)
    print(" 2.获统计有效点的数量")
    #valid_indices = np.argwhere(valid_mask)
    valid_count = np.count_nonzero(mask)
    print(f"  有效点数量: {valid_count} (占总点数的 {valid_count / reflectivity.size * 100:.2f}%)")
    print(" 3.将reflectivity转化为vtk格式数据")
    #reflectivity = reflectivity[mask].astype(np.float32)
    reflectivity = numpy_support.numpy_to_vtk(reflectivity[mask], deep=True)
    reflectivity.SetName("Reflectivity")
    #获取有效点的索引
    print('第二步：开始坐标转换...')
    points_coord,point_size = data_process(mask,area)
    print('第三步：转化为vtkPolyData数据...')
    print(" 1.创建点云数据")
    points = vtk.vtkPoints()
    points.SetData(numpy_support.numpy_to_vtk(points_coord, deep=True))
    print(" 2.创建多边形数据")
    poly_data = vtk.vtkPolyData()
    poly_data.SetPoints(points)
    print(" 3.添加反射率数据")
    poly_data.GetPointData().SetScalars(reflectivity)
    print(" 4.创建顶点单元，为每个点创建一个顶点单元")
    cells = vtk.vtkCellArray()
    cells.AllocateEstimate(valid_count, 1)
    poly_data.SetVerts(cells)
    '''
    glyph_filter = vtk.vtkVertexGlyphFilter()
    glyph_filter.SetInputData(poly_data)
    glyph_filter.Update()
    poly_data = glyph_filter.GetOutput()
    '''
    print(" 5.vtkPolyData数据完成")
    print('第四步：创建GPU加速的点云渲染器...')
    print(" 1.创建渲染器")
    mapper = Create_render_method(poly_data, point_size, method='PointGaussian')
    print(" 2.创建颜色查找表")
    lookup_table = create_color_lookup(color_scheme="gray")
    mapper.SetLookupTable(lookup_table)
    mapper.SelectColorArray("Reflectivity")
    mapper.SetScalarVisibility(True)
    print(" 3.创建Actor")
    actor = vtk.vtkLODActor()
    #actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    print("=======================\n")
    return actor
"""
def create_volume_actor(reflectivity,area=None):
    print("\n===== 创建数据Actor =====")
    print(f"原始数据维度: z={reflectivity.shape[0]}, y={reflectivity.shape[1]}, x={reflectivity.shape[2]}")
    print(f"NaN值比例: {np.isnan(reflectivity).mean() * 100:.2f}%")
    print('第一步：开始处理数据...')
    #创建掩码：过滤NaN值
    print(" 1.创建非NaN值掩码")
    valid_mask = ~np.isnan(reflectivity)
    print(" 2.获取有效点索引")
    valid_indices = np.argwhere(valid_mask)
    valid_count = np.count_nonzero(valid_mask)
    print(f"有效点数量: {valid_count} (占总点数的 {valid_count / reflectivity.size * 100:.2f}%)")
    #获取有效点的索引
    print('第二步：开始坐标转换...')
    if area is None:
        point_size=0.5
        # x和y方向间隔为1
        scale_x, scale_y ,scale_z = 1.0,1.0,3.0 # 例如，如果希望z方向间距为5，则设为5.0
        # 获取有效点索引并缩放
        x_coords = valid_indices[:, 2] * scale_x
        y_coords = valid_indices[:, 1] * scale_y
        z_coords = valid_indices[:, 0] * scale_z
       
    else:
        point_size=350
        i_idx, j_idx, k_idx = valid_indices[:, 2], valid_indices[:, 1], valid_indices[:, 0]
        # 地理坐标范围
        lon_min, lon_max = area['minlon'],area['maxlon']
        lat_min, lat_max = area['minlat'],area['maxlat']
        alt_min, alt_max = area['minhigh'],area['maxhigh']  # 高度范围 (单位:米)
        # 3. 创建坐标数组
        print(" 1.创建坐标数组")
        lons = np.linspace(lon_min, lon_max, reflectivity.shape[2], dtype=np.float32)
        lats = np.linspace(lat_min, lat_max, reflectivity.shape[1], dtype=np.float32)
        alts = np.linspace(alt_min, alt_max, reflectivity.shape[0], dtype=np.float32)
        # 4. 提取有效点的坐标
        print(" 2.提取有效点坐标")
        valid_lons = lons[i_idx]
        valid_lats = lats[j_idx]
        valid_alts = alts[k_idx]
        print(" 3.计算笛卡尔坐标")
        x_coords, y_coords, z_coords = vectorized_geographic_to_cartesian(valid_lons,valid_lats,valid_alts)
        del valid_lons, valid_lats, valid_alts
        gc.collect()
    del valid_indices 
    gc.collect()
    print(" 4.组合坐标点")
    points_array = np.column_stack((x_coords, y_coords, z_coords)).astype(np.float32)
    del x_coords, y_coords, z_coords
    gc.collect()
    print('坐标转换完成')
    print('第三步：转化为vtkPolyData数据...')
    print(" 1.创建点云数据")
    points = vtk.vtkPoints()
    points.SetData(numpy_support.numpy_to_vtk(points_array, deep=True))
    del points_array
    gc.collect()
    print(" 2.创建多边形数据")
    poly_data = vtk.vtkPolyData()
    poly_data.SetPoints(points)
    del points
    gc.collect()
    print(" 3.添加标量数据")
    valid_values = reflectivity[valid_mask].astype(np.float32)
    vtk_array = numpy_support.numpy_to_vtk(valid_values, deep=True)
    vtk_array.SetName("Reflectivity")
    poly_data.GetPointData().SetScalars(vtk_array)
    print(" 4.创建顶点单元，为每个点创建一个顶点单元")
    cells = vtk.vtkCellArray()
    for i in range(valid_count):
        cells.InsertNextCell(1)
        cells.InsertCellPoint(i)
    poly_data.SetVerts(cells)
    print(" 5.vtkPolyData数据完成")
    print('第四步：创建GPU加速的点云渲染器...')
    print(" 1.创建渲染器")
    mapper = vtk.vtkPointGaussianMapper()
    mapper.SetInputData(poly_data)
    del poly_data
    gc.collect()
    mapper.SetScaleFactor(point_size)  # 初始点大小
    mapper.SetScalarRange(np.min(valid_values), np.max(valid_values))
    mapper.SetEmissive(False)  # 自发光效果
    print(" 2.创建颜色查找表")
    lookup_table = vtk.vtkLookupTable()
    lookup_table.SetNumberOfTableValues(256)
    lookup_table.SetRange(-30, 20)
    # 设置灰色到白色的颜色和透明度
    '''
    gray = -(np.linspace(0.4, 1, 256)-1)**2+1#控制颜色
    #alph_range = np.linspace(0.05, 1, 256)**3#控制透明度
    alpha = np.linspace(0, 1, 256)**2.5#控制透明度
    light=list(range(256))#控制透亮度
    for i in range(256):  # 将 i 从 0 到 255 进行循环np.linspace(0, 1, 256)
        # 设置 RGB 和透明度
        lookup_table.SetTableValue(light[i], gray[i], gray[i], gray[i], alpha[i]) 
    '''
    colors = [(0, 0, 0),
            (0.0, 0.0, 1.0),      # 纯蓝
            (0.0, 0.33, 0.67),    # 蓝绿色调
            (0.0, 0.67, 0.33),    # 青绿色调
            (0.0, 1.0, 0.0),      # 纯绿
            (0.33, 0.67, 0.0),    # 黄绿色调
            (0.67, 0.33, 0.0),    # 橙黄色调
            (1.0, 1.0, 0.0),      # 纯黄
            (1.0, 0.75, 0.0),     # 金黄
            (1.0, 0.5, 0.0),      # 橙色
            (1.0, 0.33, 0.0),     # 红橙色
            (1.0, 0.17, 0.0),     # 橘红
            (1.0, 0.0, 0.0),      # 纯红
            (0.9, 0.0, 0.0),      # 暗红
            (0.7, 0.0, 0.35),     # 红紫色 (修改点1: 红色减少，蓝色分量引入)
            (0.4, 0.0, 0.4),      # 标准紫色 (修改点2: 红色与蓝色分量平衡)
            ]
    
    # 创建颜色插值
    num_colors = len(colors)
    color_positions = np.linspace(0, 1, num_colors)
    # 设置透明度 - 保持原来的透明度曲线
    #alpha = np.linspace(0, 1, 256)**2.5
    alpha = np.linspace(0.12, 0.5, 256)**2.5#控制透明度
    #alpha = np.linspace(0, 0.2, 256)#控制透明度
    # 为256个颜色值创建插值
    for i in range(256):
        pos = i / 255.0  # 位置在0-1之间
        # 找到当前位置在哪个颜色区间
        for j in range(num_colors - 1):
            if color_positions[j] <= pos <= color_positions[j + 1]:
                # 线性插值
                t = (pos - color_positions[j]) / (color_positions[j + 1] - color_positions[j])
                r = colors[j][0] + t * (colors[j + 1][0] - colors[j][0])
                g = colors[j][1] + t * (colors[j + 1][1] - colors[j][1])
                b = colors[j][2] + t * (colors[j + 1][2] - colors[j][2])
                break
        else:
            # 如果超出范围，使用最后一个颜色
            r, g, b = colors[-1]
        # 设置颜色和透明度
        #lookup_table.SetTableValue(i, r, g, b, 0.4)
        lookup_table.SetTableValue(i, r, g, b, alpha[i])
    
    lookup_table.Build()
    mapper.SetLookupTable(lookup_table)
    mapper.SelectColorArray("Reflectivity")
    mapper.SetScalarVisibility(True)
    print(" 3.创建Actor")
    actor = vtk.vtkLODActor()
    #actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    print("=======================\n")
    return actor

"""
def create_camera_info_actor():
    # 创建文本Actor来显示相机信息
    text_actor = vtk.vtkTextActor()
    text_actor.SetPosition(10, 10)  # 左下角位置
    text_actor.GetTextProperty().SetFontFamilyToArial()
    text_actor.GetTextProperty().SetFontSize(14)
    text_actor.GetTextProperty().SetColor(1, 1, 1)  # 白色文本
    text_actor.GetTextProperty().SetBackgroundColor(0.2, 0.2, 0.2)  # 深灰背景
    text_actor.GetTextProperty().SetBackgroundOpacity(0.7)
    return text_actor

def create_text_actor(text,text_size):
    """
    创建居中显示的文本Actor
    
    参数:
        text (str): 显示的文本内容
        position (tuple): 文本中心位置 (x, y)，使用归一化坐标 (0.0-1.0)
        font_size (int): 字体大小
        text_color (tuple): 文本颜色 (R, G, B)
        bg_color (tuple): 背景颜色 (R, G, B)
        bg_opacity (float): 背景透明度 (0.0-1.0)
    
    返回:
        vtk.vtkTextActor: 配置好的文本Actor
    """
    # 创建文本Actor
    text_actor = vtk.vtkTextActor()
    text_actor.SetInput(text)
    # 获取文本属性
    text_property = text_actor.GetTextProperty()
    # 设置文本居中显示
    text_property.SetJustificationToCentered()      # 水平居中
    text_property.SetVerticalJustificationToCentered()  # 垂直居中
    # 设置字体属性
    text_property.SetFontFamilyToArial()
    text_property.SetFontSize(text_size)#字体大小
    text_property.SetColor(0,0,0)#文本颜色
    text_property.SetBold(True)#设置字体粗细
    # 设置文本位置（归一化坐标）
    text_actor.SetPosition2(0.5, 0.95)#平面位置
    # 使用SetPositionCoordinate设置在窗口上部中心位置
    coord = text_actor.GetPositionCoordinate()
    coord.SetCoordinateSystemToNormalizedViewport()
    coord.SetValue(0.5, 0.98)
    return text_actor

def createa_render(volume_actor,map_actor):
    # 创建一个渲染器、渲染窗口和互器
    renderer = vtk.vtkRenderer()
    renderer.SetBackground(1.0, 1.0, 1.0)  # 浅蓝色背景
    # 创建文本Actor
    camera_text_actor = create_camera_info_actor()
    renderer.AddActor2D(camera_text_actor)
    # 全方位光源设置 - 从六个主要方向照射
    light_positions = [
        (0, 10, 0),   # 右 观看方向(x,y,z) x左右，y上下，z里外
        (0, -10, 0),  # 左
        (10, 0, 0),   # 上侧
        (-10, 0, 0),  # 下侧
        (0, 0, 10),    # 屏幕外
        (0, 0, -10)    # 屏幕里
    ]
    for i, pos in enumerate(light_positions):
        light = vtk.vtkLight()
        light.SetPosition(*pos)
        light.SetFocalPoint(0, 0, 0)
        light.SetColor(1.0, 1.0, 1.0)
        light.SetIntensity(1)  # 每个光源中等强度
        renderer.AddLight(light)
        #print(f"添加光源 {i+1} 在位置 {pos}")
    # 添加actor到渲染器
    renderer.AddActor(volume_actor)
    #renderer.AddVolume(volume)
    renderer.AddActor(map_actor)
    del volume_actor,map_actor
    gc.collect()
    return renderer, camera_text_actor

class KeyPressCallback:
    def __init__(self, screenshot_path, render_window, render_window_interactor):
        """
        初始化键盘回调类
        screenshot_path: 截图保存路径
        render_window_interactor: VTK窗口交互器
        render_window: VTK渲染窗口
        """
        self.screenshot_path = screenshot_path
        self.render_window_interactor = render_window_interactor
        self.render_window = render_window
        print("\n===== 交互控制说明 =====")
        print("鼠标左键拖动: 旋转场景")
        print("鼠标右键拖动: 平移场景")
        print("滚轮: 缩放场景")
        print("S键: 保存截图")
        print("ESC键: 退出程序")
        print("=======================\n")
        
    def __call__(self, obj, event):
        """回调函数逻辑 (VTK 事件回调接口)"""
        # 获取按下的键
        key = obj.GetKeySym()
        if key == "s" or key == "S":
            # 保存截图
            screenshot_file = f"{self.screenshot_path}/screenshot.png"
            w2if = vtk.vtkWindowToImageFilter()
            w2if.SetInput(self.render_window)
            w2if.SetScale(2)  # 高质量截图
            w2if.Update()
            writer = vtk.vtkPNGWriter()
            writer.SetFileName(screenshot_file)
            writer.SetInputData(w2if.GetOutput())
            writer.Write()
            print(f"截图已保存至: {screenshot_file}")
        elif key == "Escape":
            # 退出程序
            print("程序退出")
            self.render_window_interactor.TerminateApp()
        
# 自定义交互器样式类
class CameraInteractorStyle(vtk.vtkInteractorStyleTrackballCamera):
    def __init__(self, renderer, text_actor):
        self.AddObserver("MouseMoveEvent", self.mouse_move_event)
        self.renderer = renderer
        self.text_actor = text_actor
        self.update_camera_info()
    
    def mouse_move_event(self, obj, event):
        # 调用父类方法处理默认的鼠标交互
        self.OnMouseMove()
        # 更新相机参数显示
        self.update_camera_info()
    
    def update_camera_info(self):
        camera = self.renderer.GetActiveCamera()
        pos = camera.GetPosition()
        focal = camera.GetFocalPoint()
        view_up = camera.GetViewUp()
        
        # 格式化相机参数信息
        info = f"Camera Position: ({pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f})\n"
        info += f"Focal Point: ({focal[0]:.2f}, {focal[1]:.2f}, {focal[2]:.2f})\n"
        info += f"View Up: ({view_up[0]:.2f}, {view_up[1]:.2f}, {view_up[2]:.2f})"
        
        # 更新文本显示
        self.text_actor.SetInput(info)
        self.text_actor.GetTextProperty().SetFontSize(14)
        self.text_actor.GetTextProperty().SetColor(1, 1, 1)  # 白色文本
        self.text_actor.GetTextProperty().SetBackgroundColor(0.2, 0.2, 0.2)  # 深灰背景
        self.text_actor.GetTextProperty().SetBackgroundOpacity(0.7)
        # 请求重新渲染
        self.renderer.GetRenderWindow().Render()

# 创建渲染窗口
def setup_window(renderer,back_size,OffScreen=False):
    #创建渲染窗口
    render_window = vtk.vtkRenderWindow()
    render_window.SetWindowName("3D Weather Visualization")
    render_window.SetSize(int(back_size['length']), int(back_size['width']))  # 增加分辨率以获得更清晰的截图
    render_window.AddRenderer(renderer)
    del renderer
    gc.collect()
    if OffScreen:
        render_window.SetOffScreenRendering(True)#设置离屏渲染
    return render_window

# 创建交互器
def setup_interactor(render_window,output_path,renderer=None,text_actor=None,):
    # 创建交互器
    render_window_interactor = vtk.vtkRenderWindowInteractor()
    render_window_interactor.SetRenderWindow(render_window)
    # 设置交互风格为TrackballCamera（支持左键拖动旋转）
    if text_actor is None and renderer is None:
        style = vtk.vtkInteractorStyleTrackballCamera()
    else:
        style = CameraInteractorStyle(renderer, text_actor)
    render_window_interactor.SetInteractorStyle(style)
    # 添加键盘回调
    #render_window_interactor.AddObserver("KeyPressEvent", keypress_callback)
    callback = KeyPressCallback(output_path,render_window ,render_window_interactor)
    render_window_interactor.AddObserver("KeyPressEvent",callback)
    return render_window_interactor

#  设置视角
class CameraManager:
    def __init__(self, output_path,Earth=False, take_snapshot=False):
        self.output_path = output_path
        os.makedirs(output_path, exist_ok=True)
        self.Earth = Earth
        self.take_snapshot = take_snapshot
    def configure_camera(self,renderer,render_window,view,name=None):
        camera = renderer.GetActiveCamera()
        if self.Earth:
            # 计算焦点的笛卡尔坐标（地球表面点）
            center_lon, center_lat, camera_high = view["camera_position"]
            x, y, z = vectorized_geographic_to_cartesian(center_lon, center_lat, camera_high)
            # 设置相机位置和焦点
            camera.SetPosition(x, y, z)
            camera.SetFocalPoint(0,0,0)# 设置焦点在场景中心
            camera.SetViewUp(0, 0, 1)# 设置上方向（始终指向北极）
            #camera.SetViewAngle(90)
        else:
            # 使用自定义相机位置
            camera.SetPosition(view["camera_position"])
            camera.SetFocalPoint(view["focal_point"])  # 设置焦点在场景中心
            camera.SetViewUp(view["view_up"])  # 设置相机上方向
        '''
        # 使用自定义相机位置
        camera.SetPosition(view["camera_position"])
        camera.SetFocalPoint(view["focal_point"])  # 设置焦点在场景中心
        camera.SetViewUp(view["view_up"])  # 设置相机上方向
        '''
        # 通用相机设置
        #camera.Zoom(1)  # 应用缩放
        renderer.ResetCameraClippingRange()
        # 渲染
        render_window.Render()
        #render_window.SwapBuffers() # 交换缓冲区（阻塞直到垂直同步）
        render_window.WaitForCompletion()
        # 如果需要截图
        if self.take_snapshot:
            """内部方法：捕获并保存渲染窗口截图"""
            render_window.Render()  # 再渲染一次
            render_window.WaitForCompletion()  # 等待渲染完成
            # 创建截图过滤器
            window_to_image = vtk.vtkWindowToImageFilter()
            window_to_image.SetInput(render_window)
            window_to_image.SetInputBufferTypeToRGB()
            window_to_image.ReadFrontBufferOff()  # 使用后缓冲避免闪烁
            window_to_image.SetScale(1) 
            window_to_image.Update()
            if name is None:
                name=view['name']
            # 创建文件名（包含场景名和视角名）
            filename = f"{self.output_path}/{name}.png"
            # 保存为PNG
            writer = vtk.vtkPNGWriter()
            writer.SetFileName(filename)
            writer.SetInputConnection(window_to_image.GetOutputPort())
            writer.Write()
            print(f"已保存快照: {filename}")
        return render_window

def crop_borders(image, tolerance=10):
    """自动裁剪图像周围的空白区域"""
    img_array = np.array(image.convert('RGB'))
    height, width, _ = img_array.shape
    
    # 检测背景色（取四个角的平均颜色）
    corners = [
        img_array[0, 0], img_array[0, -1], 
        img_array[-1, 0], img_array[-1, -1]]
    bg_color = np.mean(corners, axis=0).astype(int)
    
    # 创建掩码检测非背景区域
    mask = np.any(np.abs(img_array - bg_color) > tolerance, axis=2)
    
    # 找到边界
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    
    top, bottom = np.where(rows)[0][[0, -1]]
    left, right = np.where(cols)[0][[0, -1]]
    
    return image.crop((left, top, right + 1, bottom + 1))

def Generate_GIF(image_path,keywords,frame_duration):
    name="_".join(keywords)
    output_gif = f"{image_path}/{name}.gif"  # 输出 GIF 文件名
    #keyword = "angle_view"  # 文件名关键词
    #frame_duration = 100  # 每帧显示时间（毫秒）
    # 获取所有符合条件的 PNG 文件
    #image_files = [f for f in os.listdir(image_path)if f.endswith('.png') and keyword in f]
    image_files = []
    for f in os.listdir(image_path):
        if f.endswith('.png'):
            # 使用all()函数检查所有关键词
            if all(keyword in f for keyword in keywords):
                image_files.append(f)
            else:
                continue
        else:
            continue
        #print(f)
    # 按文件名排序
    image_files.sort()
    # 打开所有图片并存入列表
    images = []
    for img_file in image_files:
        img_path = os.path.join(image_path, img_file)
        try:
            img = Image.open(img_path)
            images.append(img.copy())  # 复制图像对象
            img.close()  # 关闭文件
        except Exception as e:
            print(f"无法打开 {img_file}: {str(e)}")
    # 检查是否找到图片
    if not images:
        print("未找到符合条件的图片！")
        exit()
    # 保存为动态 GIF
    images[0].save(output_gif,save_all=True,
                   append_images=images[1:],  # 添加剩余帧
                   duration=frame_duration,
                   loop=0,  # 0 表示无限循环
                   optimize=True)
    print(f"成功生成 GIF: {output_gif},包含 {len(images)} 帧")

def add_title_to_image(input_image_path, output_image_path, title, 
                      title_size=30, title_margin=5, bold=True):
    # 打开原始图片
    img = Image.open(input_image_path)
    
    # 剪裁掉四周空白部分
    img_cropped = crop_borders(img, tolerance=10)
    
    # 创建一个新的图像，向上添加空白部分
    new_height = img_cropped.height + title_size + title_margin + 15
    new_img = Image.new("RGB", (img_cropped.width, new_height), (255, 255, 255))
    
    # 将剪裁后的图片粘贴到新图像的下方
    new_img.paste(img_cropped, (0, title_size + title_margin + 10))
    
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