# -*- coding: utf-8 -*-
"""
Created on Wed Apr 16 13:55:41 2025
@author: 59278
"""
import numpy as np
import FUNC_plot_image as plot
import FUNC_read_data as read
from scipy import signal
import transbigdata as tbd
#file_path="D:/result/3D_RAGASA/Gen_RAGASA_20250924_0220_2km.nc"
file_path="D:/result/3D_RAGASA/Gen_RAGASA_20250924_0220_2km_infrared.nc"
image_save="D:/images"

#%%
'''MODIS CWP'''
MOD06="D:/data/Modis/MOD06_L2.A2025267.0215.061.2025267154706.hdf"
MOD03="D:\data\Modis\MOD03.A2025267.0215.061.2025267091636.hdf"
area={'minlon':100,
     'maxlon':130,
     'minlat':10,
     'maxlat':30}

CWP=read.read_source_data(MOD06,'Cloud_Water_Path',dtype=np.float32)
CWP=np.log10(CWP)[:,2:-2]
latMOD=read.read_source_data(MOD03,'Latitude',dtype=np.float32)[:,2:-2]
lonMOD=read.read_source_data(MOD03,'Longitude',dtype=np.float32)[:,2:-2]
lon_latMOD = np.stack((lonMOD,latMOD))
data_name=['Modis CWP 2025-09-24 02:15_color']
PLOT=plot.Geographical(image_save,area,line=90)
PLOT.images([CWP],[lon_latMOD],data_name,'Modis_CWP_20250924_0215',MinMax=[0,3.5])

#%%
'''3DRF to LWP'''
def get_index(varb,area,resolution):
    params = tbd.area_to_params([area['minlon'],area['minlat'],area['maxlon'],area['maxlat']],
                                accuracy=1000*resolution)
    params['deltalon']=0.01
    params['deltalat']=0.01
    index=np.empty(varb.shape)
    index[:,0],index[:,1] = tbd.GPS_to_grid(varb[:,0], varb[:,1],params)
    return index.astype(np.int16)

CWP_border=np.ones(CWP.shape)
CWP_border[12:-12,4:-4]=np.nan
lon_latMOD = np.stack((lonMOD[~np.isnan(CWP_border)],latMOD[~np.isnan(CWP_border)]))
mask=(latMOD>area['minlat']) & (latMOD<area['maxlat']) & (lonMOD>area['minlon']) & (lonMOD<area['maxlon'])
location =np.stack((lonMOD[mask],latMOD[mask]), axis=-1)
index=get_index(location,area,1)
dbz=read.read_source_data(file_path,"out",dtype=np.float32)
z=10**(dbz/10)
lwc=1.04*(z**(0.5))
lwc[np.isnan(lwc)]=0
lwp=np.nansum(lwc,axis=0)
lwp[lwp<1]=np.nan
lwp=np.log10(lwp)

lon=np.arange(area['minlon']-0.02,area['maxlon'],0.01)
lat=np.arange(area['minlat'],area['maxlat']+0.02,0.01)
lon_lat = np.stack(np.meshgrid(lon,lat))
#indexfinal=index[(index[:,0] < len(lon)) & (index[:, 0] > 0) & (index[:, 1] < len(lat)) & (index[:, 1] > 0)]
#indexfinal=index[(index[:,0] < len(lon)) & (index[:, 0] > 0) & (index[:, 1] < len(lat)) & (index[:, 1] > 0)]
alpha=np.zeros(lwp.shape)
alpha[index[:,1],index[:,0]]=1
kernel = np.ones((5,5))
alpha=signal.convolve2d(alpha, kernel, mode='same')
alpha=np.where((alpha==0),0.5,1)
data_name=['CRR-LDM-Full_color','CRR-LDM-Full_Path']
PLOT=plot.Geographical(image_save,area,line=90)
PLOT.images([lwp,None],[lon_lat,lon_latMOD],data_name,'lwp_20250924_0220_Full',MinMax=[0,2.5],alpha=alpha)

#%%计算SSIM
from skimage.metrics import structural_similarity as ssim
CWP_modis = np.zeros(lwp.shape)
CWP_modis[index[:,1],index[:,0]]=CWP[mask][:]
CWP_modis[np.isnan(CWP_modis)]=0.0

LWP_Himawrai=np.zeros(lwp.shape)
LWP_Himawrai[index[:,1],index[:,0]]=lwp[index[:,1],index[:,0]]
LWP_Himawrai[np.isnan(LWP_Himawrai)]=0.0
cwp_norm = (CWP_modis[104:2000,0:2091] - 0) / 3.5
lwp_norm = (LWP_Himawrai[104:2000,0:2091] - 0) / 2.5
ssim_out=ssim(cwp_norm, lwp_norm,win_size=51, data_range=1.0)
#print(f"ssim:{ssim_out}")
#%%计算其他指标
CWP_nonan = np.where(np.isnan(CWP),0.0,CWP)
CWP_to_Hshape = np.full(lwp.shape, np.nan)
CWP_to_Hshape[index[:,1],index[:,0]]=CWP_nonan[mask][:]
mask2=~np.isnan(CWP_to_Hshape)
LWP_Himawrai=np.where(np.isnan(lwp),0.0,lwp)
#LWP_Himawrai=np.where(mask2,LWP_Himawrai,np.nan)
LWP_Himawrai=LWP_Himawrai[mask2]
CWP_modis=CWP_to_Hshape[mask2]


cwp_norm = CWP_modis
lwp_norm = LWP_Himawrai
#cwp_norm = (CWP_modis - 0) / 3.5
#lwp_norm = (LWP_Himawrai - 0) / 2.5


from scipy.stats import spearmanr
spearmanr_corr, p = spearmanr(cwp_norm, lwp_norm)
#print(f"spearmanr: {spearmanr_corr:.4f}")

from scipy.stats import pearsonr
pearsonr_corr, p = pearsonr(cwp_norm, lwp_norm)
#print(f"pearsonr: {pearsonr_corr:.4f}")

"""互信息"""
from sklearn.metrics import mutual_info_score
mi = mutual_info_score(cwp_norm, lwp_norm)  # 或先用 np.histogramdd 自定义分箱
#print(f"互信息mi: {mi:.4f}")

from sklearn.metrics import normalized_mutual_info_score
nmi = normalized_mutual_info_score(cwp_norm, lwp_norm)
#print(f"NMI: {nmi:.4f}")

from minepy import MINE
sample_idx = np.random.choice(len(cwp_norm), 50000, replace=False)
mine = MINE(alpha=0.6, c=15)#alpha=0.6, c=15
mine.compute_score(cwp_norm[sample_idx], lwp_norm[sample_idx])
mic_result = mine.mic()
#print(f"最大互信息系数 (MIC) 为: {mic_result:.4f}")


print(f"ssim: {ssim_out:.4f}")
print(f"spearmanr: {spearmanr_corr:.4f}")
print(f"pearsonr: {pearsonr_corr:.4f}")
print(f"互信息mi: {mi:.4f}")
print(f"NMI: {nmi:.4f}")
print(f"最大互信息系数 (MIC) 为: {mic_result:.4f}")