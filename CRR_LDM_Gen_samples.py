# -*- coding: utf-8 -*-
"""
Created on Sun Jun  8 13:31:16 2025

@author: 59278
"""

import os
#os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import numpy as np
import FUNC_read_data as read
import FUNC_analyse_data as analyse
import tensorflow as tf
import X_Latent_DDIM_UNet_2km as DDIM_UNet
import X_VAE_model_2km as VAE

#设置扩散步长
timestep=500
denoise_step=5
var_name=['albedo_01','albedo_02','albedo_03','albedo_04','albedo_05','albedo_06',
          'tbb_07','tbb_08','tbb_09','tbb_10','tbb_11','tbb_13','tbb_14',
          'tbb_15','tbb_16','SOZ']
tar_name=['cloud_scenario','Radar_Reflectivity']
#%%
test_files= ['2020_Himawari_cloudsat_128_cloud_SAZ']
data_path = "/work/home/acmh4zm9q3/Data_Train"
save_name = "Full_GEN_samples_2020"
save_path = "/work/home/acmh4zm9q3/Model_test"
DDIM_weight="/work/home/acmh4zm9q3/LDM_Out/weight/Latent_DDIM_final/Lat_DDIM_ema_time5_epoch29_loss0.312.weights.h5"
encoder_weight="/work/home/acmh4zm9q3/VAE_out/weight/kl_0001/encoder_time4_epochs29_loss416.884.weights.h5"
decoder_weight="/work/home/acmh4zm9q3/VAE_out/weight/kl_0001/decoder_time4_epochs29_loss416.884.weights.h5"
#%%
def make_dataset(data_path,data_files,var_name,random=None,trans=False):
    dicts = {key: [] for key in var_name}
    for file_name in data_files:
        file_path = data_path+'/'+file_name+'.nc'
        data_set = read.read_varible(file_path,var_name,random=random,trans=trans) 
        for key, value in data_set.items():
            dicts[key].append(value)
    data = {key: np.concatenate(value) for key, value in dicts.items()}
    print(f'样本数量{len(data[var_name[0]])}')
    return data

def generate_data(inputs,timestep,denoise_step):
    skip_timestep=timestep//denoise_step
    variable = tf.convert_to_tensor(inputs, dtype=tf.float32)
    noise = tf.random.normal(shape=(len(variable),32, 32, 1), dtype=tf.float32)
    samples = tf.clip_by_value(noise ,-3, 3)
    #samples = tf.zeros((len(variable), 32, 32, 1), dtype=tf.float32)
    progbar = tf.keras.utils.Progbar(timestep)
    for t in range(timestep-1,0,-skip_timestep):
        tk=t
        ts=t-skip_timestep
        if ts<0:
            ts=0
        print(tk,ts)
        t_k = tf.cast(tf.fill(variable.shape[0], tk), dtype=tf.int32)
        t_s = tf.cast(tf.fill(variable.shape[0], ts), dtype=tf.int32)
        pred_noise = network.predict([samples,t_k,variable], verbose=1, batch_size=160)
        samples = Diffusion.DDIM_denoise(pred_noise,samples,t_k,t_s,clip_denoised=True)
        progbar.update(timestep - t) 
    return samples
#%%
"""读取数据集"""
test_data=make_dataset(data_path,test_files,var_name+tar_name,trans=True)
test_target = read.scale_Reflect(test_data['Radar_Reflectivity'])
test_input  = read.scale_varible(test_data,var_name)
print('test_target:',test_target.shape,'test_input:',test_input.shape)
#%%
"""初始化VAE，加载权重"""
encoder=VAE.Encoder(test_target.shape[1], test_target.shape[2])
decoder=VAE.Decoder(32, 32)
encoder.load_weights(encoder_weight)
decoder.load_weights(decoder_weight)
#%%
"""VAE编码"""
mean,log_var,latent = encoder.predict(test_target, verbose=1, batch_size=160)
test_data['Tar_Latent']=latent[...,0]
test_data['Tar_mean']=mean[...,0]
test_data['Tar_logvar']=log_var[...,0]
del mean,log_var,latent
#%%
"""初始化DDIM，加载权重"""
Diffusion = DDIM_UNet.GaussianDiffusion(timesteps=timestep,clip_min=-3.0,clip_max=3.0,)
network = DDIM_UNet.U_Net(32,64,len(var_name))
network.load_weights(DDIM_weight)
#%%
"""DDIM生成"""
for i in range(30):
    Gen_latent = generate_data(test_input,timestep,denoise_step)
    Gen_result = decoder.predict(Gen_latent,verbose=1, batch_size=160)
    test_data['Gen_latent'] = Gen_latent.numpy()[...,0]
    test_data['Gen_result'] = read.rescale_gen(Gen_result)[...,0]
    del Gen_latent,Gen_result
    result_rmse = analyse.Count_remse(test_data['Gen_result'],test_data['Radar_Reflectivity'])
    result_mean,result_var=analyse.mean_var(result_rmse)
    print(f'result_mean:{result_mean}  result_var:{result_var}')
    del result_mean,result_var,result_rmse 
    #%%
    """保存数据"""
    read.save_data(test_data,save_path,f'{save_name}_{i}')
    print(f'数据已保存{save_name}_{i}')



