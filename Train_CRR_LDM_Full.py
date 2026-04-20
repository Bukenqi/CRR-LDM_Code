# -*- coding: utf-8 -*-
"""
Created on Mon Dec  2 18:42:27 2024

@author: 59278
"""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import glob
import numpy as np
#from tqdm import tqdm
import FUNC_plot_image as plot
import FUNC_read_data as read
import FUNC_analyse_data as analyse
import tensorflow as tf
import X_Latent_DDIM_UNet_2km as DDIM_UNet
import X_VAE_model_2km as VAE

#设置扩散步长
timestep=500
#训练次数
time=5
star=0
#选择数据集
var_name=['albedo_01','albedo_02','albedo_03','albedo_04','albedo_05','albedo_06','tbb_07',
       'tbb_08','tbb_09','tbb_10','tbb_11','tbb_12','tbb_13','tbb_14','tbb_15','tbb_16','SOZ'
]
tar_name=['cloud_scenario','Radar_Reflectivity']
#%%
train_files=['2016_Himawari_cloudsat_128_cloud_SAZ','2017_Himawari_cloudsat_128_cloud_SAZ','2018_Himawari_cloudsat_128_cloud_SAZ','2019_Himawari_cloudsat_128_cloud_SAZ']
nocloud_file=['2018_Himawari_cloudsat_128_nocloud_SAZ']
test_files=['2020_Himawari_cloudsat_128_cloud_SAZ']
save_file='Latent_DDIM_final'
data_path="/work/home/acmh4zm9q3/Data_Train"
image_save=f"/work/home/acmh4zm9q3/LDM_Out/images/{save_file}"
weight_save=f"/work/home/acmh4zm9q3/LDM_Out/weight/{save_file}"
result_save=f"/work/home/acmh4zm9q3/LDM_Out/result/{save_file}"
encoder_weight="/work/home/acmh4zm9q3/VAE_out/weight/kl_0001/encoder_time4_epochs29_loss416.884.weights.h5"
decoder_weight="/work/home/acmh4zm9q3/VAE_out/weight/kl_0001/decoder_time4_epochs29_loss416.884.weights.h5"

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
    #noise_sample = tf.random.normal(shape=(32, 32, 1), dtype=tf.float32)
    #samples = tf.repeat(tf.expand_dims(noise_sample, axis=0), repeats=variable.shape[0], axis=0)
    #samples = tf.zeros((len(variable), 32, 32, 1), dtype=tf.float32)
    noise = tf.random.normal(shape=(len(variable),32, 32, 1), dtype=tf.float32)
    samples = tf.clip_by_value(noise ,-3, 3)
    #samples = tf.random.truncated_normal(shape=(len(variable),32, 32, 1), mean=0.0, stddev=1.0)
    progbar = tf.keras.utils.Progbar(timestep)
    for t in range(timestep-1,0,-skip_timestep):
        tk=t
        ts=t-skip_timestep
        if ts<0:
            ts=0
        print(tk,ts)
        t_k = tf.cast(tf.fill(variable.shape[0], tk), dtype=tf.int32)
        t_s = tf.cast(tf.fill(variable.shape[0], ts), dtype=tf.int32)
        pred_noise =model.ema_network.predict([samples,t_k,variable], verbose=1, batch_size=8)
        samples = Diffusion.DDIM_denoise(pred_noise,samples,t_k,t_s,clip_denoised=True)
        progbar.update(timestep - t) 
    return samples

class DiffusionModel(tf.keras.Model):
    def __init__(self, network,ema_network,Diffusion,timesteps):
        super().__init__()
        self.network = network
        self.ema_network = ema_network
        #self.encoder = encoder
        #self.decoder = decoder
        self.timesteps = timesteps
        self.Diffusion = Diffusion
        self.ema = 0.999
    
    def compile(self, **kwargs):
        super().compile(**kwargs)
        #self.image_loss_fn = tf.keras.losses.Huber()
        #self.noise_loss_fn = tf.keras.losses.MeanAbsoluteError()
        #self.start_loss_fn = tf.keras.losses.MeanSquaredError()
        self.image_loss_fn = tf.keras.losses.MeanAbsoluteError()
        self.noise_loss_fn = tf.keras.losses.MeanAbsoluteError()
        self.start_loss_fn = tf.keras.losses.MeanAbsoluteError()
        self.noise_loss_tracker = tf.keras.metrics.Mean(name="noise_loss")
        self.image_loss_tracker = tf.keras.metrics.Mean(name="image_loss")
        self.start_loss_tracker = tf.keras.metrics.Mean(name="start_loss")
        self.total_loss_tracker = tf.keras.metrics.Mean(name="total_loss")

    def train_step(self, data):
        variables, target = data
        batch_size = tf.shape(variables)[0]
        t = tf.random.uniform(minval=0, maxval=self.timesteps, shape=(batch_size,), dtype=tf.int32)
        t_end = tf.zeros(shape=(batch_size,), dtype=tf.int32)
        t_star = tf.ones(shape=(batch_size,), dtype=tf.int32) * (self.timesteps - 1)  # 确保不越界
        with tf.GradientTape() as tape:
            # 计算noise_loss
            noise = tf.random.normal(shape=tf.shape(target), dtype=target.dtype)
            noise = tf.clip_by_value(noise, -3.0, 3.0)
            images_t = self.Diffusion.Add_noise(target, t, noise)
            pred_noise = self.network([images_t, t, variables], training=True)
            noise_loss = self.noise_loss_fn(noise, pred_noise)
            # 计算image_loss
            pred_target= self.Diffusion.DDIM_denoise(pred_noise,images_t,t,t_end,clip_denoised=True)
            image_loss = self.image_loss_fn(target, pred_target)
            # 使用images_star作为去噪起点
            images_star = tf.random.normal(shape=tf.shape(target), dtype=target.dtype)
            images_star = tf.clip_by_value(images_star, -3.0, 3.0)
            pred_noise_star = self.network([images_star, t_star, variables], training=True)
            pred_adnoise = self.Diffusion.DDIM_denoise(pred_noise_star, images_star, t_star, t, clip_denoised=True)
            start_loss = self.start_loss_fn(images_t, pred_adnoise)
            total_loss = image_loss+noise_loss+start_loss
        gradients = tape.gradient(total_loss, self.network.trainable_weights)
        self.optimizer.apply_gradients(zip(gradients, self.network.trainable_weights))

        self.noise_loss_tracker.update_state(noise_loss)
        self.image_loss_tracker.update_state(image_loss)
        self.start_loss_tracker.update_state(start_loss)
        self.total_loss_tracker.update_state(total_loss)
        # 9. Updates the weight values for the network with EMA weights
        for weight, ema_weight in zip(self.network.weights, self.ema_network.weights):
            ema_weight.assign(self.ema * ema_weight + (1 - self.ema) * weight)
        #return {"loss": loss}
        return {"loss": noise_loss,
                self.image_loss_tracker.name: self.image_loss_tracker.result(),
                self.noise_loss_tracker.name: self.noise_loss_tracker.result(),
                self.start_loss_tracker.name: self.start_loss_tracker.result(),
                self.total_loss_tracker.name: self.total_loss_tracker.result(),
        }

class SaveWeights(tf.keras.callbacks.Callback):
    def __init__(self, model, Weight_path, time):
        super(SaveWeights, self).__init__()
        self.network = model.network
        self.ema_network = model.ema_network
        self.Weight_path= Weight_path
        self.time = time  # New parameter n
        os.makedirs(Weight_path, exist_ok=True)

    def on_epoch_end(self, epoch, logs=None):
        #filepath = self.filepath_template.format(n=self.n, epoch=epoch, loss=logs['loss'])
        loss_value = logs.get("noise_loss", 0.0) if logs else 0.0
        self.network.save_weights(
            os.path.join(self.Weight_path, 
                        f'Lat_DDIM_time{self.time}_epoch{epoch}_loss{loss_value:.3f}.weights.h5'))
        # EMA 网络权重
        self.ema_network.save_weights(
            os.path.join(self.Weight_path, 
                         f'Lat_DDIM_ema_time{self.time}_epoch{epoch}_loss{loss_value:.3f}.weights.h5'))
        print(f'time{time}_epoch{epoch}权重已保存.')

class PlotImages(tf.keras.callbacks.Callback):
    def __init__(self,test_data,timesteps,time,result_save,):
        super(PlotImages, self).__init__()
        self.test_data=test_data
        self.timesteps=timesteps
        self.result_save=result_save
        self.time=time
    def on_epoch_end(self,epoch, logs=None):
        Pre_latent = generate_data(self.test_data['test_input'],self.timesteps,10)
        Pre_latent = Pre_latent.numpy()[...,0]
        latent_rmse = analyse.Count_remse(Pre_latent,self.test_data['Target_Latent'])
        latent_mean,latent_var=analyse.mean_var(latent_rmse)
        print('latent_mean',latent_mean,'latent_var',latent_var)
        
        pre_out = decoder.predict(Pre_latent ,verbose=1, batch_size=192)
        result = read.rescale_gen(pre_out)[...,0]
        result_rmse = analyse.Count_remse(result,self.test_data['Radar_Reflectivity'])
        result_mean,result_var=analyse.mean_var(result_rmse)
        print('result_mean',result_mean,'result_var',result_var)
        self.test_data['Latent_Diffusion']=result
        self.test_data['Predict_latent']=Pre_latent
        self.test_data['result_rmse']=result_rmse
        self.test_data['latent_rmse']=latent_rmse
        save_name=f'time{self.time}_epoch{epoch}_mean{result_mean:.2f}_var{result_var:.2f}'
        read.save_data(self.test_data,self.result_save,save_name)
        print(f'{save_name}.nc已保存')
        del Pre_latent,latent_rmse,latent_mean,latent_var
        del pre_out,result,result_rmse,result_mean,result_var
        #画图
        data_name=[['tbb_08_line','tbb_09_line'],
                   ['tbb_12_line','tbb_13_line'],
                   'cloud_class',
                   'cloudsat Reflectivity_Reflectivity',
                   'Latent Diffusion_Reflectivity',
                   'Target latent_Reflectivity',
                   'Predict latent_Reflectivity']
        data_list=[[self.test_data['tbb_08'],self.test_data['tbb_09']],
                   [self.test_data['tbb_12'],self.test_data['tbb_13']],
                   self.test_data['cloud_scenario'],
                   self.test_data['Radar_Reflectivity'],
                   self.test_data['Latent_Diffusion'],
                   self.test_data['Target_Latent'],
                   self.test_data['Predict_latent'],]
        hight_layer=[None,None,None,None,None,
                     np.arange(480, 480+32*480,480),
                     np.arange(480, 480+32*480,480)]
        PLOT.images(data_list,data_name,hight_layer=hight_layer,save_name=save_name)
        print(f'{save_name}图片已完成')
        del data_list

PLOT=plot.Comparison(image_save,resolution=2)
print('画图函数初始化完成')
test_data=make_dataset(data_path,test_files,var_name+tar_name,trans=True)
#test_data_nocloud=make_dataset(data_path,nocloud_file,var_name+tar_name,random=10,trans=True)
cloud_data=make_dataset(data_path,train_files,var_name+['Radar_Reflectivity'],trans=True)
nocloud_data=make_dataset(data_path,nocloud_file,var_name+['Radar_Reflectivity'],random=2000,trans=True)

train_data={}
for varname in var_name+['Radar_Reflectivity']:
    train_data[varname] = np.concatenate((cloud_data[varname], nocloud_data[varname]), axis=0)
    del cloud_data[varname], nocloud_data[varname]
train_target = read.scale_Reflect(train_data['Radar_Reflectivity'])
train_input = read.scale_varible(train_data,var_name)
del train_data,nocloud_data,cloud_data
print('训练集制已读取')
print('train_target:',train_target.shape)
print('train_input:',train_input.shape)

encoder=VAE.Encoder(train_target.shape[1], train_target.shape[2])
decoder=VAE.Decoder(32, 32)
encoder.load_weights(encoder_weight)
decoder.load_weights(decoder_weight)
_,_,train_target = encoder.predict(train_target,verbose=1,batch_size=192)
for i in range(2):
    train_target = np.concatenate((train_target, train_target), axis=0)
    train_input = np.concatenate((train_input, train_input), axis=0)
print('训练集制作已完成')
print('target形状',train_target.shape)
print('input形状',train_input.shape)

test_target = read.scale_Reflect(test_data['Radar_Reflectivity'])
test_input  = read.scale_varible(test_data,var_name)
print('测试集已读取')
print('test_target:',test_target.shape)
print('test_input:',test_input.shape)
mean,log_var,latent = encoder.predict(test_target, verbose=1, batch_size=4)
test_data['Target_Latent']=latent[...,0]
test_data['generate_mean']=mean[...,0]
test_data['generate_logvar']=log_var[...,0]
test_data['test_input']=test_input
print('测试集制作已完成')
print('latent形状',latent.shape)
print('test_input形状',test_input.shape)
del mean,log_var,latent,test_input,test_target
print('测试集多余内存已清理')
# 初始化高斯扩散实用程序的实例
Diffusion = DDIM_UNet.GaussianDiffusion(timesteps=timestep,clip_min=-3.0,clip_max=3.0,)
#初始化unet
print('模型初始化形状',train_target.shape[1],train_input.shape[1],train_input.shape[2])
network = DDIM_UNet.U_Net(train_target.shape[1],train_input.shape[1],train_input.shape[2])
ema_network = DDIM_UNet.U_Net(train_target.shape[1],train_input.shape[1],train_input.shape[2])
#组装模型 
model = DiffusionModel(network,ema_network,Diffusion,timestep)
# 编译模型
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=2e-4),jit_compile=True)
#model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=2e-4,beta_1=0.9,beta_2=0.999,epsilon=1e-7),jit_compile=True)

data = tf.data.Dataset.from_tensor_slices((train_input.astype(np.float32), train_target.astype(np.float32)))
data = data.cache()  # 缓存数据到内存（如果内存足够）
data = data.shuffle(buffer_size=len(train_input), reshuffle_each_iteration=True)
del train_input,train_target

'''训练模型'''
for n in range(star,time+1):
    #定义画图函数
    plot_callback = PlotImages(test_data,timestep,n,result_save)
    #plot_callback2 = PlotImages(decoder,test_data_nocloud,timestep,f'{n}_nocloud',result_save)
    #定义权重保存函数
    weight_callback = SaveWeights(model,weight_save,n)
    #获取文件夹中权重文件是否存在
    weight_files = glob.glob(os.path.join(weight_save, '*.h5'))
    if weight_files!=[]:
        print('加载权重')
        # 选择最新的权重文件
        latest_weight = max(weight_files, key=os.path.getctime)
        # 加载最新的权重到模型中
        model.network.load_weights(latest_weight)
        print(f"第{n}次训练")
    else:
        print("开始训练")
    # 拟合模型并保存权重
    batch_sizes=2**(n+2)
    print(f'batch_size:{batch_sizes}')
    dataset = data.batch(batch_sizes)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)# 使用并行处理和预取
    model.fit(dataset,epochs=30,callbacks=[weight_callback,plot_callback])
    #image_data(test_variables,test_real,timestep,image_save,result_save,time=n)
print('训练结束')