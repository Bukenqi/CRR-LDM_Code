# -*- coding: utf-8 -*-
"""
Created on Sun Nov 10 17:18:50 2024

@author: 59278
"""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import glob
import numpy as np
import FUNC_read_data as read
import FUNC_plot_image as plot
import FUNC_analyse_data as analyse
import tensorflow as tf
import X_VAE_model_2km as model

#训练次数
times=5
star=0
#选择数据集
var_name=['Radar_Reflectivity','cloud_scenario']
train_files=['2016_Himawari_cloudsat_128_cloud_SAZ','2017_Himawari_cloudsat_128_cloud_SAZ','2018_Himawari_cloudsat_128_cloud_SAZ','2019_Himawari_cloudsat_128_cloud_SAZ']
test_files=['2020_Himawari_cloudsat_128_cloud_SAZ']
nocloud_file=['2018_Himawari_cloudsat_128_nocloud_SAZ']
'''
save_file='long128_new3'
data_path='/public/home/xiongqq/Data_Train'
image_save='/public/home/xiongqq/Model_out_VAE/image/{}'.format(save_file)
weight_save='/public/home/xiongqq/Model_out_VAE/weight/{}'.format(save_file)
'''
save_file='long128_1_0001'
data_path='/work/home/acmh4zm9q3/Data_Train'
image_save='/work/home/acmh4zm9q3/VAE_out/image/{}'.format(save_file)
weight_save='/work/home/acmh4zm9q3/VAE_out/weight/{}'.format(save_file)

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

def Partition(samples,random_size):
    samples_size = samples.shape[0]
    indices = np.random.choice(samples_size, size=random_size, replace=False)
    mask = np.zeros(samples_size, dtype=bool)
    mask[indices] = True
    return mask,~mask

class VAE(tf.keras.Model):
    def __init__(self, encoder, decoder, **kwargs):
        super().__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        # 计算总损失
        self.total_loss_tracker = tf.keras.metrics.Mean(name="total_loss")
        # 计算重构损失
        self.reconstruction_loss_tracker = tf.keras.metrics.Mean(name="reconstruction_loss")
        # 计算KL散度损失
        self.kl_loss_tracker = tf.keras.metrics.Mean(name="kl_loss")
    
    # 返回模型的评估指标,包含总损失、重构损失和KL散度损失
    @property
    def metrics(self):
        return [self.total_loss_tracker,
                self.reconstruction_loss_tracker,
                self.kl_loss_tracker,]

    def train_step(self, data):
        with tf.GradientTape() as tape:# 开启梯度记录模式
            # 从输入数据中提取特征并得到潜在变量z的均值、方差和z本身
            z_mean, z_log_var, z = self.encoder(data)
            # 解码器将潜在变量z转换为重构后的数据
            reconstruction = self.decoder(z)
            #print(tf.reduce_min(reconstruction),tf.reduce_min(z_mean),tf.reduce_min(z_log_var),tf.reduce_min(z))
            # 计算重构损失（二进制交叉熵）
            #reconstruction_loss = tf.reduce_mean(tf.reduce_sum(tf.keras.losses.binary_crossentropy(data, reconstruction),axis=(1,2)))
            reconstruction_loss = tf.reduce_sum(tf.reduce_sum(tf.keras.losses.MeanSquaredError(tf.keras.losses.Reduction.NONE)(data, reconstruction)))
            # 计算KL散度损失（负对数似然）
            kl_loss = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
            #kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=(1,2)))
            allkl_loss = tf.reduce_sum(tf.reduce_sum(kl_loss, axis=(1,2,3)))
            #kl_loss = tf.reduce_mean(kl_loss)
            # 总损失为重构损失加KL散度损失
            #total_loss = reconstruction_loss*8 + allkl_loss*0.1
            total_loss = reconstruction_loss + allkl_loss*0.001
            
        # 计算梯度并更新权重
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        
        # 更新指标
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(allkl_loss)
        
        return {"total_loss": self.total_loss_tracker.result(),
                "reconstruction_loss": self.reconstruction_loss_tracker.result(),
                "kl_loss": self.kl_loss_tracker.result(),}

class SaveWeights(tf.keras.callbacks.Callback):
    def __init__(self, encoder, decoder, weight_save, time):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.weight_save = weight_save
        self.time = time  # New parameter n
        os.makedirs(weight_save, exist_ok=True)
    def on_epoch_end(self, epoch, logs=None):
        # Constructing the filepath with n, epoch number, and loss
        encoderpath = f'{self.weight_save}/encoder_time{self.time}_epochs{epoch:02d}_loss{logs["reconstruction_loss"]:.3f}.weights.h5'
        self.encoder.save_weights(encoderpath)
        decoderpath = f'{self.weight_save}/decoder_time{self.time}_epochs{epoch:02d}_loss{logs["reconstruction_loss"]:.3f}.weights.h5'
        self.decoder.save_weights(decoderpath)

class PlotImages(tf.keras.callbacks.Callback):
    def __init__(self,data,encoder, decoder,time):
        super(PlotImages, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.data=data
        self.time=time
        os.makedirs(image_save, exist_ok=True)
        
    def on_epoch_end(self,epoch, logs=None):
        test_target = read.scale_Reflect(self.data['Radar_Reflectivity'])
        #test_target = tf.image.resize(test_target, (256, 256), method=tf.image.ResizeMethod.AREA)
        z_mean, z_log_var, z = self.encoder.predict(test_target, verbose=1, batch_size=192)
        gen = self.decoder.predict(z, verbose=1, batch_size=192)
        #out = tf.image.resize(gen, (64, 160), method=tf.image.ResizeMethod.AREA)
        result = read.rescale_gen(gen[:,:,:,0])
        rmse = analyse.Count_remse(result,self.data['Radar_Reflectivity'])
        mean,var=analyse.mean_var(rmse)
        #画图
        data_name=['cloud_class',
                'real_Reflectivity',
                'VAE_Reflectivity',
                'laten_Reflectgrey',
                'mean_Reflectgrey',
                'log_var_Reflectgrey']
        data_list=[self.data['cloud_scenario'],
                self.data['Radar_Reflectivity'],
                result ,
                z[...,0],
                z_mean[...,0],
                z_log_var[...,0]]
        hight_layer=[None,None,None,
                     np.arange(480, 480+32*480,480),
                     np.arange(480, 480+32*480,480),
                     np.arange(480, 480+32*480,480)]
        #PLOT.images(data_list,data_name,hight_layer=hight_layer,save_name=save_name)
        PLOT.images(data_list,data_name,hight_layer=hight_layer,save_name=f'time{self.time}_epoch{epoch}_mean{mean:.2f}_var{var:.2f}')

'''初始化画图'''
PLOT=plot.Comparison(image_save,resolution=4)
'''训练集'''
test_nocloud = make_dataset(data_path,nocloud_file,var_name,random=1000,trans=True)
train_nocloud=make_dataset(data_path,nocloud_file,var_name,random=2000,trans=True)
test_cloud = make_dataset(data_path,test_files,var_name,trans=True)
train_cloud= make_dataset(data_path,train_files,var_name,trans=True)
'''
test_index,train_index=Partition(data_cloud,1000)
train_cloud = data_cloud[train_index]
test_cloud = data_cloud[test_index]
'''
train_data={}
for varname in var_name:
    train_data[varname] = np.concatenate((train_cloud[varname], train_nocloud[varname]), axis=0)
    print(varname,len(train_data[varname]))
train_target = read.scale_Reflect(train_data['Radar_Reflectivity'])
#train_target = tf.image.resize(train_target, (256, 256), method=tf.image.ResizeMethod.AREA)
print(train_target.shape)

data = tf.data.Dataset.from_tensor_slices((train_target))
data = data.shuffle(buffer_size=len(train_target))
#print(data.shape)
encoder=model.Encoder(train_target.shape[1], train_target.shape[2])
decoder=model.Decoder(32, 32)
vae = VAE(encoder, decoder)
vae.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.85, beta_2=0.9,amsgrad=True),metrics=['accuracy'])
'''训练模型'''
for time in range(star,times+1):
    plot_callback = PlotImages(test_cloud,vae.encoder,vae.decoder,f'{time}_cloud')
    plot_callback1 = PlotImages(test_nocloud,vae.encoder,vae.decoder,f'{time}_nocloud')
    weight_callback = SaveWeights(vae.encoder,vae.decoder,weight_save,time)
    weight_encoder = glob.glob(os.path.join(weight_save, 'encoder*.h5'))
    weight_decoder = glob.glob(os.path.join(weight_save, 'decoder*.h5'))
    if weight_encoder!=[] and weight_decoder!=[]:
        print('加载权重')
        # 选择最新的权重文件
        latest_encoder = max(weight_encoder, key=os.path.getctime)
        latest_decoder = max(weight_decoder, key=os.path.getctime)
        # 加载最新的权重到模型中
        vae.encoder.load_weights(latest_encoder)
        vae.decoder.load_weights(latest_decoder)
        print(f"第{time}次训练")
    else:
        print("开始训练")
    dataset = data.batch(2**(time+1))
    vae.fit(dataset,epochs=30,callbacks=[weight_callback,plot_callback,plot_callback1])
print('训练结束')