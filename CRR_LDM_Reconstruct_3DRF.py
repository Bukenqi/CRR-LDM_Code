# -*- coding: utf-8 -*-
"""
Created on Fri Apr 17 15:33:17 2026

@author: Admin
"""

import time
import os
import pwd
import gc
#os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from tqdm import tqdm
import numpy as np
import FUNC_read_data as read
import tensorflow as tf
import X_Latent_DDIM_2km_UNet_final as DDIM_UNet
import X_VAE_model_2km_new as VAE
from scipy.ndimage import convolve
import calendar
import multiprocessing as mp
from multiprocessing import shared_memory
import threading
import queue
#%%
Gen_batch= 512
move_step= 8
addnoise_step = 500
denoise_step=5
MASK_DTYPE = np.uint8
tital='Full_Disk'
area={'minlat':-60,'maxlat':60,'minlon':80,'maxlon':120}
model='CRR-LDM-Full'#'CRR-LDM-IR'
data="/work/home/acmh4zm9q3/Data_test"
save='/work/home/acmh4zm9q3/Data_3D'
time_option={'year':[2024],#可以接受输入'2024~2025',或者列表[2024,2025]
      'month':[11],#可以接受输入'1~12',或者列表[2,3,6,8],或字符串'all'
      'day':[10],#可以接受输入'0~29',或者列表[1,2,7,6,10],或字符串'all'
      'hour':[0,1,2,3,4,5,6],#可以接受输入'0~23',或者列表[6,9,17,19,21],或字符串'all'
      'min':'all',#可以接受输入'0~50',或者列表[0,10,20,30,40,50],或字符串'all'
      }

#%%
save_path=f'{save}/{tital}'
file_path=f"{data}/{tital}"
decoder_weight="/work/home/acmh4zm9q3/VAE_out/weight/decoder_time4_epochs29_loss416.884.weights.h5"
if 'Full' in model:
    DDIM_weight="/work/home/acmh4zm9q3/LDM_out/weight/CRR_LDM_Full/Lat_DDIM_time5_epoch29_loss0.312.weights.h5"
    var_name=['albedo_01','albedo_02','albedo_03','albedo_04','albedo_05','albedo_06','tbb_07',
              'tbb_08','tbb_09','tbb_10','tbb_11','tbb_12','tbb_13','tbb_14','tbb_15','tbb_16',
              'SOZ','longitude','latitude']
elif 'IR' in model:
    DDIM_weight="/work/home/acmh4zm9q3/LDM_out/weight/CRR_LDM_IR/Lat_DDIM_time5_epoch29_loss0.310.weights.h5"
    var_name=['albedo_01','albedo_02','albedo_03','albedo_04','albedo_05','albedo_06','tbb_07',
              #'tbb_08','tbb_09','tbb_10','tbb_11','tbb_12','tbb_13','tbb_14','tbb_15','tbb_16',
              'SOZ','longitude','latitude']
else:
    print('模型输入错误')
'''
#%%
save_path=f'{save}/{tital}'
file_path=f"{data}/{tital}"
decoder_weight="D:/weights/VAE/decoder_time4_epochs29_loss416.884.weights.h5"
if 'Full' in model:
    DDIM_weight="D:/weights/CRR_LDM_Full/Lat_DDIM_time5_epoch29_loss0.312.weights.h5"
    var_name=['albedo_01','albedo_02','albedo_03','albedo_04','albedo_05','albedo_06','tbb_07',
              'tbb_08','tbb_09','tbb_10','tbb_11','tbb_12','tbb_13','tbb_14','tbb_15','tbb_16',
              'SOZ','longitude','latitude']
elif 'IR' in model:
    DDIM_weight="D:/weights/CRR_LDM_IR/Lat_DDIM_time5_epoch29_loss0.310.weights.h5"
    var_name=['albedo_01','albedo_02','albedo_03','albedo_04','albedo_05','albedo_06','tbb_07',
              #'tbb_08','tbb_09','tbb_10','tbb_11','tbb_12','tbb_13','tbb_14','tbb_15','tbb_16',
              'SOZ','longitude','latitude']
else:
    print('模型输入错误')
'''
#%%获取用户名称
dir_stat = os.stat(save_path)
owner_uid = dir_stat.st_uid
owner_gid = dir_stat.st_gid
owner_name = pwd.getpwuid(owner_uid).pw_name
print(f"用户名称：{owner_name}")
print(f"用户 UID：{owner_uid}")
print(f"用户 GID：{owner_gid}")
#%%
def data_area(file_path,var_name,area=None):
    data = read.read_varible(file_path,var_name,Normal=True)
    if area is not None:
        lat_index = np.where((data['latitude'] >= area['minlat']) & (data['latitude'] <= area['maxlat']))[0]
        lon_index = np.where((data['longitude'] >= area['minlon']) & (data['longitude'] <= area['maxlon']))[0]
        lon,lat = np.meshgrid(data['longitude'][lon_index ],data['latitude'][lat_index])
        del data['latitude'],data['longitude']
        index=np.ix_(lat_index, lon_index)
        for name in var_name[:-2]:
            data[name]=data[name][index]
        input_varible = np.stack(list(data.values()),axis=2)
        del data
    else:
        lon,lat = np.meshgrid(data['longitude'],data['latitude'])
        del data['latitude'],data['longitude']
        input_varible = np.stack(list(data.values()),axis=2)
        del data
    return input_varible.astype(np.float32),lat.astype(np.float32),lon.astype(np.float32)

def time_list(time_option,key,year=None, month=None):
    if isinstance(time_option[key], list):
        out = time_option[key]
    else:
        if time_option['month']=='all':
            start,end=1,12
        elif time_option['day']=='all':
            _,end=calendar.monthrange(year, month)
            start=1
        elif time_option['hour']=='all':
            start,end=0,23
        elif time_option['min']=='all':
            start,end=0,50
        else:
            start,end = map(int, time_option[key].split('~'))
        #print(start,end)
        if key=='min':
            out = list(range(start, end+10,10))
        else:
            out = list(range(start, end+1))
    return out

def screen_data(time_option):
    """数据筛选"""
    years=time_list(time_option,'year')
    months=time_list(time_option,'month')
    hours=time_list(time_option,'hour')
    minutes=time_list(time_option,'min')
    file_list=[]
    name_list=[]
    for year in years:
        for month in months:
            days=time_list(time_option,'day',year,month)
            for day in days:
                for hour in hours:
                    for mins in minutes:
                        Himawari_file = f'NC_H09_{year}{month:02d}{day:02d}_{hour:02d}{mins:02d}_R21_FLDK.06001_06001.nc'
                        Himawari_path = f'{file_path}/{Himawari_file}'
                        if os.path.exists(Himawari_path):
                            file_list.append(Himawari_path)
                            name_list.append(f'Gen_{tital}_{year}{month:02d}{day:02d}_{hour:02d}{mins:02d}_2km')
                        else:
                            print(f"文件 {Himawari_file} 不存在")
    if not file_list:
        print("没有数据集，结束当前任务")
        exit()  # 或 quit()
    return file_list,name_list

class BatchResultUpdater:
    def __init__(self, threshold=-0.96):
        self.threshold = threshold

    def cpu_postprocess_batch(self, batch_np):
        batch_np = batch_np.astype(np.float32, copy=False)
        batch_np[batch_np < self.threshold] = np.nan
        batch_np = (batch_np[..., 0] + 1.0) / 2.0
        batch_np = np.transpose(batch_np, (0, 2, 1)).astype(np.float32, copy=False)
        return batch_np

    def update_result_lon(self, result_array, processed, batch_index_array):
        for i in range(len(batch_index_array)):
            fixed_index, valid_start, valid_end, local_start, local_end = batch_index_array[i]
            dst = result_array[fixed_index, valid_start:valid_end, :]
            src = processed[i, local_start:local_end, :]
            result_array[fixed_index, valid_start:valid_end, :] = np.nanmean(
                np.stack((dst, src), axis=0), axis=0
            )
        return result_array

    def update_result_lat(self, result_array, processed, batch_index_array):
        for i in range(len(batch_index_array)):
            fixed_index, valid_start, valid_end, local_start, local_end = batch_index_array[i]
            dst = result_array[valid_start:valid_end, fixed_index, :]
            src = processed[i, local_start:local_end, :]
            result_array[valid_start:valid_end, fixed_index, :] = np.nanmean(
                np.stack((dst, src), axis=0), axis=0
            )
        return result_array
    '''
    def cpu_consumer_worker(self, tensor_queue, result_array, stop_token, axis='lon'):
        while True:
            item = tensor_queue.get()

            if item is stop_token:
                tensor_queue.task_done()
                break
            batch_idx, batch_tensor, batch_index_array = item
            try:
                processed = self.cpu_postprocess_batch(batch_tensor.numpy())
                del batch_tensor
                if axis == 'lon':
                    self.update_result_lon(result_array, processed, batch_index_array)
                elif axis == 'lat':
                    self.update_result_lat(result_array, processed, batch_index_array)
                else:
                    raise ValueError("axis must be 'lon' or 'lat'")
                del processed
            finally:
                gc.collect()
                tensor_queue.task_done()
    ''' 
    def cpu_consumer_worker(self, tensor_queue, result_array, stop_token, error_queue=None):
        while True:
            item = tensor_queue.get()
    
            try:
                if item is stop_token:
                    break
    
                batch_idx, batch_tensor, batch_index_array, batch_flg = item
    
                processed = self.cpu_postprocess_batch(batch_tensor)
                del batch_tensor
    
                lon_mask = (batch_flg == 0)
                if np.any(lon_mask):
                    self.update_result_lon(result_array,processed[lon_mask],batch_index_array[lon_mask])
    
                lat_mask = (batch_flg == 1)
                if np.any(lat_mask):
                    self.update_result_lat(result_array,processed[lat_mask],batch_index_array[lat_mask])
    
                del processed
    
            except Exception as e:
                if error_queue is not None:
                    try:
                        error_queue.put_nowait(e)
                    except queue.Full:
                        pass
                break
    
            finally:
                gc.collect()
                tensor_queue.task_done()



class SliceIndexer2D:
    def __init__(self, in_len=64, resolution=2, dtype=np.float32):
        self.in_len = int(in_len)
        self.resolution = int(resolution)
        self.dtype = dtype

    def _build_index_array(self, fixed_index, raw_start, raw_end, raw_limit):
        valid_start_in = np.maximum(raw_start, 0)
        valid_end_in = np.minimum(raw_end, raw_limit)
        valid_width = valid_end_in - valid_start_in

        local_start = (valid_start_in - raw_start) * self.resolution
        local_end = local_start + valid_width * self.resolution

        valid_start = valid_start_in * self.resolution
        valid_end = valid_end_in * self.resolution

        index_array = np.stack(
            [fixed_index, valid_start, valid_end, local_start, local_end],
            axis=1
        ).astype(np.int32, copy=False)

        keep = valid_width > 0
        return index_array[keep], keep

    def slicer_lon(self, inputs_np, padding=0):
        lat, lon, c = inputs_np.shape
        front = int(padding)
        padded_len = int(np.ceil((lon + front) / self.in_len)) * self.in_len
        back = padded_len - (lon + front)

        padded = np.pad(inputs_np, ((0, 0), (front, back), (0, 0)), mode='constant')
        n_blocks = padded_len // self.in_len

        samples = padded.reshape(lat, n_blocks, self.in_len, c)
        samples = samples.reshape(-1, self.in_len, c).astype(self.dtype, copy=False)

        fixed_index = np.repeat(np.arange(lat, dtype=np.int32) * self.resolution, n_blocks)
        block_id = np.tile(np.arange(n_blocks, dtype=np.int32), lat)

        raw_start = block_id * self.in_len - front
        raw_end = raw_start + self.in_len

        index_array, keep = self._build_index_array(fixed_index, raw_start, raw_end, lon)
        samples = samples[keep]

        return samples, index_array

    def slicer_lat(self, inputs_np, padding=0):
        lat, lon, c = inputs_np.shape
        front = int(padding)
        padded_len = int(np.ceil((lat + front) / self.in_len)) * self.in_len
        back = padded_len - (lat + front)

        padded = np.pad(inputs_np, ((front, back), (0, 0), (0, 0)), mode='constant')
        n_blocks = padded_len // self.in_len

        samples = padded.reshape(n_blocks, self.in_len, lon, c)
        samples = np.transpose(samples, (2, 0, 1, 3))
        samples = samples.reshape(-1, self.in_len, c).astype(self.dtype, copy=False)

        fixed_index = np.repeat(np.arange(lon, dtype=np.int32) * self.resolution, n_blocks)
        block_id = np.tile(np.arange(n_blocks, dtype=np.int32), lon)

        raw_start = block_id * self.in_len - front
        raw_end = raw_start + self.in_len

        index_array, keep = self._build_index_array(fixed_index, raw_start, raw_end, lat)
        samples = samples[keep]
        return samples, index_array


class LDM_Generator:
    def __init__(self, DDIM_weight, decoder_weight, batchs, addnoise_step, denoise_step):
        self.Diffusion = DDIM_UNet.GaussianDiffusion(timesteps=addnoise_step, clip_min=-3.0, clip_max=3.0)
        self.decoder = VAE.Decoder(32, 32)
        self.decoder.load_weights(decoder_weight)
        self.u_net = DDIM_UNet.U_Net(32, 64, len(var_name)-2)
        self.u_net.load_weights(DDIM_weight)
        self.addnoise_step = addnoise_step
        self.denoise_step = denoise_step
        self.batch_size= batchs

    @tf.function(jit_compile=True)  # 启用 XLA
    def Generate_2D_Ref(self,inputs):
        # 将原有 generate_data 中的 Python 逻辑全部用 TensorFlow 重写
        # 包括循环、噪声生成、网络调用等
        # 注意：网络调用需使用 model.call，而不是 model.predict
        variable = tf.convert_to_tensor(inputs, dtype=tf.float32)
        noise = tf.random.normal(shape=(tf.shape(variable)[0], 32, 32, 1), dtype=tf.float32)
        latent = tf.clip_by_value(noise, -3.0, 3.0)
        skip_timestep = self.addnoise_step // self.denoise_step
        # 注意：此处需使用 tf.while_loop 或 for loop（在 tf.function 内自动图展开）
        for t in tf.range(self.addnoise_step-1, 0, -skip_timestep):
            ts = t - skip_timestep
            ts = tf.where(ts < 0, 0, ts)
            #print(t,ts)
            t_k = tf.fill([tf.shape(variable)[0]], t)
            t_s = tf.fill([tf.shape(variable)[0]], ts)
            pred_noise = self.u_net([latent, t_k, variable])  # 直接调用模型，而非 predict
            latent = self.Diffusion.DDIM_denoise(pred_noise, latent, t_k, t_s, clip_denoised=True)
        sample = self.decoder(latent)
        return sample
    
    def Generate_3D_Ref(self, inputs, resolution=2, padding=None, Content=None, position=0):
        lat, lon, _ = inputs.shape
        expected_shape = (lat * resolution, lon * resolution, 64)
        result_array = np.full(expected_shape, np.nan, dtype=np.float32)
    
        if Content is None:
            Content = f"GPU{position}"
    
        pad_val = 0 if padding is None else int(padding)
    
        # 1) 两类切片
        samples_lon, index_lon = self.slice_indexer.slicer_lon(inputs, padding=pad_val)
        samples_lat, index_lat = self.slice_indexer.slicer_lat(inputs, padding=pad_val)
    
        # 2) 合并样本、索引和标记
        samples = np.concatenate([samples_lon, samples_lat], axis=0).astype(np.float32, copy=False)
        index_array = np.concatenate([index_lon, index_lat], axis=0).astype(np.int32, copy=False)
        flg = np.concatenate([np.zeros(samples_lon.shape[0], dtype=np.uint8),   # 0 -> lon
                              np.ones(samples_lat.shape[0], dtype=np.uint8)     # 1 -> lat
                              ], axis=0)
    
        # 3) 提前释放中间变量，降低内存峰值
        del samples_lon, samples_lat, index_lon, index_lat
        gc.collect()
    
        # 4) 构建数据集
        dataset = tf.data.Dataset.from_tensor_slices((samples, index_array, flg))
        dataset = dataset.batch(self.batch_size)
        dataset = dataset.prefetch(tf.data.AUTOTUNE)
    
        # 5) 启动消费者线程
        tensor_queue = queue.Queue(maxsize=5)
        consumer_error_queue = queue.Queue(maxsize=1)
        stop_token = object()
    
        consumer_thread = threading.Thread(target=self.batch_updater.cpu_consumer_worker,
                                           args=(tensor_queue, result_array, stop_token, consumer_error_queue),
                                           daemon=True)
        consumer_thread.start()
        try:
            total_batches = dataset.cardinality().numpy()
    
            with tqdm(total=total_batches, desc=f"{Content}-mixed", position=position, leave=False) as pbar:
                for batch_idx, (batch_input, batch_index_array, batch_flg) in enumerate(dataset):
                    if not consumer_error_queue.empty():
                        raise consumer_error_queue.get()
    
                    batch_output = self.Generate_2D_Ref(batch_input)
    
                    # 主线程里转 numpy，再交给消费者线程，稳定一些
                    batch_output_np = batch_output.numpy()
                    batch_index_array_np = batch_index_array.numpy()
                    batch_flg_np = batch_flg.numpy()
    
                    #del batch_output
    
                    tensor_queue.put((batch_idx,batch_output_np,batch_index_array_np,batch_flg_np))
                    pbar.update(1)
    
            tensor_queue.put(stop_token)
            tensor_queue.join()
            consumer_thread.join()
            
            if not consumer_error_queue.empty():
                raise consumer_error_queue.get()
        finally:
            del samples, index_array, flg, dataset
            gc.collect()
        return result_array


def create_shared_array(shape, dtype, fill_value=None):
    dtype = np.dtype(dtype)
    nbytes = int(np.prod(shape)) * dtype.itemsize
    shm = shared_memory.SharedMemory(create=True, size=nbytes)
    arr = np.ndarray(shape, dtype=dtype, buffer=shm.buf)
    if fill_value is not None:
        arr.fill(fill_value)
    return shm, arr

def attach_shared_array(name, shape, dtype):
    dtype = np.dtype(dtype)
    shm = shared_memory.SharedMemory(name=name)
    arr = np.ndarray(shape, dtype=dtype, buffer=shm.buf)
    return shm, arr

# =========================
# 子进程
# =========================

def worker(gpu_id,move,
           input_shm_name,input_shape,input_dtype,
           result_shm_name,result_shape, result_dtype,
           count_shm_name, count_shape, count_dtype,
           lock, error_queue):
    input_shm = None
    result_shm = None
    count_shm = None
    try:
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            target_gpu = gpus[gpu_id % len(gpus)]
            tf.config.set_visible_devices([target_gpu], 'GPU')
            tf.config.experimental.set_memory_growth(target_gpu, True)
        input_shm, inputs = attach_shared_array(input_shm_name, input_shape, input_dtype)
        result_shm, result_array = attach_shared_array(result_shm_name, result_shape, result_dtype)
        count_shm, count_array = attach_shared_array(count_shm_name, count_shape, count_dtype)
        #模型初始化
        generator = LDM_Generator(DDIM_weight, decoder_weight, Gen_batch, addnoise_step, denoise_step)
        generator.slice_indexer = SliceIndexer2D(in_len=64, resolution=2)
        generator.batch_updater = BatchResultUpdater(threshold=-0.96)
        # 只拿自己这一层共享视图
        local_result=generator.Generate_3D_Ref(inputs=inputs,resolution=2,padding=move,
                                  Content=f"GPU{gpu_id}-move{move}",position=gpu_id)
        valid_mask = ~np.isnan(local_result)
        with lock:
            result_array[valid_mask] += local_result[valid_mask]
            count_array[valid_mask] += 1
        del local_result, inputs, generator
        gc.collect()
        
    except Exception as e:
        error_queue.put(f"GPU {gpu_id} move{move} 出错: {e}")
        raise
    finally:
        if input_shm is not None:
            input_shm.close()
        if result_shm is not None:
            result_shm.close()
        if count_shm is not None:
            count_shm.close()

# =========================
# 主程序
# =========================
if __name__ == '__main__':
    mp.set_start_method('spawn', force=True)

    gpus = tf.config.list_physical_devices('GPU')
    num_gpus = max(len(gpus), 1)

    file_list, name_list = screen_data(time_option)
    Processing_failed_file = []

    for path, save_name in zip(file_list, name_list):
        inputs, _, _ = data_area(path, var_name, area)
        print(f'inputs：形状{inputs.shape}, 最大值{np.max(inputs)}，最小值{np.min(inputs)}')

        lat_size, lon_size, chanel = inputs.shape
        out_shape = (lat_size * 2, lon_size * 2, 64)

        move_list = list(np.arange(0, 64, move_step))
        num_tasks = len(move_list)

        # 1) 创建共享输入
        input_shm, input_array = create_shared_array(inputs.shape, np.float32)
        input_array[:] = inputs[:]
        del inputs
        gc.collect()

        # 2) 创建共享结果栈：(任务数, H, W, 64)
        result_shm, result_array = create_shared_array(out_shape, np.float32, fill_value=0.0)
        count_shm, count_array = create_shared_array(out_shape, np.uint8, fill_value=0)

        error_queue = mp.Queue()
        processes = []
        error_occurred = False
        lock = mp.Lock()

        # 3) 每个 move 一个进程
        for slot_idx, move in enumerate(move_list):
            gpu_id = slot_idx % num_gpus
            p = mp.Process(target=worker,args=(gpu_id, move,input_shm.name, input_array.shape, np.float32,
                                               result_shm.name, out_shape, np.float32,
                                               count_shm.name, out_shape, np.uint8,
                                               lock, error_queue))
            processes.append(p)
            p.start()

        # 4) 等待结束
        for p in processes:
            p.join()

        for p in processes:
            if p.exitcode != 0:
                error_occurred = True
                print(f"子进程 {p.pid} 异常退出，退出码 {p.exitcode}")

        while not error_queue.empty():
            print(error_queue.get())

        if error_occurred:
            print(f"文件 {path} 处理失败，跳过当前文件。")
            Processing_failed_file.append(path)
            input_shm.close()
            input_shm.unlink()
            result_shm.close()
            result_shm.unlink()
            count_shm.close()
            count_shm.unlink()
            continue
        input_shm.close()
        input_shm.unlink()
        # 5) 最终合并
        Reflectivity = np.divide(result_array,count_array,out=np.full_like(result_array, 0.0, dtype=np.float32),
                                 where=(count_array > 0)).astype(np.float32, copy=False)
        print(f"Reflectivity合并后: 有效点{np.sum(~np.isnan(Reflectivity))}，"
              f"最大值{np.nanmax(Reflectivity)}，最小值{np.nanmin(Reflectivity)}")
        # 6) 释放共享内存
        result_shm.close()
        result_shm.unlink()
        count_shm.close()
        count_shm.unlink()
        print(f"Reflectivity: 有效点{np.sum(~np.isnan(Reflectivity))}，最大值{np.nanmax(Reflectivity)}，最小值{np.nanmin(Reflectivity)}")
        Reflectivity = Reflectivity.transpose(2,0,1)[::-1,::-1,:]
        #%% 卷积平滑处理
        kernel = np.ones((1,4,4))/11
        Reflectivity = convolve(Reflectivity, kernel, mode='nearest')
        print(f"Reflectivity填充: 有效点{np.sum(Reflectivity >0.001)}，最大值{Reflectivity.max()}，最小值{Reflectivity.min()}")
        kernel = np.ones((1,3,3))/9
        for i in range(2):
            Reflectivity = convolve(Reflectivity, kernel, mode='nearest')
            print(f"Reflectivity平滑: 有效点{np.sum(Reflectivity >0.001)}，最大值{Reflectivity.max()}，最小值{Reflectivity.min()}")
        Reflectivity=Reflectivity*55-35
        print(f"Reflectivity缩放: 有效点{np.sum(Reflectivity >0.001)}，最大值{Reflectivity.max()}，最小值{Reflectivity.min()}")
        Reflectivity[Reflectivity>20]=20
        Reflectivity[Reflectivity<=-34]=np.nan
        print(f"Reflectivity最终: 有效点{np.sum(Reflectivity >0.001)}，最大值{Reflectivity.max()}，最小值{Reflectivity.min()}")
        #%% 保存NC
        start_time = time.perf_counter()
        
        data={}
        High=np.arange(480, 15840,240) 
        Lat=np.linspace(area['minlat'], area['maxlat']+0.01, int(2+(area['maxlat']-area['minlat'])/0.01), dtype=np.float32) 
        Lon=np.linspace(area['minlon'], area['maxlon']+0.01, int(2+(area['maxlon']-area['minlon'])/0.01), dtype=np.float32)
        '''
        data = {'Lat': Lat,     
                'Lon': Lon,
                'reflectivity':Reflectivity.astype(np.float32),}
        np.savez(f"{save_path}/{save_name}.npz", **data)
        # 修改文件所有者和组
        os.chown(f"{save_path}/{save_name}.npz", owner_uid, owner_gid)
        # 可选：赋予用户和组读写权限
        os.chmod(f"{save_path}/{save_name}.npz", 0o777)
        '''
        data = {'name': save_name,
                'shape': {'High':High, 'Lat':Lat, 'Lon':Lon},
                'reflectivity': {'value': Reflectivity.astype(np.float32) ,
                                 'dims':['High','Lat','Lon'],
                                 'long_name':None,
                                 'description':'Equivalent W-band',
                                 'units':'dbz',},
                }
        read.save_data(data,save_path,f'{save_name}')
        # 修改文件所有者和组
        os.chown(f"{save_path}/{save_name}.nc", owner_uid, owner_gid)
        # 可选：赋予用户和组读写权限
        os.chmod(f"{save_path}/{save_name}.nc", 0o777)
        end_time = time.perf_counter()
        elapsed = end_time - start_time
        print(f"总耗时：{elapsed:.4f} 秒")
    for fail_file in Processing_failed_file:
        print('生成失败的文件')
        print(fail_file)

