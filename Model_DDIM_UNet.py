# -*- coding: utf-8 -*-
"""
Created on Fri Sep 12 21:42:39 2025

@author: 59278
"""

import math
import numpy as np
# Requires TensorFlow >=2.11 for the GroupNormalization layer.
#import tensorrt as trt
import tensorflow as tf
from tensorflow import keras
from keras.models import Model
    
'''
varible = np.random.uniform(low=0, high=1, size=(8, 64, 32))
noise = np.random.uniform(low=-3, high=3, size=(8, 32,32,32))
a = keras.layers.Activation("tanh")(varible)
a = a.numpy()
b = keras.layers.Activation("swish")(varible)
b = b.numpy()
c = keras.layers.Activation("relu")(varible)
c = c.numpy()
d = keras.layers.Activation("sigmoid")(varible)
d = d.numpy()

a = keras.layers.GroupNormalization(epsilon=1e-5)(varible)
a = a.numpy()
b = keras.layers.GroupNormalization(epsilon=1e-5)(noise)
b = b.numpy()

a = keras.layers.LayerNormalization(epsilon=1e-5)(varible)
a = a.numpy()
b = keras.layers.LayerNormalization(epsilon=1e-5)(noise)
b = b.numpy()

b = keras.layers.LayerNormalization(epsilon=1e-5)(varible)
b = b.numpy()
c = keras.layers.BatchNormalization(epsilon=1e-5)(varible)
c = c.numpy()
'''
'''给步长进行编码'''
class TimeEmbedding(keras.layers.Layer):
    def __init__(self, dim, **kwargs):
        super(TimeEmbedding,self).__init__(**kwargs)
        self.dim = dim
        self.half_dim = dim // 2
        self.emb = math.log(10000) / (self.half_dim - 1)
        self.emb = tf.exp(tf.range(self.half_dim, dtype=tf.float32) * -self.emb)

    def call(self, inputs):
        inputs = tf.cast(inputs, dtype=tf.float32)
        emb = inputs[:, None] * self.emb[None, :]
        emb = tf.concat([tf.sin(emb), tf.cos(emb)], axis=-1)
        return tf.cast(emb, dtype=inputs.dtype)
   
'''创建采样层'''
class PaddedConv2D(keras.layers.Layer):
    def __init__(self, filters, kernel_size, padding=0, strides=1, **kwargs):
        super(PaddedConv2D,self).__init__(**kwargs)
        self.padding2d = keras.layers.ZeroPadding2D(padding)
        self.conv2d = keras.layers.Conv2D(filters, kernel_size, strides=strides)

    def call(self, inputs):
        x = self.padding2d(inputs)
        return self.conv2d(x)

class PaddedConv2DTranspose(keras.layers.Layer):
    def __init__(self, filters, kernel_size, strides=1, **kwargs):
        super(PaddedConv2DTranspose,self).__init__(**kwargs)
        self.Conv2DT = keras.layers.Conv2DTranspose(filters, kernel_size, strides, padding='same')
        self.padConv2D=PaddedConv2D(filters,3,padding=1)
        
    def call(self, inputs):
        x = self.Conv2DT(inputs)
        return self.padConv2D(x)
  
class Conv2D_Block(keras.layers.Layer):
    def __init__(self, filters, kernel_size, padding=0, strides=1, **kwargs):
        super(Conv2D_Block,self).__init__(**kwargs)
        self.norm = keras.layers.GroupNormalization(epsilon=1e-5)
        #self.active=keras.layers.LeakyReLU(0.8)
        self.active=keras.layers.Activation("swish")
        #self.active = activation_fn
        self.conv = PaddedConv2D(filters, kernel_size, padding=padding, strides=strides)
    def call(self, inputs):
        return self.conv(self.active(self.norm(inputs)))


class Conv2DT_Block(keras.layers.Layer):
    def __init__(self, filters, kernel_size, padding=0, strides=1, **kwargs):
        super(Conv2DT_Block,self).__init__(**kwargs)
        self.norm = keras.layers.GroupNormalization(epsilon=1e-5)
        #self.active=keras.layers.LeakyReLU(0.8)
        self.active=keras.layers.Activation("swish")
        #self.active = activation_fn
        self.conv = PaddedConv2DTranspose(filters, kernel_size, strides=strides)
        
    def call(self, inputs):
        return self.conv(self.active(self.norm(inputs)))
'''
class Upsample(keras.layers.Layer):
    def __init__(self,filters, kernel_size, strides=1, **kwargs):
        super().__init__(**kwargs)
        self.up = keras.Sequential([
            keras.layers.GroupNormalization(epsilon=1e-5),
            keras.layers.Activation("swish"),
            PaddedConv2D(filters, 3, padding=1),
            keras.layers.GroupNormalization(epsilon=1e-5),
            keras.layers.Activation("swish"),
            PaddedConv2DTranspose(filters, kernel_size, strides=strides),
            keras.layers.GroupNormalization(epsilon=1e-5),
            keras.layers.Activation("swish"),
            PaddedConv2D(filters, 3, padding=1),
            ])
        self.final = PaddedConv2D(filters,1)
        
    def call(self, inputs):
        return self.final(self.up(inputs))

class Downsample(keras.layers.Layer):
    def __init__(self, filters, kernel_size, strides=1, padding=0, **kwargs):
        super().__init__(**kwargs)
        self.down = keras.Sequential([
            keras.layers.GroupNormalization(epsilon=1e-5),
            keras.layers.Activation("swish"),
            PaddedConv2D(filters, 3, padding=1),
            keras.layers.GroupNormalization(epsilon=1e-5),
            keras.layers.Activation("swish"),
            PaddedConv2D(filters, kernel_size, strides=strides, padding=padding),
            keras.layers.GroupNormalization(epsilon=1e-5),
            keras.layers.Activation("swish"),
            PaddedConv2D(filters, 3, padding=1),
            ])
        self.final = PaddedConv2D(filters,1)

    def call(self, inputs):
        return self.final(self.down(inputs))
'''
#%%
class Resnet_Block(keras.layers.Layer):
    def __init__(self, output_dim, **kwargs):
        super(Resnet_Block,self).__init__(**kwargs)
        self.output_dim = output_dim
        self.Conv1 = Conv2D_Block(output_dim, 3, padding=1)
        self.Conv2 = Conv2D_Block(output_dim, 3, padding=1)

    def build(self, input_shape):
        if input_shape[-1] != self.output_dim:
            self.residual_projection = PaddedConv2D(self.output_dim, 1)
        else:
            self.residual_projection = lambda x: x
            
    def call(self, inputs):
        x = self.Conv2(self.Conv1(inputs))
        return x + self.residual_projection(inputs)
'''
noise = keras.layers.Input(shape=(32, 32, 1), name="image_input")
a = Resnet_Block(64)(noise)
'''
class ResNet_Step(keras.layers.Layer):
    def __init__(self, output_dim, **kwargs):
        super(ResNet_Step,self).__init__(**kwargs)
        self.output_dim = output_dim
        self.Conv1 = Conv2D_Block(output_dim, 3, padding=1)
        self.Embeddings = keras.Sequential([keras.layers.Activation("tanh"),
                                            keras.layers.Dense(output_dim)])
        self.Conv2 = Conv2D_Block(output_dim, 3, padding=1)

    def build(self, input_shape):
        if input_shape[0][-1] != self.output_dim:
            self.residual_projection = PaddedConv2D(self.output_dim, 1)
        else:
            self.residual_projection = lambda x: x

    def call(self, inputs):
        inputs, embeddings = inputs
        x = self.Conv1(inputs)
        embeddings = self.Embeddings(embeddings)
        x = x + embeddings[:, None, None]
        x = self.Conv2(x)
        return x + self.residual_projection(inputs)

class Expand_Block(keras.layers.Layer):
    def __init__(self, filters, kernel_size, strides=1,padding='same', **kwargs):
        super(Expand_Block,self).__init__(**kwargs)
        self.ConvT = keras.layers.Conv2DTranspose(filters,kernel_size,strides,padding)
        self.Conv = keras.layers.Conv2D(filters, 3, padding="same")
        self.norm = keras.layers.GroupNormalization(epsilon=1e-5)
        self.active=keras.layers.Activation("swish")
        #self.active = activation_fn
        #self.active=keras.layers.LeakyReLU(0.8)
        #self.norm = keras.layers.BatchNormalization(momentum=0.8)
    def call(self, inputs):
        return self.active(self.norm(self.Conv(self.ConvT(inputs))))
'''
variable=keras.layers.Input(shape=(1,32,18), name="variable_input")
x=Conv2DTranspose_Block(256, (4, 1),padding='valid')(variable)
x=Conv2DTranspose_Block(128, kernel_size=(3, 1))(x)
'''

class condition_Variable(keras.layers.Layer):
    def __init__(self, output_dim, **kwargs):
        super(condition_Variable,self).__init__(**kwargs)
        self.output_dim = output_dim
        self.Dense = keras.layers.Dense(256)
        self.Expand = keras.Sequential([Expand_Block(256, kernel_size=(4, 1),padding='valid'),
                                        Expand_Block(128, kernel_size=(3, 1),strides=(2,1)),
                                        Expand_Block(64, kernel_size=(3, 1),strides=(2,1)),
                                        Expand_Block(64, kernel_size=(3, 1),strides=(2,1)),
                                        keras.layers.Activation("softmax")])
        self.Conv = keras.Sequential([keras.layers.Conv2D(64, 3, padding="same"),
                                      keras.layers.LeakyReLU(0.2),
                                      keras.layers.BatchNormalization(momentum=0.8)])
        self.Concat = keras.layers.Concatenate(axis=-1)
        self.final = keras.Sequential([PaddedConv2D(64, kernel_size=3, padding=1),
                                       PaddedConv2D(output_dim, 1)])
    def build(self,input_shape):
        self.lv = input_shape[1][1]
        self.cv = input_shape[1][-1]
        self.ln = input_shape[0][1]
        self.reshape = keras.layers.Reshape((1,self.ln,(self.lv//self.ln)*self.cv))

    def call(self, inputs):
        noise,variable = inputs
        x = self.Conv(self.Expand(self.Dense(self.reshape(variable))))
        return self.final(self.Concat([x,noise]))

'''
variable = keras.layers.Input(shape=(64, 9), name="variable_input")
noise = keras.layers.Input(shape=(32, 32, 1), name="image_input")

noise = np.random.uniform(low=-3, high=3, size=(8,32,32,1))
variable = np.random.uniform(low=0, high=1, size=(8, 64, 9))
a=condition_Variable(16)([noise,variable])
'''

#%%
'''transformer用，实际作用未知'''
class GEGLU(keras.layers.Layer):
    def __init__(self, output_dim, **kwargs):
        super(GEGLU,self).__init__(**kwargs)
        self.output_dim = output_dim
        self.dense = keras.layers.Dense(output_dim * 2)

    def call(self, inputs):
        x = self.dense(inputs)
        x, gate = x[..., : self.output_dim], x[..., self.output_dim :]
        tanh_res = keras.activations.tanh(gate * 0.7978845608 * (1 + 0.044715 * (gate**2)))
        return x * 0.5 * gate * (1 + tanh_res)

'''位置编码'''
class PositionalEmbedding(keras.layers.Layer):
    def __init__(self, **kwargs):
        super(PositionalEmbedding,self).__init__()
        self.normal = keras.layers.LayerNormalization(epsilon=1e-5)
        
    def build(self,input_shape):
        if len(input_shape)==4:
            _,self.h, self.w, self.c=input_shape
            self.long = self.h*self.w
            self.Embeding = keras.layers.Embedding(self.long, self.c)
            self.first = keras.layers.Reshape((self.long, self.c))
            self.final = keras.layers.Reshape((self.h,self.w,self.c))
        elif len(input_shape)==3:
            _,self.long,self.c=input_shape
            self.first = lambda x: x
            self.Embeding = keras.layers.Embedding(self.long, self.c)
            self.final = lambda x: x
            
    def positional_encoding(self,length, depth):
        depth = tf.cast(depth // 2, tf.int32)# 将depth转换为整数并除以2
        positions = tf.range(length, dtype=tf.float32)[:, tf.newaxis]# 生成位置数组 (length, 1)
        depths = tf.range(depth, dtype=tf.float32)[tf.newaxis, :]/tf.cast(depth, tf.float32)# 生成深度数组 (1, depth)
        angle_rates = 1.0 / (tf.pow(10000.0, depths))# 计算角度率 (1, depth)
        angle_rads = positions * angle_rates# 计算角度弧度 (length, depth)
        pos_encoding = tf.concat([tf.sin(angle_rads), tf.cos(angle_rads)], axis=-1)# 拼接正弦和余弦编码
        return pos_encoding

    def call(self, inputs):
        x = self.first(self.normal(inputs))
        y = tf.expand_dims(tf.range(self.long, dtype=tf.int32), axis=0)
        y = self.Embeding(y)*20
        z = self.positional_encoding(self.long, self.c)[tf.newaxis]
        return self.final(x+y+z)
'''
noise = keras.layers.Input(shape=(32, 32, 64), name="image_input")
varible = keras.layers.Input(shape=(64,18), name="varible_input")

noise = np.random.uniform(low=-3, high=3, size=(16, 32,32, 32))
varible = np.random.uniform(low=0, high=1, size=(16, 64, 18))

a=PositionalEmbedding()(noise)
a = a.numpy()
b=PositionalEmbedding()(varible)
b = b.numpy()

class PositionalEmbedding(keras.layers.Layer):
    def __init__(self,**kwargs):
        super(PositionalEmbedding,self).__init__()
    
    def build(self,input_shape):
        _,self.h,self.w,self.c=input_shape
        self.reshape1 = keras.layers.Reshape((self.h*self.w, self.c))
        self.Embedding = keras.layers.Embedding(self.h*self.w, self.c)
        self.reshape2 = keras.layers.Reshape((self.h, self.w, self.c))
        
    def call(self,image):
        positions = tf.expand_dims(tf.range(self.h*self.w, dtype=tf.int32), axis=0)
        y = self.Embedding(positions)
        x = self.reshape1(image)
        return self.reshape2(x + y)
'''

class Attention(keras.layers.Layer):
    def __init__(self,channel,head,**kwargs):
        super(Attention, self).__init__(**kwargs)
        self.head = head
        self.channel = channel
        self.dense_q = keras.layers.Dense(channel*head, use_bias=False)
        self.dense_k = keras.layers.Dense(channel*head, use_bias=False)
        self.dense_v = keras.layers.Dense(channel*head, use_bias=False)
        
    def call(self, inputs,context=None):
        if context is None:
            context = inputs
        q = self.dense_q(inputs)
        k = self.dense_k(context)
        v = self.dense_v(context)
        batch = tf.shape(q)[0]
        q_seq = tf.shape(q)[1]
        k_seq = tf.shape(k)[1]
        q = tf.reshape(q, [batch, q_seq, self.head, self.channel])
        k = tf.reshape(k, [batch, k_seq, self.head, self.channel])
        v = tf.reshape(v, (batch, k_seq, self.head, self.channel))
        q = tf.transpose(q, perm=[0, 2, 1, 3])  # (batch, heads, q_seq, dim)
        k = tf.transpose(k, perm=[0, 2, 1, 3])  # (batch, heads, k_seq, dim)
        v = tf.transpose(v, perm=[0, 2, 1, 3])  # (batch, heads, k_seq, dim)
        matmul_qk = tf.matmul(q, k, transpose_b=True)
        dk = tf.cast(tf.shape(k)[-1], q.dtype)
        score = matmul_qk / tf.math.sqrt(dk)
        weights = tf.nn.softmax(score)  # (bs, num_heads, time, time)
        #weights = keras.activations.softmax(score,axis=[-2,-1])
        out = tf.matmul(weights, v)
        out = tf.reshape(tf.transpose(out, perm=[0, 2, 1, 3]),(batch,q_seq,self.head*self.channel))
        return out

class Self_Attention(keras.layers.Layer):
    def __init__(self,channel,head,**kwargs):
        super(Self_Attention, self).__init__(**kwargs)
        self.channel = channel
        self.head = head
        self.Attention = Attention(self.channel,self.head)

    def build(self, input_shape):
        self.dim = input_shape# 获取最后一个维度
        if len(self.dim)==4:
            _,self.h,self.w,self.c = self.dim
            self.norml = keras.layers.GroupNormalization(epsilon=1e-5)
            self.first = keras.layers.Reshape((self.h*self.w,self.c))
            self.final = keras.Sequential([keras.layers.Reshape((self.h,self.w,self.head*self.channel)),
                                           PaddedConv2D(self.dim[-1], 3,padding=1)])
        elif len(self.dim)==3:
            self.norml = keras.layers.LayerNormalization(epsilon=1e-5)
            self.first = lambda x: x
            #self.first = keras.layers.Dense(self.channel)
            self.final = keras.layers.Dense(self.dim[-1])

    def call(self, inputs):
        x = self.first(self.norml(inputs))
        x = self.Attention(x)
        return self.final(x)+inputs
'''

def U_Net(c):
    noise = keras.layers.Input(shape=(32, 32, 64), name="image_input")
    #variable = keras.layers.Input(shape=(64, 18), name="variable_input")
    x = Self_Attention(c,2)(noise)
    model= Model(inputs=noise, outputs=x, name="unet")
    return model
net=U_Net(32)
net.summary()
net.save_weights('C:/Users/59278/Desktop/Self_Attention.h5')


inputs = keras.layers.Input(shape=(32, 32, 64), name="image_input")
context = keras.layers.Input(shape=(64, 18), name="variable_input")
x = Self_Attention(32,8)(inputs) + inputs
x.shape
y = Self_Attention(32,8)(context)+ context
y.shape
'''
class Cross_Attention(keras.layers.Layer):
    def __init__(self,channel,head,**kwargs):
        super(Cross_Attention, self).__init__(**kwargs)
        self.channel = channel
        self.head = head
        self.Attention = Attention(self.channel,self.head)

    def build(self, input_shape):
        self.dim1 = input_shape[0]
        if len(self.dim1)==4:
            _,self.h,self.w,self.c = self.dim1
            self.norm1 = keras.layers.GroupNormalization(epsilon=1e-5)
            self.first1 = keras.layers.Reshape((self.h*self.w,self.c))
            self.final = keras.Sequential([keras.layers.Reshape((self.h,self.w,self.head*self.channel)),
                                           PaddedConv2D(self.dim1[-1], 3,padding=1)])
        elif len(self.dim1)==3:
            self.norm1 = keras.layers.LayerNormalization(epsilon=1e-5)
            self.first1 = lambda x: x
            self.final = keras.layers.Dense(self.dim1[-1])
            
        self.dim2 = input_shape[1]
        if len(self.dim2)==4:
            _,self.h2,self.w2,self.c2 = self.dim2
            self.norm2 = keras.layers.GroupNormalization(epsilon=1e-5)
            self.first2 = keras.layers.Reshape((self.h2*self.w2,self.c2))
        elif len(self.dim2)==3:
            self.norm2 = keras.layers.LayerNormalization(epsilon=1e-5)
            self.first2 = lambda x: x

    def call(self, inputs):
        noise,varb = inputs
        x = self.first1(self.norm1(noise))
        y = self.first2(self.norm2(varb))
        z = self.Attention(x,y)
        return self.final(z)+noise
'''
inputs = keras.layers.Input(shape=(32, 32, 8), name="image_input")
context = keras.layers.Input(shape=(64, 18), name="variable_input")

inputs = np.random.uniform(low=-1, high=1, size=(16, 32,32, 32))
context = np.random.uniform(low=-1, high=1, size=(16, 64, 32))
x1 = Cross_Attention(32,4)([inputs,context])+ inputs
x1.shape
x2 = Cross_Attention(32,4)([context,inputs])+ context
x2.shape
y1 = Cross_Attention(32,4)([inputs,inputs])+ inputs
y1.shape
y2 = Cross_Attention(32,4)([context,context])+ context
y2.shape
'''

class Cross_Transformer(keras.layers.Layer):
    def __init__(self,channel,head,layers,**kwargs):
        super(Cross_Transformer,self).__init__(**kwargs)
        self.channel = channel
        self.head = head
        self.layers= layers
        self.atten_block=[Cross_Attention(self.channel,self.head) for _ in range(self.layers)]
        self.geglu = GEGLU(channel * 4)
        
    def build(self, input_shape):
        self.dim = input_shape[0]
        if len(self.dim) == 3:
            self.norml = keras.layers.LayerNormalization(epsilon=1e-5)
            self.final = keras.layers.Dense(self.dim[-1])
        elif len(self.dim) == 4:
            self.norml = keras.layers.GroupNormalization(epsilon=1e-5)
            self.final = PaddedConv2D(self.dim[-1],3,padding=1)

    def call(self,inputs):
        noise,varb = inputs
        x = noise
        for atten in self.atten_block:
            x = atten([x,varb])
        out = self.final(self.geglu(self.norml(x))) + x
        return out

'''

def U_Net(c):
    noise = keras.layers.Input(shape=(32, 32, 64), name="image_input")
    variable = keras.layers.Input(shape=(64, 18), name="variable_input")
    x = Cross_Transformer(c,2,2)([noise,variable])
    model= Model(inputs=[noise,variable], outputs=x, name="unet")
    return model
net=U_Net(32)
net.summary()
net.save_weights('C:/Users/59278/Desktop/Cross_Transformer.h5')


inputs = keras.layers.Input(shape=(32, 32, 32), name="image_input")
context = keras.layers.Input(shape=(64, 32), name="variable_input")
a=Cross_Transformer(32,4,2)([inputs,context])
a.shape

inputs = np.random.uniform(low=-3, high=3, size=(16, 32,32, 32))
context = np.random.uniform(low=-1, high=1, size=(16, 64, 8))
a=Cross_Transformer(32,4,2)([inputs,context])
a=a.numpy()
'''

class Basic_Transformer(keras.layers.Layer):
    def __init__(self,channel,head,layers,**kwargs):
        super(Basic_Transformer,self).__init__(**kwargs)
        self.channel = channel
        self.head = head
        self.layers= layers
        self.atten_block = [Self_Attention(self.channel,self.head) for _ in range(self.layers)]
        self.geglu = GEGLU(channel * 4)
        
    def build(self, input_shape):
        if len(input_shape) == 3:
            self.normal = keras.layers.LayerNormalization(epsilon=1e-5)
            self.final_out = keras.layers.Dense(input_shape[-1])
        elif len(input_shape) == 4:
            self.normal = keras.layers.GroupNormalization(epsilon=1e-5)
            self.final_out = PaddedConv2D(input_shape[-1],3,padding=1)

    def call(self,inputs):
        x = inputs
        for atten in self.atten_block:
            x = atten(x)
        out = self.final_out(self.geglu(self.normal(x))) + x
        return out

'''
def U_Net(c):
    noise = keras.layers.Input(shape=(32, 32, 64), name="image_input")
    #variable = keras.layers.Input(shape=(64, 18), name="variable_input")
    x = Basic_Transformer(c,2,2)(noise)
    model= Model(inputs=noise, outputs=x, name="unet")
    return model
net=U_Net(32)
net.summary()
net.save_weights('C:/Users/59278/Desktop/Basic_Transformer.h5')

inputs = keras.layers.Input(shape=(32, 32, 32), name="image_input")
context = keras.layers.Input(shape=(64, 18), name="variable_input")

a=Basic_Transformer(32,4,2)(inputs)
a.shape
b=Basic_Transformer(32,4,2)(context)
b.shape

inputs = np.random.uniform(low=-3, high=3, size=(16, 32,32, 32))
context = np.random.uniform(low=-1, high=1, size=(16, 64, 8))

a=Basic_Transformer(32,4,2)(inputs)
a=a.numpy()
'''

class SpatialTransformer(keras.layers.Layer):
    def __init__(self,channel,head,layers,**kwargs):
        super(SpatialTransformer,self).__init__(**kwargs)
        self.channel = channel
        self.head = head
        self.layers= layers
        #self.PE3 = keras.layers.Permute((2, 1))
        self.norm1 = keras.layers.GroupNormalization(epsilon=1e-5)
        self.norm2 = keras.layers.LayerNormalization(epsilon=1e-5)
        self.norm3 = keras.layers.LayerNormalization(epsilon=1e-5)
        self.first1 = PaddedConv2D(self.channel,3,padding=1)
        self.first2 = keras.layers.Dense(self.channel)
        self.first3 = keras.layers.Dense(self.channel)
        self.self_atten1 = Basic_Transformer(self.channel,self.head,1)
        self.self_atten2 = Basic_Transformer(self.channel,self.head,1)
        #self.self_atten3 = Basic_Transformer(self.channel,self.head,1)
        self.cross_atten1 = Cross_Transformer(self.channel,self.head,self.layers)
        #self.cross_atten2 = Cross_Transformer(self.channel,self.head,self.layers)
        self.self_atten4 = Basic_Transformer(self.channel,self.head,1)
        #self.final_out = PaddedConv2D(dim, 1)
        
    def build(self, input_shape):
        self.final_out = PaddedConv2D(input_shape[0][-1], 1)

    def call(self, inputs):
        noise,varb = inputs
        x = self.self_atten1(self.first1(self.norm1(noise)))
        y1 = self.self_atten2(self.first2(self.norm2(varb)))
        #y2 = self.self_atten3(self.first3(self.norm3(self.PE3(varb))))
        x = self.cross_atten1([x,y1])
        #x = self.cross_atten2([x,y2])
        x = self.self_atten4(x)
        return self.final_out(x) + noise
'''
inputs = keras.layers.Input(shape=(32, 32, 32), name="image_input")
context = keras.layers.Input(shape=(64, 18), name="variable_input")

def U_Net(c):
    noise = keras.layers.Input(shape=(32, 32, 64), name="image_input")
    variable = keras.layers.Input(shape=(64, 18), name="variable_input")
    x = SpatialTransformer(c,2,2)([noise,variable])
    model= Model(inputs=[noise,variable], outputs=x, name="unet")
    return model
net=U_Net(32)
net.summary()
net.save_weights('C:/Users/59278/Desktop/SpatialTransformer.h5')

inputs = np.random.uniform(low=-3, high=3, size=(16, 32,32, 32))
context = np.random.uniform(low=-1, high=1, size=(16, 64,18))

a=SpatialTransformer(32,4,2)([inputs,context])
a=a.numpy()
'''

def U_Net(img_size,variable_long,variable_num):
    cyclic=1
    channel_rate=1
    head=1
    layers=1
    '''
    noise = keras.layers.Input(shape=(32, 32, 1), name="image_input")
    variable = keras.layers.Input(shape=(64, 18), name="variable_input")
    step = keras.layers.Input(shape=(), dtype=tf.int32, name="time_input")
    
    noise = np.random.uniform(low=-3, high=3, size=(8,32,32,1))
    variable = np.random.uniform(low=0, high=1, size=(8, 64, 9))
    step = np.random.randint(0, 100, size=(noise.shape[0],), dtype=np.int32)
    '''
    noise = keras.layers.Input(shape=(img_size, img_size, 1), name="image_input")
    variable = keras.layers.Input(shape=(variable_long,variable_num), name="variable_input")
    step = keras.layers.Input(shape=(), dtype=tf.int32, name="time_input")
    '''时间编码'''
    step_emb = TimeEmbedding(128*channel_rate)(step)
    '''噪声初始变换'''
    x = condition_Variable(32*channel_rate)([noise,variable])

    outputs = []
    x = Resnet_Block(32*channel_rate)(x)
    outputs.append(x)
    
    for _ in range(cyclic):
        x = ResNet_Step(32*channel_rate)([x, step_emb])
        x = SpatialTransformer(32*channel_rate,head,layers)([x,variable])
        outputs.append(x)
    x = Conv2D_Block(64*channel_rate, 3, strides=2, padding=1)(x)  # Downsample 2x
    outputs.append(x)
    
    for _ in range(cyclic):
        x = ResNet_Step(64*channel_rate)([x, step_emb])
        x = SpatialTransformer(64*channel_rate,head,layers)([x,variable])
        outputs.append(x)
    x = Conv2D_Block(128*channel_rate, 3, strides=2, padding=1)(x)  # Downsample 2x
    outputs.append(x)
    
    for _ in range(cyclic):
        x = ResNet_Step(128*channel_rate)([x, step_emb])
        x = SpatialTransformer(128*channel_rate,head,layers)([x,variable])
        outputs.append(x)
    x = Resnet_Block(128*channel_rate)(x)  
    outputs.append(x)
    '''
    for _ in range(cyclic):
        x = ResNet_Step(128*channel_rate)([x, step_emb])
        outputs.append(x)
    '''
    # Middle flow
    x = ResNet_Step(256*channel_rate)([x, step_emb])
    x = SpatialTransformer(256*channel_rate,head,layers)([x,variable])
    x = ResNet_Step(256*channel_rate)([x, step_emb])

    # Upsampling flow
    '''
    for _ in range(cyclic+1):
        x = keras.layers.Concatenate()([x, outputs.pop()])
        x = ResNet_Step(128*channel_rate)([x, step_emb])
    '''
    #x = Conv2DTranspose_Block(128*channel_rate, 3)(x)
    x = keras.layers.Concatenate()([x, outputs.pop()])
    x = Resnet_Block(128*channel_rate)(x) 
    for _ in range(cyclic):
        x = keras.layers.Concatenate()([x, outputs.pop()])
        x = ResNet_Step(128*channel_rate)([x, step_emb])
        x = SpatialTransformer(128*channel_rate,head,layers)([x,variable])
    
    x = keras.layers.Concatenate()([x, outputs.pop()])
    x = Conv2DT_Block(64*channel_rate, 3, strides=2)(x)
    for _ in range(cyclic):
        x = keras.layers.Concatenate()([x, outputs.pop()])
        x = ResNet_Step(64*channel_rate)([x, step_emb])
        x = SpatialTransformer(64*channel_rate,head,layers)([x,variable])
    
    x = keras.layers.Concatenate()([x, outputs.pop()])
    x = Conv2DT_Block(32*channel_rate, 3, strides=2)(x)
    for _ in range(cyclic):
        x = keras.layers.Concatenate()([x, outputs.pop()])
        x = ResNet_Step(32*channel_rate)([x, step_emb])
        x = SpatialTransformer(32*channel_rate,head,layers)([x,variable])
        #x = Resnet_Block(32*channel_rate)(x)
    x = keras.layers.Concatenate()([x, outputs.pop()])
    x = Resnet_Block(32*channel_rate)(x)
    x = keras.layers.GroupNormalization(epsilon=1e-5)(x)
    x = keras.layers.Activation("swish")(x)
    #x = keras.layers.LeakyReLU(0.8)(x)
    x = PaddedConv2D(32,3,padding=1)(x)
    output = PaddedConv2D(1, 1)(x)
    #x = PaddedConv2D(1, 1)(x)
    #output = activation_fn(x)
    model= Model(inputs=[noise,step,variable], outputs=output, name="unet")
    return model

'''

net=U_Net(32,64,10)
net.summary()
net.save_weights('C:/Users/59278/Desktop/unet.h5')

from tensorflow.keras.utils import plot_model
plot_model(net,to_file='C:/Users/59278/Desktop/1111.png',show_shapes=True, show_layer_names=True)

net=U_Net(32,64,17)
noise = np.random.uniform(low=-3, high=3, size=(8,32,32,1))
varible = np.random.uniform(low=0, high=1, size=(8, 64, 17))
step = np.random.randint(0, 100, size=(noise.shape[0],), dtype=np.int64)
a=net([noise,step,varible])

a=a.numpy()

aa=a[0].numpy()
bb=a[1].numpy()
'''

"""
高斯扩散功能.
Args:
    beta_start: 计划方差的起始值
    beta_end: 计划方差的最终值
    timesteps: 前进过程中的时间步数
"""
class GaussianDiffusion:
    def __init__(
        self,
        beta_start=1e-4,
        beta_end=0.02,
        timesteps=1000,
        clip_min=-1.0,
        clip_max=1.0,
        dtype=tf.float32
    ):
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.timesteps = timesteps
        self.clip_min = clip_min
        self.clip_max = clip_max
        self.dtype=dtype

        # Define the linear variance schedule
        self.betas = betas = np.linspace(beta_start,beta_end,timesteps,dtype=np.float64)  # Using float64 for better precision
        self.num_timesteps = int(timesteps)

        alphas = 1.0 - betas
        alphas_cumprod = np.cumprod(alphas, axis=0)
        alphas_cumprod_prev = np.append(1.0, alphas_cumprod[:-1])

        self.betas = tf.constant(betas, dtype=dtype)
        self.alphas_cumprod = tf.constant(alphas_cumprod, dtype=dtype)
        self.alphas_cumprod_prev = tf.constant(alphas_cumprod_prev, dtype=dtype)

        '''√aˉ'''
        self.sqrt_alphas_cumprod = tf.constant(np.sqrt(alphas_cumprod), dtype=dtype)
        '''√1-aˉ'''
        self.sqrt_one_minus_alphas_cumprod = tf.constant(np.sqrt(1.0 - alphas_cumprod), dtype=dtype)
        '''log(1-aˉ)'''
        self.log_one_minus_alphas_cumprod = tf.constant(np.log(1.0 - alphas_cumprod), dtype=dtype)
        '''1/√aˉ'''
        self.sqrt_recip_alphas_cumprod = tf.constant(np.sqrt(1.0 / alphas_cumprod), dtype=dtype)
        '''√(1/aˉ-1)'''
        self.sqrt_recipm1_alphas_cumprod = tf.constant(np.sqrt(1.0 / alphas_cumprod - 1), dtype=dtype)

        # Calculations for posterior q(x_{t-1} | x_t, x_0)
        self.denoise_variance = tf.constant((betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)), dtype=dtype)
        self.denoise_log_variance = tf.constant(np.log(np.maximum(self.denoise_variance, 1e-20)), dtype=dtype)
        self.mean_part1 = tf.constant(betas * np.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod),dtype=dtype,)
        self.mean_part2 = tf.constant((1.0 - alphas_cumprod_prev) * np.sqrt(alphas) / (1.0 - alphas_cumprod),dtype=dtype,)

    def index_t(self,batch_size,a, t):
        #t = tf.cast(t, dtype=tf.int32)
        #print(t.shape,a.shape)
        out = tf.gather(a, t)
        #print('匹配ok')
        return tf.reshape(out, [batch_size, 1, 1, 1])

    def q_mean_variance(self, image, t):
        batch_size = tf.shape(image)[0]
        mean = self.index_t(batch_size,self.sqrt_alphas_cumprod, t) * image
        variance = self.index_t(batch_size,1.0 - self.alphas_cumprod, t)
        log_variance = self.index_t(batch_size,self.log_one_minus_alphas_cumprod, t)
        return mean, variance, log_variance

    def Add_noise(self, image, t, noise):

        batch_size = tf.shape(image)[0]
        add_image=(self.index_t(batch_size,self.sqrt_alphas_cumprod, t) * image
                   + self.index_t(batch_size,self.sqrt_one_minus_alphas_cumprod, t)* noise)
        return tf.clip_by_value(add_image, self.clip_min, self.clip_max)

    def DDPM_denoise(self, pred_noise, image_t, t, clip_denoised=True):
        pred_noise = tf.cast(pred_noise, dtype=self.dtype)
        image_t = tf.cast(image_t, dtype=self.dtype)
        batch_size = tf.shape(image_t)[0]
        """
        根据x_t = √αˉ_t ⋅ x_0 + √(1−αˉ_t)⋅noise
        推出x_0 = (1/√αˉ_t)x_t - (√(1-αˉ_t)/√αˉ_t)⋅noise
        """
        x_0 = (self.index_t(batch_size,self.sqrt_recip_alphas_cumprod, t) * image_t
            - self.index_t(batch_size,self.sqrt_recipm1_alphas_cumprod, t) * pred_noise)
        """限制image的取值范围，以免超过真值的范围"""
        if clip_denoised:
            x_0 = tf.clip_by_value(x_0, self.clip_min, self.clip_max)
        """计算下一步图像分布的均值和方差"""
        mean = (self.index_t(batch_size,self.mean_part1, t) * x_0 
                + self.index_t(batch_size,self.mean_part2, t) * image_t)
        #variance = self.index_t(batch_size,self.denoise_variance, t)
        log_variance = self.index_t(batch_size,self.denoise_log_variance, t)
        """根据均值方差重采样"""
        nonzero_mask = tf.reshape(1 - tf.cast(tf.equal(t, 0), dtype=self.dtype), [tf.shape(image_t)[0], 1, 1, 1])# No noise when t == 0
        noise = tf.random.normal(shape=tf.shape(image_t), dtype=self.dtype)
        return mean + nonzero_mask * tf.exp(0.5 * log_variance) * noise
    '''
    def DDIM_denoise(self, pred_noise,image_k,t_k,t_s,clip_denoised=True):
        n=0.5
        if clip_denoised:
            image_k = tf.clip_by_value(image_k, self.clip_min, self.clip_max)
        batch_size = tf.shape(pred_noise)[0]
        """计算下一步图像分布的均值和方差"""
        alpha_s=self.index_t(batch_size,self.alphas_cumprod, t_s)
        alpha_k=self.index_t(batch_size,self.alphas_cumprod, t_k)
        beta_k=self.index_t(batch_size,self.betas, t_k)
        #x0_pred = (x_k - tf.sqrt(1.0 - alpha_k) * noise_k) / tf.sqrt(alpha_k)
        """方差(1-α_k)*(1-αˉ_s)/(1-αˉ_k)"""
        denoise_var = (beta_k * (1.0 - alpha_s) / (1.0 - alpha_k)) * n
        """log(方差)"""
        denoise_log_variance = tf.math.log(tf.maximum(tf.cast(denoise_var, dtype=tf.float32),tf.constant(1e-20, dtype=tf.float32)))
        """均值:mean=t_k * √αˉ_s / √αˉ_k + (√(1 - αˉ_s - σ2) - √(αˉ_s * (1-αˉ_k) / αˉ_k))*noise"""
        mean_part1 = tf.sqrt(alpha_s / alpha_k) * image_k
        mean_part2 = (tf.sqrt(1-alpha_s-denoise_var) - tf.sqrt(alpha_s*(1-alpha_k)/alpha_k)) * pred_noise
        denoise_mean = mean_part1 + mean_part2
        noise = tf.random.normal(shape=tf.shape(image_k), dtype=image_k.dtype)
        #print(tf.reduce_max(denoise_mean),tf.reduce_min(denoise_mean))
        nonzero_mask = tf.reshape(1 - tf.cast(tf.equal(t_k,t_s), tf.float32), [tf.shape(image_k)[0], 1, 1, 1])
        #x_s = denoise_mean + nonzero_mask*tf.sqrt(denoise_var) * noise
        x_s = denoise_mean + nonzero_mask*tf.exp(0.5 * denoise_log_variance) * noise
        if clip_denoised:
            x_s = tf.clip_by_value(x_s, self.clip_min, self.clip_max)
        return x_s
    '''
    def DDIM_denoise(self, pred_noise,image_k,t_k,t_s,clip_denoised=True):
        pred_noise = tf.cast(pred_noise, dtype=self.dtype)
        image_k = tf.cast(image_k, dtype=self.dtype)
        n=0
        batch_size = tf.shape(pred_noise)[0]
        """计算下一步图像分布的均值和方差"""
        alpha_s=self.index_t(batch_size,self.alphas_cumprod, t_s)
        alpha_k=self.index_t(batch_size,self.alphas_cumprod, t_k)
        beta_k=self.index_t(batch_size,self.betas, t_k)
        #x0_pred = (x_k - tf.sqrt(1.0 - alpha_k) * noise_k) / tf.sqrt(alpha_k)
        """方差(1-α_k)*(1-αˉ_s)/(1-αˉ_k)"""
        denoise_var = (beta_k * (1.0 - alpha_s) / (1.0 - alpha_k)) * n
        """log(方差)"""
        #denoise_log_variance = tf.math.log(tf.maximum(tf.cast(denoise_var, dtype=pred_noise.dtype),tf.constant(1e-20, dtype=pred_noise.dtype)))
        """均值:mean=t_k * √αˉ_s / √αˉ_k + (√(1 - αˉ_s - σ2) - √(αˉ_s * (1-αˉ_k) / αˉ_k))*noise"""
        mean_part1 = tf.sqrt(alpha_s / alpha_k)*(image_k-tf.sqrt(1.0 - alpha_k)*pred_noise)
        mean_part2 = tf.sqrt(1-alpha_s-denoise_var)* pred_noise
        denoise_mean = mean_part1 + mean_part2
        noise = tf.random.normal(shape=tf.shape(image_k), dtype=self.dtype)
        noise = tf.clip_by_value(noise ,-3, 3)
        #print(tf.reduce_max(denoise_mean),tf.reduce_min(denoise_mean))
        nonzero_mask = tf.reshape(1 - tf.cast(tf.equal(t_k,t_s), dtype=self.dtype), [tf.shape(image_k)[0], 1, 1, 1])
        #x_s = denoise_mean + nonzero_mask*tf.sqrt(denoise_var) * noise
        x_s = denoise_mean + nonzero_mask*denoise_var*noise
        if clip_denoised:
            x_s = tf.clip_by_value(x_s, self.clip_min, self.clip_max)
        return x_s
    

'''
class GaussianDiffusion:
    def __init__(
        self,
        beta_start=1e-4,
        beta_end=0.02,
        timesteps=1000,
        clip_min=-1.0,
        clip_max=1.0,
    ):
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.timesteps = timesteps
        self.clip_min = clip_min
        self.clip_max = clip_max

        # Define the linear variance schedule
        self.betas = betas = np.linspace(beta_start,beta_end,timesteps,dtype=np.float32,)  # Using float64 for better precision
        
        self.num_timesteps = int(timesteps)

        alphas = 1.0 - betas
        alphas_cumprod = np.cumprod(alphas, axis=0)
        alphas_cumprod_prev = np.append(1.0, alphas_cumprod[:-1])

        self.betas = tf.constant(betas, dtype=tf.float32)
        self.alphas_cumprod = tf.constant(alphas_cumprod, dtype=tf.float32)
        self.alphas_cumprod_prev = tf.constant(alphas_cumprod_prev, dtype=tf.float32)

        # Calculations for diffusion q(x_t | x_{t-1}) and others
        self.sqrt_alphas_cumprod = tf.constant(np.sqrt(alphas_cumprod), dtype=tf.float32)

        self.sqrt_one_minus_alphas_cumprod = tf.constant(np.sqrt(1.0 - alphas_cumprod), dtype=tf.float32)

        self.log_one_minus_alphas_cumprod = tf.constant(np.log(1.0 - alphas_cumprod), dtype=tf.float32)

        self.sqrt_recip_alphas_cumprod = tf.constant(np.sqrt(1.0 / alphas_cumprod), dtype=tf.float32)
        
        self.sqrt_recipm1_alphas_cumprod = tf.constant(np.sqrt(1.0 / alphas_cumprod - 1), dtype=tf.float32)

        # Calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = (betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod))
        
        self.posterior_variance = tf.constant(posterior_variance, dtype=tf.float32)

        # Log calculation clipped because the posterior variance is 0 at the beginning
        # of the diffusion chain
        self.posterior_log_variance_clipped = tf.constant(np.log(np.maximum(posterior_variance, 1e-20)), dtype=tf.float32)

        self.posterior_mean_coef1 = tf.constant(betas * np.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod),dtype=tf.float32,)

        self.posterior_mean_coef2 = tf.constant((1.0 - alphas_cumprod_prev) * np.sqrt(alphas) / (1.0 - alphas_cumprod),dtype=tf.float32,)

    def _extract(self, a, t, x_shape):
        """
        在指定的时间步长提取一些系数，然后重塑为[batch_size, 1,1,1,1，…]]作扩散之用。
        Args:
            a: 要提取的张量
            t: 要提取其系数的时间步长
            x_shape: 当前批样的形状
        """
        batch_size = x_shape[0]
        out = tf.gather(a, t)
        return tf.reshape(out, [batch_size, 1, 1, 1])

    def q_mean_variance(self, x_start, t):
        """
        提取当前时间步长的均值和方差.
        Args:
            x_start: 初始样品(在第一个扩散步骤之前)
            t: 等于当前时间步长
        """
        x_start_shape = tf.shape(x_start)
        mean = self._extract(self.sqrt_alphas_cumprod, t, x_start_shape) * x_start
        variance = self._extract(1.0 - self.alphas_cumprod, t, x_start_shape)
        log_variance = self._extract(self.log_one_minus_alphas_cumprod, t, x_start_shape)
        return mean, variance, log_variance

    def q_sample(self, x_start, t, noise):
        """
        数据扩散.
        Args:
            x_start: Initial sample (before the first diffusion step)
            t: Current timestep
            noise: Gaussian noise to be added at the current timestep
        Returns:
            Diffused samples at timestep `t`
        """
        x_start_shape = tf.shape(x_start)
        return (self._extract(self.sqrt_alphas_cumprod, t, tf.shape(x_start)) * x_start
                + self._extract(self.sqrt_one_minus_alphas_cumprod, t, x_start_shape) * noise)

    def predict_start_from_noise(self, x_t, t, noise):
        x_t_shape = tf.shape(x_t)
        return (self._extract(self.sqrt_recip_alphas_cumprod, t, x_t_shape) * x_t
                - self._extract(self.sqrt_recipm1_alphas_cumprod, t, x_t_shape) * noise)

    def q_posterior(self, x_start, x_t, t):
        """Compute the mean and variance of the diffusion
        posterior q(x_{t-1} | x_t, x_0).

        Args:
            x_start: Stating point(sample) for the posterior computation
            x_t: Sample at timestep `t`
            t: Current timestep
        Returns:
            Posterior mean and variance at current timestep
        """
        x_t_shape = tf.shape(x_t)
        posterior_mean = (self._extract(self.posterior_mean_coef1, t, x_t_shape) * x_start
                          + self._extract(self.posterior_mean_coef2, t, x_t_shape) * x_t)
        posterior_variance = self._extract(self.posterior_variance, t, x_t_shape)
        posterior_log_variance_clipped = self._extract(self.posterior_log_variance_clipped, t, x_t_shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def p_mean_variance(self, pred_noise, x, t, clip_denoised=True):
        x_recon = self.predict_start_from_noise(x, t=t, noise=pred_noise)
        if clip_denoised:
            x_recon = tf.clip_by_value(x_recon, self.clip_min, self.clip_max)
        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(x_start=x_recon, x_t=x, t=t)
        return model_mean, posterior_variance, posterior_log_variance

    def p_sample(self, pred_noise, x, t, clip_denoised=True):
        """Sample from the diffusion model.

        Args:
            pred_noise: Noise predicted by the diffusion model
            x: Samples at a given timestep for which the noise was predicted
            t: Current timestep
            clip_denoised (bool): Whether to clip the predicted noise
                within the specified range or not.
        """
        model_mean, _, model_log_variance = self.p_mean_variance(pred_noise, x=x, t=t, clip_denoised=clip_denoised)
        noise = tf.random.normal(shape=x.shape, dtype=x.dtype, seed=1337)
        # No noise when t == 0
        nonzero_mask = tf.reshape(1 - tf.cast(tf.equal(t, 0), tf.float32), [tf.shape(x)[0], 1, 1, 1])
        return model_mean + nonzero_mask * tf.exp(0.5 * model_log_variance) * noise
'''