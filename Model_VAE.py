# -*- coding: utf-8 -*-
"""
Created on Wed Dec 11 13:44:32 2024

@author: 59278
"""
import numpy as np
import tensorflow as tf
import keras

'''创建采样层'''
class Sampling(keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.seed_generator = 1337#tf.random.Generator.from_seed(1337)

    def call(self, inputs):
        z_mean, z_log_var = inputs
        noise = tf.random.normal(shape=tf.shape(z_mean), seed=1337,dtype = z_mean.dtype)
        #noise = tf.random.normal(shape=tf.shape(z_mean))
        #epsilon = tf.clip_by_value(epsilon, -1.0, 1.0)
        out = z_mean + tf.exp(0.5 * z_log_var) * noise
        return tf.clip_by_value(out, -3.0, 3.0)

class PaddedConv2D(keras.layers.Layer):
    def __init__(self, filters, kernel_size, padding=0, strides=1, **kwargs):
        super().__init__(**kwargs)
        self.padding2d = keras.layers.ZeroPadding2D(padding)
        self.conv2d = keras.layers.Conv2D(filters, kernel_size, strides=strides)

    def call(self, inputs):
        x = self.padding2d(inputs)
        return self.conv2d(x)

class PaddedConv2DT(keras.layers.Layer):
    def __init__(self, filters, kernel_size, padding=0, strides=1, **kwargs):
        super().__init__(**kwargs)
        #self.Normal = keras.layers.LayerNormalization(epsilon=1e-5)
        #self.activ = keras.layers.Activation("swish")
        # 转置卷积（逆卷积）
        self.conv2dT = keras.layers.Conv2DTranspose(filters,kernel_size,strides=strides,padding='same')
        # 裁剪掉原始填充部分
        #self.cropping2d = keras.layers.Cropping2D(padding)# 裁剪量与原填充量一致
    def call(self, inputs):
        x = self.conv2dT(inputs)   # 先进行转置卷积
        return x#self.cropping2d(x)   # 再裁剪填充区域

class ResnetBlock(keras.layers.Layer):
    def __init__(self, output_dim, **kwargs):
        super().__init__(**kwargs)
        self.output_dim = output_dim
        self.norm1 = keras.layers.BatchNormalization(epsilon=1e-5)
        self.conv1 = PaddedConv2D(output_dim, 3, padding=1)
        self.norm2 = keras.layers.BatchNormalization(epsilon=1e-5)
        self.conv2 = PaddedConv2D(output_dim, 3, padding=1)

    def build(self, input_shape):
        if input_shape[-1] != self.output_dim:
            self.residual_projection = PaddedConv2D(self.output_dim, 1)
        else:
            self.residual_projection = lambda x: x

    def call(self, inputs):
        x = self.conv1(keras.activations.swish(self.norm1(inputs)))
        x = self.conv2(keras.activations.swish(self.norm2(x)))
        return x + self.residual_projection(inputs)

class AttentionBlock(keras.layers.Layer):
    def __init__(self, output_dim, **kwargs):
        super().__init__(**kwargs)
        self.output_dim = output_dim
        self.norm = keras.layers.BatchNormalization(epsilon=1e-5)
        self.q = PaddedConv2D(output_dim, 1)
        self.k = PaddedConv2D(output_dim, 1)
        self.v = PaddedConv2D(output_dim, 1)
        self.proj_out = PaddedConv2D(output_dim, 1)

    def call(self, inputs):
        x = self.norm(inputs)
        q, k, v = self.q(x), self.k(x), self.v(x)

        # Compute attention
        shape = tf.shape(q)
        h, w, c = shape[1], shape[2], shape[3]
        q = tf.reshape(q, (-1, h * w, c))  # b, hw, c
        k = tf.transpose(k, (0, 3, 1, 2))
        k = tf.reshape(k, (-1, c, h * w))  # b, c, hw
        y = q @ k
        y = y * 1 / tf.sqrt(tf.cast(c, self.compute_dtype))
        y = keras.activations.softmax(y)

        # Attend to values
        v = tf.transpose(v, (0, 3, 1, 2))
        v = tf.reshape(v, (-1, c, h * w))
        y = tf.transpose(y, (0, 2, 1))
        x = v @ y
        x = tf.transpose(x, (0, 2, 1))
        x = tf.reshape(x, (-1, h, w, c))
        return self.proj_out(x) + inputs
n=2
#
'''构建编码器'''
def Encoder(img_height, img_width):
    """ImageEncoder is the VAE Encoder for StableDiffusion."""
    '''
    img_height, img_width=64,128
    '''
    inputs = keras.layers.Input(shape=(img_height, img_width, 1), name="image_input")
    x = PaddedConv2D(1, 1)(inputs)
    x = PaddedConv2D(128//n, 1)(x)

    x = ResnetBlock(128//n)(x)
    x = PaddedConv2D(128//n, 1)(x)
    x = ResnetBlock(128//n)(x)
    x = PaddedConv2D(128//n, 3, padding=1)(x)
    x = ResnetBlock(128//n)(x)
    x = PaddedConv2D(128//n, 3, padding=1)(x)
    x = ResnetBlock(128//n)(x)
    x = PaddedConv2D(128//n, 1)(x)
    x = ResnetBlock(128//n)(x)

    x = ResnetBlock(256//n)(x)
    x = PaddedConv2D(256//n, 1)(x)
    x = ResnetBlock(256//n)(x)
    x = PaddedConv2D(256//n, 1)(x)
    x = ResnetBlock(256//n)(x)
    x = PaddedConv2D(256//n, 3, padding=1)(x)
    x = ResnetBlock(256//n)(x)
    x = PaddedConv2D(256//n, 3, padding=1)(x)
    x = ResnetBlock(256//n)(x)
    x = PaddedConv2D(256//n, 3, padding=1, strides=(1,2))(x)
    x = ResnetBlock(256//n)(x)
    x = PaddedConv2D(256//n, 3, padding=1)(x)
    x = ResnetBlock(256//n)(x)
    x = PaddedConv2D(256//n, 3, padding=1)(x)
    x = ResnetBlock(256//n)(x)
    x = PaddedConv2D(256//n, 1)(x)
    x = ResnetBlock(256//n)(x)
    x = PaddedConv2D(256//n, 1)(x)
    x = ResnetBlock(256//n)(x)

    x = ResnetBlock(512//n)(x)
    x = PaddedConv2D(512//n, 1)(x)
    x = ResnetBlock(512//n)(x)
    x = PaddedConv2D(512//n, 1)(x)
    x = ResnetBlock(512//n)(x)
    x = PaddedConv2D(512//n, 3, padding=1)(x)
    x = ResnetBlock(512//n)(x)
    x = PaddedConv2D(512//n, 3, padding=1)(x)
    x = ResnetBlock(512//n)(x)
    x = PaddedConv2D(512//n, 3, padding=1, strides=2)(x)
    x = ResnetBlock(512//n)(x)
    x = PaddedConv2D(512//n, 3, padding=1)(x)
    x = ResnetBlock(512//n)(x)
    x = PaddedConv2D(512//n, 3, padding=1)(x)
    x = ResnetBlock(512//n)(x)
    x = PaddedConv2D(512//n, 1)(x)
    x = ResnetBlock(512//n)(x)
    x = PaddedConv2D(512//n, 1)(x)
    x = ResnetBlock(512//n)(x)

    x = keras.layers.GroupNormalization(epsilon=1e-5)(x)
    x = keras.layers.Activation("tanh")(x)
    x = PaddedConv2D(512//n, 1)(x)
    z_mean = PaddedConv2D(1, 1)(x)
    z_log_var = PaddedConv2D(1, 3, padding=1)(x)
    z = Sampling()([z_mean, z_log_var])
    return keras.Model(inputs, [z_mean, z_log_var, z], name="encoder")
'''
a=np.random.uniform(-1, 1, (8, 64, 128))
ecoder=Encoder(a.shape[1], a.shape[2])
z_mean, z_log_var, z=ecoder.predict(a, verbose=1, batch_size=8)
'''

'''构建解码器'''
def Decoder(img_height, img_width):
    '''
    img_height, img_width=32,32
    '''
    inputs = keras.layers.Input((img_height, img_width,1), name="image_input")
    x = PaddedConv2D(1, 1)(inputs)
    x = PaddedConv2D(512//n, 1)(x)

    x = ResnetBlock(512//n)(x)
    x = PaddedConv2D(512//n, 1)(x)
    x = ResnetBlock(512//n)(x)
    x = PaddedConv2D(512//n, 1)(x)
    x = ResnetBlock(512//n)(x)
    x = PaddedConv2DT(512//n, 3, padding=1)(x)
    x = ResnetBlock(512//n)(x)
    x = PaddedConv2DT(512//n, 3, padding=1)(x)
    x = ResnetBlock(512//n)(x)
    x = PaddedConv2DT(512//n, 3, padding=1, strides=2)(x)
    x = ResnetBlock(512//n)(x)
    x = PaddedConv2DT(512//n, 3, padding=1)(x)
    x = ResnetBlock(512//n)(x)
    x = PaddedConv2DT(512//n, 3, padding=1)(x)
    x = ResnetBlock(512//n)(x)
    x = PaddedConv2D(512//n, 1)(x)
    x = ResnetBlock(512//n)(x)
    x = PaddedConv2D(512//n, 1)(x)
    x = ResnetBlock(512//n)(x)

    x = ResnetBlock(256//n)(x)
    x = PaddedConv2D(256//n, 1)(x)
    x = ResnetBlock(256//n)(x)
    x = PaddedConv2D(256//n, 1)(x)
    x = ResnetBlock(256//n)(x)
    x = PaddedConv2DT(256//n, 3, padding=1)(x)
    x = ResnetBlock(256//n)(x)
    x = PaddedConv2DT(256//n, 3, padding=1)(x)
    x = ResnetBlock(256//n)(x)
    x = PaddedConv2DT(256//n, 3, padding=1, strides=(1,2))(x)
    x = ResnetBlock(256//n)(x)
    x = PaddedConv2DT(256//n, 3, padding=1)(x)
    x = ResnetBlock(256//n)(x)
    x = PaddedConv2DT(256//n, 3, padding=1)(x)
    x = ResnetBlock(256//n)(x)
    x = PaddedConv2D(256//n, 1)(x)
    x = ResnetBlock(256//n)(x)
    x = PaddedConv2D(256//n, 1)(x)
    x = ResnetBlock(256//n)(x)
    
    x = ResnetBlock(128//n)(x)
    x = PaddedConv2D(128//n, 1)(x)
    x = ResnetBlock(128//n)(x)
    x = PaddedConv2DT(128//n, 3, padding=1)(x)
    x = ResnetBlock(128//n)(x)
    x = PaddedConv2DT(128//n, 3, padding=1)(x)
    x = ResnetBlock(128//n)(x)
    x = PaddedConv2D(128//n, 1)(x)
    x = ResnetBlock(128//n)(x)

    x = keras.layers.GroupNormalization(epsilon=1e-5)(x)
    x = keras.layers.Activation("tanh")(x)
    x = PaddedConv2D(128//n, 1)(x)
    x = PaddedConv2D(1, 1)(x)
    out = keras.layers.Activation("tanh")(x)
    #x = keras.layers.Conv2D(1, 3,activation="tanh")(x)
    #x = keras.layers.Activation("tanh")(x)
    #x = keras.layers.Resizing(160,64, interpolation='bicubic')(x)
    return keras.Model(inputs, out, name="decoder")
'''
b=np.random.uniform(-1, 1, (8, 32, 32))
decoder=Decoder(b.shape[1], b.shape[2])
out=decoder.predict(b, verbose=1, batch_size=4)
'''