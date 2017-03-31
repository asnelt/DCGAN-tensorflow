#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 23 10:47:58 2017

@author: manuel
"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
class batch_norm(object):
  def __init__(self, epsilon=1e-5, momentum = 0.9, name="batch_norm"):
    with tf.variable_scope(name):
      self.epsilon  = epsilon
      self.momentum = momentum
      self.name = name

  def __call__(self, x, train=True):
    return tf.contrib.layers.batch_norm(x,
                      decay=self.momentum, 
                      updates_collections=None,
                      epsilon=self.epsilon,
                      scale=True,
                      is_training=train,
                      scope=self.name)

def conv_cond_concat(x, y):
  """Concatenate conditioning vector on feature map axis."""
  x_shapes = x.get_shape()
  y_shapes = y.get_shape()
  return tf.concat([x, y*tf.ones([x_shapes[0], x_shapes[1], y_shapes[2]])], 2)

def conv1d(input_, output_dim, 
       k_h=5, d_h=2, stddev=0.02,
       name="conv2d"):
  with tf.variable_scope(name):
    w = tf.get_variable('w', [k_h, input_.get_shape()[-1], output_dim],
              initializer=tf.truncated_normal_initializer(stddev=stddev))
    conv = tf.nn.conv1d(input_, w, stride=d_h, padding='SAME')
    biases = tf.get_variable('biases', [output_dim], initializer=tf.constant_initializer(0.0))
    conv = tf.reshape(tf.nn.bias_add(conv, biases), conv.get_shape())

    return conv

 
def nn_resize(input_, output_shape,
       k_h=5, d_h=1, stddev=0.02,
       name="deconv2d"):
  with tf.variable_scope(name):
    # filter : [height, width, output_channels, in_channels]
    w = tf.get_variable('w', [k_h, input_.get_shape()[-1], output_shape[-1]],
              initializer=tf.truncated_normal_initializer(stddev=stddev))   
    #we first resize the input and then apply a standard convolution
    aux_input = tf.expand_dims(input_,axis=2)#to use the function resize_images we add an extra dimension to the input 
    resized_image = tf.image.resize_images(aux_input, output_shape[1:3], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    resized_image = resized_image[:,:,0,:]#get rid of the extra dimension
    deconv = tf.nn.conv1d(resized_image, w, stride=d_h, padding='SAME')
    biases = tf.get_variable('biases', [output_shape[-1]], initializer=tf.constant_initializer(0.0))
    deconv = tf.reshape(tf.nn.bias_add(deconv, biases), deconv.get_shape())
   
    return deconv
    
def lrelu(x, leak=0.2, name="lrelu"):
  return tf.maximum(x, leak*x)

def linear(input_, output_size, scope=None, stddev=0.02, bias_start=0.0, with_w=False):
  shape = input_.get_shape().as_list()

  with tf.variable_scope(scope or "Linear"):
    matrix = tf.get_variable("Matrix", [shape[1], output_size], tf.float32,
                 tf.random_normal_initializer(stddev=stddev))
    bias = tf.get_variable("bias", [output_size],
      initializer=tf.constant_initializer(bias_start))
    if with_w:
      return tf.matmul(input_, matrix) + bias, matrix, bias
    else:
      return tf.matmul(input_, matrix) + bias

def refractory_period(refrPer,r,name):
    print('imposing refractory period of ' + str(refrPer))
    r = r[:,:,0]
    margin = np.zeros((r.shape[0],refrPer))
    r = np.hstack((margin,np.hstack((r,margin))))
    r_flat = r.flatten()
    spiketimes = np.nonzero(r_flat>0)
    spiketimes = np.sort(spiketimes)
    isis = np.diff(spiketimes)
    too_close = np.nonzero(isis<=refrPer)
    while len(too_close[0])>0:
        spiketimes = np.delete(spiketimes,too_close[0][0]+1)
        isis = np.diff(spiketimes)
        too_close = np.nonzero(isis<=refrPer)
        
    r_flat = np.zeros(r_flat.shape)
    r_flat[spiketimes] = 1
    r = np.reshape(r_flat,r.shape)
    r = r[:,refrPer:-refrPer]
    spk_autocorrelegram(r,name)
    r = np.expand_dims(r,2)
    return r
  
def spk_autocorrelegram(r,name):
    print('plot autocorrelogram')
    lag = 10
    margin = np.zeros((r.shape[0],lag))
    r = np.hstack((margin,np.hstack((r,margin))))
    r_flat = r.flatten()
    spiketimes = np.nonzero(r_flat>0)
    ac = np.zeros(2*lag+1)
    for ind_spk in range(len(spiketimes[0])):
        spike = spiketimes[0][ind_spk]
        ac = ac + r_flat[spike-lag:spike+lag+1]
        
    f = plt.figure()
    index = np.linspace(-lag,lag,2*lag+1)
    plt.plot(index, ac)
    f.savefig('samples/refractory_period' + name + '.png', bbox_inches='tight')
    plt.show()
    plt.close(f)
    
