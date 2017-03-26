#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 23 10:47:58 2017

@author: manuel
"""

import math
import numpy as np 
import tensorflow as tf

from tensorflow.python.framework import ops

from utils import *

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
  return tf.concat_v2([x, y*tf.ones([x_shapes[0], x_shapes[1], y_shapes[2]])], 2)

def conv1d(input_, output_dim, 
       k_h=5, d_h=2, stddev=0.02,
       name="conv2d"):
  with tf.variable_scope(name):
    w = tf.get_variable('w', [k_h, input_.get_shape()[-1], output_dim],
              initializer=tf.truncated_normal_initializer(stddev=stddev))
    print('ooooooooooooooooooooooooooooooooooooo')
    print('kernel size')
    print(w.get_shape())
    print('input size')
    print(input_.get_shape())
    conv = tf.nn.conv1d(input_, w, stride=d_h, padding='SAME')
    print('conv size')
    print(conv.get_shape())
    print('ooooooooooooooooooooooooooooooooooooo')
    biases = tf.get_variable('biases', [output_dim], initializer=tf.constant_initializer(0.0))
    conv = tf.reshape(tf.nn.bias_add(conv, biases), conv.get_shape())

    return conv

def deconv1d(input_, output_shape,
       k_h=5, d_h=2, stddev=0.02,
       name="deconv2d"):
  with tf.variable_scope(name):
    # filter : [height, width, output_channels, in_channels]
    w = tf.get_variable('w', [k_h, 1, output_shape[-1], input_.get_shape()[-1]],
              initializer=tf.random_normal_initializer(stddev=stddev))
    print('ooooooooooooooooooooooooooooooooooooo')
    print('kernel size')
    print(w.get_shape())
    aux_input = tf.expand_dims(input_,axis=2)
    print('input size')
    print(input_.get_shape())
    print('aux input size')
    print(aux_input.get_shape())
    deconv = tf.nn.conv2d_transpose(aux_input, w, output_shape=output_shape,strides=[1, d_h, 1, 1])
    print('deconv size')
    print(deconv.get_shape())
    print('ooooooooooooooooooooooooooooooooooooo')

    biases = tf.get_variable('biases', [output_shape[-1]], initializer=tf.constant_initializer(0.0))
    deconv = tf.reshape(tf.nn.bias_add(deconv, biases), deconv.get_shape())
   
    return deconv[:,:,0,:]
 
def nn_resize(input_, output_shape,
       k_h=5, d_h=1, stddev=0.02,
       name="deconv2d"):
  with tf.variable_scope(name):
    # filter : [height, width, output_channels, in_channels]
    w = tf.get_variable('w', [k_h, input_.get_shape()[-1], output_shape[-1]],
              initializer=tf.truncated_normal_initializer(stddev=stddev))   
    print('ooooooooooooooooooooooooooooooooooooo')
    print('kernel size')
    print(w.get_shape())
    aux_input = tf.expand_dims(input_,axis=2)
    print('input size')
    print(input_.get_shape())
    print('aux input size')
    print(aux_input.get_shape())
    resized_image = tf.image.resize_images(aux_input, output_shape[1:3], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    resized_image = resized_image[:,:,0,:]
    print('resized image shape')
    print(resized_image.get_shape())
    deconv = tf.nn.conv1d(resized_image, w, stride=d_h, padding='SAME')
    print('deconv size')
    print(deconv.get_shape())
    print('ooooooooooooooooooooooooooooooooooooo')
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
