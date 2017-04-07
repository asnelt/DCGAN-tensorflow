#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 23 10:47:07 2017

@author: manuel
"""

from __future__ import division
import os
import time
import tensorflow as tf
import numpy as np
from six.moves import xrange
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import ops
import utils
from functools import wraps

def compatibility_decorator(f):
  @wraps(f)
  def wrapper(*args, **kwds):
    name = kwds.pop('name', None)
    return f(targets=kwds['labels'], logits=kwds['logits'], name=name)
  return wrapper
   
# compatibility for TF v<1.0
if int(tf.__version__.split('.')[0]) < 1:
  tf.concat = tf.concat_v2
  tf.nn.sigmoid_cross_entropy_with_logits = compatibility_decorator(tf.nn.sigmoid_cross_entropy_with_logits)

class DCGAN(object):
  def __init__(self, sess, input_height=28, input_depth=1,
               is_crop=True, batch_size=64, sample_num = 64,
               output_height=28, output_depth=1, z_dim=100,
               gf_dim=64, df_dim=64, gfc_dim=1024, dfc_dim=1024,
               kernel_n=20, kernel_d=20, dataset_name='default',
               input_fname_pattern='*.jpg', checkpoint_dir=None,
               sample_dir=None):
    """

    Args:
      sess: TensorFlow session
      batch_size: The size of batch. Should be specified before training.
      z_dim: (optional) Dimension of dim for Z. [100]
      gf_dim: (optional) Dimension of gen filters in first conv layer. [64]
      df_dim: (optional) Dimension of discrim filters in first conv layer. [64]
      gfc_dim: (optional) Dimension of gen units for for fully connected layer. [1024]
      dfc_dim: (optional) Dimension of discrim units for fully connected layer. [1024]
      kernel_n: (optional) number of minibatch discrimination kernels. [20] Corresponds to 'B' in Salimans2016, where B=100.
      kernel_d: (optional) dimensionality of minibatch discrimination kernels. [20] Corresponds to 'C' in Salimans2016, where C=50.
    """
    self.sess = sess
    self.is_crop = is_crop
    self.is_grayscale = True
    
    self.batch_size = batch_size
    self.sample_num = sample_num

    #dimensions' sizes
    self.input_height = input_height
    self.input_depth = input_depth
    self.output_height = output_height
    self.output_depth = output_depth

    self.z_dim = z_dim

    self.gf_dim = gf_dim
    self.df_dim = df_dim

    self.gfc_dim = gfc_dim
    self.dfc_dim = dfc_dim

    # minibatch discrimination parameters
    self.kernel_n = kernel_n
    self.kernel_d = kernel_d

    # batch normalization : deals with poor initialization helps gradient flow
    self.d_bn1 = ops.batch_norm(name='d_bn1')
    self.d_bn2 = ops.batch_norm(name='d_bn2')

    self.g_bn0 = ops.batch_norm(name='g_bn0')
    self.g_bn1 = ops.batch_norm(name='g_bn1')
    self.g_bn2 = ops.batch_norm(name='g_bn2')

    self.dataset_name = dataset_name
    self.input_fname_pattern = input_fname_pattern
    self.checkpoint_dir = checkpoint_dir
    
    self.build_model()

  def build_model(self):
    image_dims = [self.output_height, self.output_depth] 
    #real samples    
    self.inputs = tf.placeholder(
      tf.float32, [self.batch_size] + image_dims, name='real_images')
    inputs = self.inputs
    #fake samples
    self.sample_inputs = tf.placeholder(
      tf.float32, [self.sample_num] + image_dims, name='sample_inputs')

    #z
    self.z = tf.placeholder(
      tf.float32, [None, self.z_dim], name='z')
    #z saved for tensorboard
    self.z_sum = tf.summary.histogram("z", self.z)

    #generator
    self.G = self.generator(self.z)
    #sampler, this is used to get samples from G
    self.sampler = self.sampler(self.z)
    
    #discriminator on the real samples 
    self.D, self.D_logits = \
        self.discriminator(inputs, reuse=False)
    #discriminator on the samples produced by G
    self.D_, self.D_logits_ = \
        self.discriminator(self.G, reuse=True)
        
    #save outputs from discriminator (real and fake) and generator for tensorboard
    self.d_sum = tf.summary.histogram("d", self.D)
    self.d__sum = tf.summary.histogram("d_", self.D_)
    G_shape = self.G.get_shape().as_list()
    aux = [G_shape[2], G_shape[0], G_shape[1]]
    G_sum = tf.reshape(self.G,aux)
    G_sum = tf.expand_dims(G_sum,axis=3)
    self.G_sum = tf.summary.image("G", G_sum)
    
    #loss functions
    #loss D real samples
    self.d_loss_real = tf.reduce_mean(
      tf.nn.sigmoid_cross_entropy_with_logits(
        logits=self.D_logits, labels=tf.ones_like(self.D)))
    #loss D fake samples
    self.d_loss_fake = tf.reduce_mean(
      tf.nn.sigmoid_cross_entropy_with_logits(
        logits=self.D_logits_, labels=tf.zeros_like(self.D_)))
    #loss G
    self.g_loss = tf.reduce_mean(
      tf.nn.sigmoid_cross_entropy_with_logits(
        logits=self.D_logits_, labels=tf.ones_like(self.D_)))

    #save D losses for tensorboard
    self.d_loss_real_sum = tf.summary.scalar("d_loss_real", self.d_loss_real)
    self.d_loss_fake_sum = tf.summary.scalar("d_loss_fake", self.d_loss_fake)                      
    self.d_loss = self.d_loss_real + self.d_loss_fake
    self.d_loss_sum = tf.summary.scalar("d_loss", self.d_loss)
    
    #save G loss for tensorboard
    self.g_loss_sum = tf.summary.scalar("g_loss", self.g_loss)
    
    #get all variables
    t_vars = tf.trainable_variables()
    #keep D and G variables
    self.d_vars = [var for var in t_vars if 'd_' in var.name]
    self.g_vars = [var for var in t_vars if 'g_' in var.name]

    #save training
    self.saver = tf.train.Saver()

  def train(self, data, config):
    """Train DCGAN"""
    #define optimizer
    d_optim = tf.train.AdamOptimizer(config.learning_rate, beta1=config.beta1) \
              .minimize(self.d_loss, var_list=self.d_vars)
    g_optim = tf.train.AdamOptimizer(config.learning_rate, beta1=config.beta1) \
              .minimize(self.g_loss, var_list=self.g_vars)
    
    #initizialize variables              
    try:
      tf.global_variables_initializer().run()
    except:
      tf.initialize_all_variables().run()

    #put all G variables saved for tensorboard together
    self.g_sum = tf.summary.merge([self.z_sum, self.d__sum,
      self.G_sum, self.d_loss_fake_sum, self.g_loss_sum])
    #put all D variables saved for tensorboard together
    self.d_sum = tf.summary.merge(
        [self.z_sum, self.d_sum, self.d_loss_real_sum, self.d_loss_sum])
    
    #save variables in ./logs
    self.writer = tf.summary.FileWriter("./logs_1d", self.sess.graph)

    #samples for visualization (will be used for self.sampler)
    sample_z = np.random.uniform(-1, 1, size=(self.sample_num , self.z_dim))
    sample_inputs = data[0:self.sample_num]
    
    counter = 1
    start_time = time.time()
    if self.load(self.checkpoint_dir):
      print(" [*] Load SUCCESS")
    else:
      print(" [!] Load failed...")

    combined_summary = tf.Summary()
    #start training
    for epoch in xrange(config.epoch):
      batch_idxs = min(len(data), config.train_size) // config.batch_size
      #go through all batches
      for idx in xrange(0, batch_idxs):
        #get batches
        batch_images = data[idx*config.batch_size:(idx+1)*config.batch_size]
        batch_z = np.random.uniform(-1, 1, [config.batch_size, self.z_dim]) \
              .astype(np.float32)
        
        #every 500 batches, we get and save a sample (note that the inputs are always the same)
        if np.mod(counter, 500) == 1 or (epoch==0 and idx<=100 and idx%10==0):
          samples, d_loss, g_loss = self.sess.run(
            [self.sampler, self.d_loss, self.g_loss],
            feed_dict={
                self.z: sample_z,
                self.inputs: sample_inputs
            }
          )
          samples_plot = samples[:,:,0]
          fig,sbplt = plt.subplots(8,8)
          for ind_pl in range(np.shape(samples_plot)[0]):
              sbplt[int(np.floor(ind_pl/8))][ind_pl%8].plot(samples_plot[int(ind_pl),:])
              sbplt[int(np.floor(ind_pl/8))][ind_pl%8].axis('off')
              if ind_pl==0:
                  fig.suptitle("[Sample] d_loss: %.8f, g_loss: %.8f" % (d_loss, g_loss))
          fig.savefig('./{}/train_{:02d}_{:04d}.png'.format(config.sample_dir, epoch, idx),dpi=199, bbox_inches='tight')
          plt.close(fig)
          
          
        if counter % 500 == 1:# and self.dataset_name!='calcium_transients':
          #get autocorrelogram
          utils.get_samples_autocorrelogram(self.sess, self,'train_{:02d}_{:04d}'.format(epoch, idx), config.sample_dir)
          
          print("[Sample] d_loss: %.8f, g_loss: %.8f" % (d_loss, g_loss)) 

        # Update D network
        _, summary_str = self.sess.run([d_optim, self.d_sum],
          feed_dict={ 
            self.inputs: batch_images,
            self.z: batch_z
          })
        
        combined_summary.MergeFromString(summary_str)
        if (counter+1) % 1000 == 0:
            self.writer.add_summary(combined_summary,counter)
            combined_summary = tf.Summary()   
        #self.writer.add_summary(summary_str, counter)

        # Update G network
        _, summary_str = self.sess.run([g_optim, self.g_sum],
          feed_dict={
            self.z: batch_z 
          })
        combined_summary.MergeFromString(summary_str)

        if (counter+1) % 1000 == 0:
            self.writer.add_summary(combined_summary,counter)
            combined_summary = tf.Summary()

        # Run g_optim twice to make sure that d_loss does not go to zero (different from paper)
        _, summary_str = self.sess.run([g_optim, self.g_sum],
          feed_dict={ 
                     self.z: batch_z
                     })
        self.writer.add_summary(summary_str, counter)
         
        #get error values for display
        errD_fake = self.d_loss_fake.eval({
            self.z: batch_z 
        })
        errD_real = self.d_loss_real.eval({
            self.inputs: batch_images
        })
        errG = self.g_loss.eval({
            self.z: batch_z
        })

        counter += 1
        print("Epoch: [%2d] [%4d/%4d] time: %4.4f, d_loss: %.8f, g_loss: %.8f" \
          % (epoch, idx, batch_idxs,
            time.time() - start_time, 
            errD_fake+errD_real, errG))

        #save training every 500 batches
        if np.mod(counter, 500) == 2:
          self.save(config.checkpoint_dir, counter)
          
  def discriminator(self, image, reuse=False):
    print('DISCRIMINATOR------------------------------------------')
    with tf.variable_scope("discriminator") as scope:
      
      #this is to use the same variables for the real and fake update
      if reuse:
        scope.reuse_variables()
      #the input is used as is without any labels
      x = image
      
      #first layer (conv2d + relu, no batch norm.)
      h0 = ops.lrelu(ops.conv1d(x, self.output_depth, name='d_h0_conv'))
      #h0_shape = h0.get_shape().as_list()
      #aux = [h0_shape[2], h0_shape[0], h0_shape[1],h0_shape[3]]
      #h0_sum = tf.reshape(h0,aux)
      
      #second layer (conv2d + batch norm. + relu)
      h1 = ops.lrelu(self.d_bn1(ops.conv1d(h0, self.df_dim, name='d_h1_conv')))
      h1 = tf.reshape(h1, [self.batch_size, -1])   
      
      #third layer (linear + batch norm. + relu)
      h2 = ops.lrelu(self.d_bn2(ops.linear(h1, self.dfc_dim, 'd_h2_lin')))

      ## Minibatch discrimination (from "improved GAN" paper, Salimans2016)
      # reshape features so that each sample is represented by a 1D vector of length A (not really needed in 1D case)
      h2_reshaped = tf.reshape(h2, [self.batch_size, -1], name='d_h3_reshaped');
      # multiply feature vector by a tensor T of size AxBxC, where B
      # is the number of kernels and C is the kernel dimensionality
      # (actually we're multipling by a tensor of size Ax(BxC) and
      # then reshaping the output to NxBxC, where N is the size of the
      # batch)
      M = tf.reshape(ops.linear(h2_reshaped, self.kernel_n * self.kernel_d),
                     (self.batch_size, self.kernel_n, self.kernel_d))

      # compute the negative exponential of the L1 distance between
      # all samples i,j as given by the L1 norm taken separately for
      # each kernel: c_b(x_i,x_j) = exp(-|M_{i,b}-M_{j,b}|) for all
      # i,j,b.
      exp_diff = tf.exp(-tf.reduce_sum(tf.abs(tf.expand_dims(M, 3) - tf.expand_dims(tf.transpose(M, [1, 2, 0]), 0)), 2))

      # prepare binary mask that we need to select, for each neuron
      # sample i, the distances to all other samples, as above we also
      # computed the distances between each sample and itself.
      big = np.zeros((self.batch_size, self.batch_size), dtype='float32')
      big += np.eye(self.batch_size)
      big = tf.expand_dims(big, 1)
      mask = 1. - big

      # compute the actual minibatch features, called o(x_i) by
      # Salimans et al.
      minibatch_features = tf.reduce_sum(exp_diff*mask, 2)
      
      print("original features: ", h2_reshaped.get_shape())
      print("minibatch_features: ", minibatch_features.get_shape())

      # concatenate original features, minibatch features
      x = tf.concat([h2_reshaped , minibatch_features], 1)
      print("x: ", x.get_shape())

      # final projection of extended feature vector      
      h3 = ops.linear(x, 1, 'd_h3_lin')
       
      # compute sigmoid and return
      return tf.nn.sigmoid(h3), h3
    
  def generator(self, z):
    print('GENERATOR------------------------------------------')
    with tf.variable_scope("generator"):
      #sizes
      s_h = self.output_height
      s_h2, s_h4 = int(s_h/2), int(s_h/4)

      #first layer (linear + batch norm. + relu)
      h0 = tf.nn.relu(
          self.g_bn0(ops.linear(z, self.gfc_dim, 'g_h0_lin')))
      
      #first layer (linear + batch norm. + relu)
      h1 = tf.nn.relu(self.g_bn1(
          ops.linear(h0, self.gf_dim*2*s_h4, 'g_h1_lin')))
      h1 = tf.reshape(h1, [self.batch_size, s_h4, self.gf_dim * 2])

      #third layer (deconv 1d + batch norm. + relu)
      h2 = tf.nn.relu(self.g_bn2(ops.nn_resize(h1,
          [self.batch_size, s_h2, 1, self.gf_dim * 2], name='g_h2')))

      #third layer (deconv 1d + signmoid, no batch norm.)
      return tf.nn.sigmoid(
          ops.nn_resize(h2, [self.batch_size, s_h, 1, self.output_depth], name='g_h3'))
  
  #this function is the same as the generator, but it is just retrieve samples 
  #(note the train=false of lines 335 and 340)
  def sampler(self, z):
    with tf.variable_scope("generator") as scope:
      scope.reuse_variables()

      #sizes
      s_h = self.output_height
      s_h2, s_h4 = int(s_h/2), int(s_h/4)

      h0 = tf.nn.relu(self.g_bn0(ops.linear(z, self.gfc_dim, 'g_h0_lin')))

      h1 = tf.nn.relu(self.g_bn1(
          ops.linear(h0, self.gf_dim*2*s_h4, 'g_h1_lin'), train=False))
      h1 = tf.reshape(h1, [self.batch_size, s_h4, self.gf_dim * 2])

      h2 = tf.nn.relu(self.g_bn2(
          ops.nn_resize(h1, [self.batch_size, s_h2, 1, self.gf_dim * 2], name='g_h2'), train=False))

      return tf.nn.sigmoid(ops.nn_resize(h2, [self.batch_size, s_h, 1, self.output_depth], name='g_h3'))

  @property
  def model_dir(self):
    return "{}_{}_{}".format(
        self.dataset_name, self.batch_size,
        self.output_height)
      
  def save(self, checkpoint_dir, step):
    model_name = "DCGAN.model"
    checkpoint_dir = os.path.join(checkpoint_dir, self.model_dir)

    if not os.path.exists(checkpoint_dir):
      os.makedirs(checkpoint_dir)

    self.saver.save(self.sess,
            os.path.join(checkpoint_dir, model_name),
            global_step=step)

  def load(self, checkpoint_dir):
    print(" [*] Reading checkpoints...")
    checkpoint_dir = os.path.join(checkpoint_dir, self.model_dir)

    ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
    if ckpt and ckpt.model_checkpoint_path:
      ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
      self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
      print(" [*] Success to read {}".format(ckpt_name))
      return True
    else:
      print(" [*] Failed to find a checkpoint")
      return False
