from __future__ import division
import os
import time
from glob import glob
import tensorflow as tf
import numpy as np
from six.moves import xrange
import matplotlib.pyplot as plt
from ops import *
from utils import *

class DCGAN(object):
  def __init__(self, sess, input_height=28, input_width=1, input_depth=1, is_crop=True,
         batch_size=64, sample_num = 64, output_height=28, output_width=1, output_depth=1,
         y_dim=2, z_dim=100, gf_dim=64, df_dim=64,
         gfc_dim=1024, dfc_dim=1024, dataset_name='default',
         input_fname_pattern='*.jpg', checkpoint_dir=None, sample_dir=None):
    """

    Args:
      sess: TensorFlow session
      batch_size: The size of batch. Should be specified before training.
      y_dim: (optional) Dimension of dim for y. [10]
      z_dim: (optional) Dimension of dim for Z. [100]
      gf_dim: (optional) Dimension of gen filters in first conv layer. [64]
      df_dim: (optional) Dimension of discrim filters in first conv layer. [64]
      gfc_dim: (optional) Dimension of gen units for for fully connected layer. [1024]
      dfc_dim: (optional) Dimension of discrim units for fully connected layer. [1024]
    """
    self.sess = sess
    self.is_crop = is_crop
    self.is_grayscale = True

    
    self.batch_size = batch_size
    self.sample_num = sample_num

    self.input_height = input_height
    self.input_width = input_width
    self.input_depth = input_depth
    self.output_height = output_height
    self.output_width = output_width
    self.output_depth = output_depth

    self.y_dim = y_dim
    self.z_dim = z_dim

    self.gf_dim = gf_dim
    self.df_dim = df_dim

    self.gfc_dim = gfc_dim
    self.dfc_dim = dfc_dim

    # batch normalization : deals with poor initialization helps gradient flow
    self.d_bn1 = batch_norm(name='d_bn1')
    self.d_bn2 = batch_norm(name='d_bn2')

    self.g_bn0 = batch_norm(name='g_bn0')
    self.g_bn1 = batch_norm(name='g_bn1')
    self.g_bn2 = batch_norm(name='g_bn2')

    self.dataset_name = dataset_name
    self.input_fname_pattern = input_fname_pattern
    self.checkpoint_dir = checkpoint_dir
    self.build_model()

  def build_model(self):
    #labels
    self.y= tf.placeholder(tf.float32, [self.batch_size, self.y_dim], name='y')
    
    image_dims = [self.output_height, self.output_width, self.output_depth] 
    #real samples    
    self.inputs = tf.placeholder(
      tf.float32, [self.batch_size] + image_dims, name='real_images')
    #fake samples
    self.sample_inputs = tf.placeholder(
      tf.float32, [self.sample_num] + image_dims, name='sample_inputs')

    inputs = self.inputs
    sample_inputs = self.sample_inputs

    #z
    self.z = tf.placeholder(
      tf.float32, [None, self.z_dim], name='z')
    self.z_sum = tf.summary.histogram("z", self.z)

    #generator
    self.G = self.generator(self.z, self.y)
    
    #discriminator on the real samples 
    self.D, self.D_logits = \
        self.discriminator(inputs, self.y, reuse=False)

    self.sampler = self.sampler(self.z, self.y)
    
    #discriminator on the samples produced by G
    self.D_, self.D_logits_ = \
        self.discriminator(self.G, self.y, reuse=True)

    self.d_sum = tf.summary.histogram("d", self.D)
    self.d__sum = tf.summary.histogram("d_", self.D_)
    self.G_sum = tf.summary.image("G", self.G)

    self.d_loss_real = tf.reduce_mean(
      tf.nn.sigmoid_cross_entropy_with_logits(
        logits=self.D_logits, targets=tf.ones_like(self.D)))
    self.d_loss_fake = tf.reduce_mean(
      tf.nn.sigmoid_cross_entropy_with_logits(
        logits=self.D_logits_, targets=tf.zeros_like(self.D_)))
    self.g_loss = tf.reduce_mean(
      tf.nn.sigmoid_cross_entropy_with_logits(
        logits=self.D_logits_, targets=tf.ones_like(self.D_)))

    self.d_loss_real_sum = tf.summary.scalar("d_loss_real", self.d_loss_real)
    self.d_loss_fake_sum = tf.summary.scalar("d_loss_fake", self.d_loss_fake)
                          
    self.d_loss = self.d_loss_real + self.d_loss_fake

    self.g_loss_sum = tf.summary.scalar("g_loss", self.g_loss)
    self.d_loss_sum = tf.summary.scalar("d_loss", self.d_loss)

    t_vars = tf.trainable_variables()

    self.d_vars = [var for var in t_vars if 'd_' in var.name]
    self.g_vars = [var for var in t_vars if 'g_' in var.name]

    self.saver = tf.train.Saver()

  def train(self, config):
    """Train DCGAN"""
    #get data
    data_X, data_y = self.load_mnist()
    
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

      
    self.g_sum = tf.summary.merge([self.z_sum, self.d__sum,
      self.G_sum, self.d_loss_fake_sum, self.g_loss_sum])
    self.d_sum = tf.summary.merge(
        [self.z_sum, self.d_sum, self.d_loss_real_sum, self.d_loss_sum])
    self.writer = tf.summary.FileWriter("./logs", self.sess.graph)

    sample_z = np.random.uniform(-1, 1, size=(self.sample_num , self.z_dim))
    
    sample_inputs = data_X[0:self.sample_num]
    sample_labels = data_y[0:self.sample_num]
  
    counter = 1
    start_time = time.time()

    if self.load(self.checkpoint_dir):
      print(" [*] Load SUCCESS")
    else:
      print(" [!] Load failed...")

    for epoch in xrange(config.epoch):
      batch_idxs = min(len(data_X), config.train_size) // config.batch_size

      for idx in xrange(0, batch_idxs):
        batch_images = data_X[idx*config.batch_size:(idx+1)*config.batch_size]
        batch_labels = data_y[idx*config.batch_size:(idx+1)*config.batch_size]
        batch_z = np.random.uniform(-1, 1, [config.batch_size, self.z_dim]) \
              .astype(np.float32)
        
        # Update D network
        _, summary_str = self.sess.run([d_optim, self.d_sum],
          feed_dict={ 
            self.inputs: batch_images,
            self.z: batch_z,
            self.y:batch_labels,
          })
        self.writer.add_summary(summary_str, counter)

        # Update G network
        _, summary_str = self.sess.run([g_optim, self.g_sum],
          feed_dict={
            self.z: batch_z, 
            self.y:batch_labels,
          })
        self.writer.add_summary(summary_str, counter)

        # Run g_optim twice to make sure that d_loss does not go to zero (different from paper)
        _, summary_str = self.sess.run([g_optim, self.g_sum],
          feed_dict={ self.z: batch_z, self.y:batch_labels })
        self.writer.add_summary(summary_str, counter)
          
        errD_fake = self.d_loss_fake.eval({
            self.z: batch_z, 
            self.y:batch_labels
        })
        errD_real = self.d_loss_real.eval({
            self.inputs: batch_images,
            self.y:batch_labels
        })
        errG = self.g_loss.eval({
            self.z: batch_z,
            self.y: batch_labels
        })

        counter += 1
        print("Epoch: [%2d] [%4d/%4d] time: %4.4f, d_loss: %.8f, g_loss: %.8f" \
          % (epoch, idx, batch_idxs,
            time.time() - start_time, errD_fake+errD_real, errG))

        if np.mod(counter, 100) == 1:
          samples, d_loss, g_loss = self.sess.run(
            [self.sampler, self.d_loss, self.g_loss],
            feed_dict={
                self.z: sample_z,
                self.inputs: sample_inputs,
                self.y:sample_labels,
            }
          )
          save_images(samples, [8, 8],
                './{}/train_{:02d}_{:04d}.png'.format(config.sample_dir, epoch, idx))
          print("[Sample] d_loss: %.8f, g_loss: %.8f" % (d_loss, g_loss)) 

        if np.mod(counter, 500) == 2:
          self.save(config.checkpoint_dir, counter)

  def discriminator(self, image, y=None, reuse=False):
    with tf.variable_scope("discriminator") as scope:
      if reuse:
        scope.reuse_variables()

      yb = tf.reshape(y, [self.batch_size, 1, 1, self.y_dim])
      x = conv_cond_concat(image, yb)

      h0 = lrelu(conv2d(x, self.output_depth + self.y_dim, name='d_h0_conv'))
      h0 = conv_cond_concat(h0, yb)

      h1 = lrelu(self.d_bn1(conv2d(h0, self.df_dim + self.y_dim, name='d_h1_conv')))
      h1 = tf.reshape(h1, [self.batch_size, -1])      
      h1 = tf.concat_v2([h1, y], 1)
        
      h2 = lrelu(self.d_bn2(linear(h1, self.dfc_dim, 'd_h2_lin')))
      h2 = tf.concat_v2([h2, y], 1)

      h3 = linear(h2, 1, 'd_h3_lin')
        
      return tf.nn.sigmoid(h3), h3

  def generator(self, z, y=None):
    with tf.variable_scope("generator") as scope:
      s_h, s_w = self.output_height, self.output_width
      s_h2, s_h4 = int(s_h/2), int(s_h/4)
      s_w2, s_w4 = int(s_w), int(s_w)

      # yb = tf.expand_dims(tf.expand_dims(y, 1),2)
      yb = tf.reshape(y, [self.batch_size, 1, 1, self.y_dim])
      z = tf.concat_v2([z, y], 1)

      h0 = tf.nn.relu(
          self.g_bn0(linear(z, self.gfc_dim, 'g_h0_lin')))
      h0 = tf.concat_v2([h0, y], 1)
      
      h1 = tf.nn.relu(self.g_bn1(
          linear(h0, self.gf_dim*2*s_h4*s_w4, 'g_h1_lin')))
      h1 = tf.reshape(h1, [self.batch_size, s_h4, s_w4, self.gf_dim * 2])

      h1 = conv_cond_concat(h1, yb)

      h2 = tf.nn.relu(self.g_bn2(deconv2d(h1,
          [self.batch_size, s_h2, s_w2, self.gf_dim * 2], name='g_h2')))
      h2 = conv_cond_concat(h2, yb)

      return tf.nn.sigmoid(
          deconv2d(h2, [self.batch_size, s_h, s_w, self.output_depth], name='g_h3'))

  def sampler(self, z, y=None):
    with tf.variable_scope("generator") as scope:
      scope.reuse_variables()

      s_h, s_w = self.output_height, self.output_width
      s_h2, s_h4 = int(s_h/2), int(s_h/4)
      s_w2, s_w4 = int(s_w), int(s_w)

      # yb = tf.reshape(y, [-1, 1, 1, self.y_dim])
      yb = tf.reshape(y, [self.batch_size, 1, 1, self.y_dim])
      z = tf.concat_v2([z, y], 1)

      h0 = tf.nn.relu(self.g_bn0(linear(z, self.gfc_dim, 'g_h0_lin')))
      h0 = tf.concat_v2([h0, y], 1)

      h1 = tf.nn.relu(self.g_bn1(
          linear(h0, self.gf_dim*2*s_h4*s_w4, 'g_h1_lin'), train=False))
      h1 = tf.reshape(h1, [self.batch_size, s_h4, s_w4, self.gf_dim * 2])
      h1 = conv_cond_concat(h1, yb)

      h2 = tf.nn.relu(self.g_bn2(
          deconv2d(h1, [self.batch_size, s_h2, s_w2, self.gf_dim * 2], name='g_h2'), train=False))
      h2 = conv_cond_concat(h2, yb)

      return tf.nn.sigmoid(deconv2d(h2, [self.batch_size, s_h, s_w, self.output_depth], name='g_h3'))

  def load_mnist(self):
    
    #we load and slice the images to build 1D samples
    
    #create artificial data
    num_samples = 40000
    num_bins = 28
    firing_rate = 10
    margin = 0
    noise = 0.1
    std_resp = 2
    peaks1 = np.random.randint(int(num_bins/4)-margin,int(num_bins/4)+margin+1,num_samples)
    peaks2 = np.random.randint(num_bins-int(num_bins/4)-margin,num_bins-int(num_bins/4)+margin+1,num_samples)
    t = np.arange(num_bins)
    X =np.zeros((2*num_samples,num_bins,1,1))
    y =np.zeros((2*num_samples,self.y_dim))
    f, (ax1, ax2) = plt.subplots(1, 2, sharey=False)
    

    for ind in range(int(num_samples)):
        r1 = firing_rate*np.exp(-(t-peaks1[ind])**2/std_resp**2) + np.random.normal(0,noise,(1,num_bins))
        r2 = firing_rate*np.exp(-(t-peaks2[ind])**2/std_resp**2) + np.random.normal(0,noise,(1,num_bins))
        r = r1 + r2
        X[ind,:,0,0] = r
        y[ind,:] = [1,0]
        if ind%100==0:
            ax1.plot(r[0])

    peaks1 = np.random.randint(int(num_bins/2)-margin,int(num_bins/2)+margin+1,num_samples)
    for ind in range(int(num_samples+1),int(2*num_samples)):
        r = firing_rate*np.exp(-(t-peaks1[ind-num_samples])**2/std_resp**2) + np.random.normal(0,noise,(1,num_bins))
        X[ind,:,0,0] = r
        y[ind,:] = [0,1]
        if ind%100==0:
            ax2.plot(r[0])
    
    plt.show()
    y_vec = y
    # Shuffle images
    seed = 547
    np.random.seed(seed)
    np.random.shuffle(X)
    np.random.seed(seed)
    np.random.shuffle(y)
 
    
    
  
    return X/X.max(),y_vec

  @property
  def model_dir(self):
    return "{}_{}_{}_{}".format(
        self.dataset_name, self.batch_size,
        self.output_height, self.output_width)
      
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
