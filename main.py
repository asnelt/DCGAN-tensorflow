#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 23 11:27:20 2017

@author: manuel
"""

import os
import numpy as np

from model import DCGAN
from utils import pp, get_samples_autocorrelogram, get_samples, compare_trainings
from dataprovider import DataProvider

import tensorflow as tf

flags = tf.app.flags
flags.DEFINE_integer("epoch", 50, "Epoch to train [50]")
flags.DEFINE_float("learning_rate", 0.0002, "Learning rate for adam [0.0002]")
flags.DEFINE_float("beta1", 0.5, "Momentum term of adam [0.5]")
flags.DEFINE_integer("train_size", np.inf, "The size of train images [np.inf]")
flags.DEFINE_integer("batch_size", 64, "The size of batch images [64]")
flags.DEFINE_integer("input_height", 28, "The height of movies to use (will be center cropped) [28]")
flags.DEFINE_integer("input_depth", 1, "The length of movies to use (will be center cropped) [1]")
flags.DEFINE_integer("output_height", 28, "The height of the output movies to produce [28]")
flags.DEFINE_integer("output_depth", 1, "The length of the output movies to produce [1]")
flags.DEFINE_string("input_fname_pattern", "*.jpg", "Glob pattern of filename of input images [*.jpg]")
flags.DEFINE_string("checkpoint_dir", "checkpoint", "Directory name to save the checkpoints [checkpoint]")
flags.DEFINE_string("sample_dir", "samples", "Directory name to save the image samples [samples]")
flags.DEFINE_boolean("is_train", False, "True for training, False for testing [False]")
flags.DEFINE_boolean("is_crop", False, "True for training, False for testing [False]")
flags.DEFINE_boolean("visualize", False, "True for visualizing, False for nothing [False]")
flags.DEFINE_integer("training_step", 250, "number of batches between weigths and performance saving")
flags.DEFINE_string("training_stage", '', "stage of the training used for the GAN")
#parameter set specifiying data
flags.DEFINE_string("dataset", "gaussian_fr", "The name of dataset. It can have a gaussian or uniform shape")
flags.DEFINE_integer("num_classes", 1, "Number of sample classes [3]")
flags.DEFINE_string("classes_proportion", 'equal', "this will control the proportion of each class. It can be 'equal' or '7030'")
flags.DEFINE_integer("num_samples", 50000, "Number of samples to generate [50000]")
flags.DEFINE_integer("num_bins", 28, "Number of spike train bins bins [28]")
flags.DEFINE_string("iteration", "0", "in case several instances are run with the same parameters")
flags.DEFINE_integer("ref_period", -1, "minimum number of ms between spikes (if < 0, no refractory period is imposed)")
flags.DEFINE_boolean("visualize_data", True, "True for visualizing data [True]")

FLAGS = flags.FLAGS

def main(_):
  pp.pprint(flags.FLAGS.__flags)
  FLAGS.checkpoint_dir = FLAGS.checkpoint_dir + '_dataset_' + FLAGS.dataset + '_num_classes_' + str(FLAGS.num_classes) + '_propClasses_' + FLAGS.classes_proportion + \
  '_num_samples_' + str(FLAGS.num_samples) + '_num_bins_' + str(FLAGS.num_bins) + '_ref_period_' + str(FLAGS.ref_period) + '_iteration_' + FLAGS.iteration
  
  FLAGS.sample_dir = FLAGS.sample_dir + '_dataset_' + FLAGS.dataset + '_num_classes_' + str(FLAGS.num_classes) + '_propClasses_' + FLAGS.classes_proportion + \
  '_num_samples_' + str(FLAGS.num_samples) + '_num_bins_' + str(FLAGS.num_bins) + '_ref_period_' + str(FLAGS.ref_period) + '_iteration_' + FLAGS.iteration
  print(FLAGS.checkpoint_dir)
  if not os.path.exists(FLAGS.checkpoint_dir):
    os.makedirs(FLAGS.checkpoint_dir)
  if not os.path.exists(FLAGS.sample_dir):
    os.makedirs(FLAGS.sample_dir)

  #gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.333)
  run_config = tf.ConfigProto()
  run_config.gpu_options.allow_growth=True

  with tf.Session(config=run_config) as sess:
    dcgan = DCGAN(
        sess,
        input_height=FLAGS.input_height,
        input_depth=FLAGS.input_depth,
        output_height=FLAGS.output_height,
        output_depth=FLAGS.output_depth,
        batch_size=FLAGS.batch_size,
        dataset_name=FLAGS.dataset,
        input_fname_pattern=FLAGS.input_fname_pattern,
        is_crop=FLAGS.is_crop,
        checkpoint_dir=FLAGS.checkpoint_dir,
        sample_dir=FLAGS.sample_dir)

    if FLAGS.is_train:
      #import or generate data
      data_provider = DataProvider(FLAGS)
      if FLAGS.visualize_data:
        data_provider.visualize()
      dcgan.train(data_provider.data, FLAGS)
    else:
      if not dcgan.load(FLAGS.checkpoint_dir,FLAGS.training_stage):
        raise Exception("[!] Train a model first, then run test mode")      

    # to_json("./web/js/layers.js", [dcgan.h0_w, dcgan.h0_b, dcgan.g_bn0],
    #                 [dcgan.h1_w, dcgan.h1_b, dcgan.g_bn1],
    #                 [dcgan.h2_w, dcgan.h2_b, dcgan.g_bn2],
    #                 [dcgan.h3_w, dcgan.h3_b, dcgan.g_bn3],
    #                 [dcgan.h4_w, dcgan.h4_b, None])

    # Below is the code for visualization
    get_samples_autocorrelogram(sess, dcgan,'fake',FLAGS,0,0)
    get_samples(sess, dcgan,FLAGS.sample_dir)
    compare_trainings(FLAGS.sample_dir,'training errors')

if __name__ == '__main__':
  tf.app.run()
