#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 19 18:33:43 2017

@author: manuel
"""
import os
main_folder = '/home/manuel/DCGAN-tensorflow/'
os.chdir(main_folder)
import utils
import numpy as np
#import matplotlib.pyplot as plt
    
from dataprovider import DataProvider

import tensorflow as tf

flags = tf.app.flags



#parameter set specifiying data
flags.DEFINE_string("dataset", "gaussian_fr", "The name of dataset. It can have a gaussian or uniform shape")
flags.DEFINE_integer("num_classes", 2, "Number of sample classes [3]")
flags.DEFINE_string("classes_proportion", '7030', "this will control the proportion of each class. It can be 'equal' or '7030'")
flags.DEFINE_integer("num_samples", 100000, "Number of samples to generate [50000]")
flags.DEFINE_integer("num_bins", 28, "Number of spike train bins bins [28]")
flags.DEFINE_string("iteration", "0", "in case several instances are run with the same parameters")
flags.DEFINE_integer("ref_period", -1, "minimum number of ms between spikes (if < 0, no refractory period is imposed)")
flags.DEFINE_string("sample_dir", "samples", "Directory name to save the image samples [samples]")
flags.DEFINE_boolean("visualize_data", False, "True for visualizing data [True]")
FLAGS = flags.FLAGS
if not os.path.exists(FLAGS.sample_dir):
    os.makedirs(FLAGS.sample_dir)
    
num_rep = 1
num_samples_per_rep = 500
data_provider = DataProvider(FLAGS)
data = data_provider.data[:,:,0]

#print(np.shape(np.vstack({tuple(row) for row in data})))

num_samples = np.shape(data)[0]

probs_rep = np.zeros((num_rep,))
numerical_probs_rep = np.zeros((num_rep,))
for ind_rep in range(num_rep):
    #print(ind_rep)
    samples = data[num_samples_per_rep*ind_rep:num_samples_per_rep*(ind_rep+1),:]
    probs_rep[ind_rep] = utils.probability_data(FLAGS,samples)
    
    numerical_prob = np.zeros((num_samples_per_rep,))
    for ind_s in range(num_samples_per_rep):
        sample = samples[ind_s,:]
        sample_mat = np.tile(sample,(num_samples,1))
        compare_mat = np.sum(np.abs(data-sample_mat),axis=1)
        numerical_prob[ind_s] = np.count_nonzero(compare_mat==0)/num_samples   
        
    numerical_probs_rep[ind_rep] = np.mean(numerical_prob)
    
    #    f = plt.figure()
    #    plt.hist(numerical_prob)
    #    plt.show()
    #    plt.close(f)
    print('-----------------')
    print('analytical probabilities ' + str(probs_rep[ind_rep]))
    print('numerical  probabilities ' + str(numerical_probs_rep[ind_rep]))
    print('ratio ' + str(numerical_probs_rep[ind_rep]/probs_rep[ind_rep]))
    print('-----------------')
