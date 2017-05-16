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
import matplotlib.pyplot as plt
    
from dataprovider import DataProvider

import tensorflow as tf

flags = tf.app.flags



#parameter set specifiying data
flags.DEFINE_string("dataset", "gaussian_fr", "The name of dataset. It can have a gaussian or uniform shape")
flags.DEFINE_integer("num_classes", 1, "Number of sample classes [3]")
flags.DEFINE_string("classes_proportion", 'equal', "this will control the proportion of each class. It can be 'equal' or '7030'")
flags.DEFINE_integer("num_samples", 500000, "Number of samples to generate [50000]")
flags.DEFINE_integer("num_bins", 28, "Number of spike train bins bins [28]")
flags.DEFINE_string("iteration", "0", "in case several instances are run with the same parameters")
flags.DEFINE_integer("ref_period", -1, "minimum number of ms between spikes (if < 0, no refractory period is imposed)")
flags.DEFINE_string("sample_dir", "samples", "Directory name to save the image samples [samples]")
flags.DEFINE_boolean("visualize_data", True, "True for visualizing data [True]")
flags.DEFINE_boolean("firing_rate", 0.5, "True for visualizing data [True]")
flags.DEFINE_integer("training_step", 250, "number of batches between weigths and performance saving")
#flags.DEFINE_boolean("neuron", 18, "Neuron for which to model the response")
FLAGS = flags.FLAGS
if not os.path.exists(FLAGS.sample_dir):
    os.makedirs(FLAGS.sample_dir)
    
num_rep = 1
data_provider = DataProvider(FLAGS)

data = data_provider.data[:,:,0]

num_samples = FLAGS.num_samples
diff_samples = np.vstack({tuple(row) for row in data})
num_diff_samples = np.shape(diff_samples)[0]


probs_rep = utils.probability_data(diff_samples,FLAGS.num_classes,num_diff_samples,FLAGS.num_bins, FLAGS.firing_rate,FLAGS.dataset,FLAGS.classes_proportion)

 
numerical_prob = np.zeros((num_diff_samples,))
for ind_s in range(num_diff_samples):
    if ind_s%1000==0:
        print(ind_s)
    sample = diff_samples[ind_s,:]
    sample_mat = np.tile(sample,(num_samples,1))
    compare_mat = np.sum(np.abs(data-sample_mat),axis=1)
    numerical_prob[ind_s] = np.count_nonzero(compare_mat==0)/num_samples   
        
print('-----------------')
print('analytical probabilities ' + str(np.mean(probs_rep)))
print('numerical  probabilities ' + str(np.mean(numerical_prob)))
print('ratio ' + str(np.mean(numerical_prob)/np.mean(probs_rep)))
print('-----------------')


samples_probs = {'numerical_prob':numerical_prob,'diff_samples':diff_samples}
np.savez(FLAGS.sample_dir + '/numerical_probs_num_samples_' + str(num_samples) + '.npz',**samples_probs)


f,sbplt = plt.subplots(1,1,figsize=(8, 8),dpi=250)  
sbplt.loglog(numerical_prob,probs_rep,'.b',basex=10)
sbplt.loglog(np.linspace(0,1,10000000),np.linspace(0,1,10000000),basex=10)
sbplt.set_xlabel('numerical probabilities')
sbplt.set_ylabel('theoretical probabilities')
plt.show()    
f.savefig(FLAGS.sample_dir + '/probs_num_samples_' + str(num_samples) + '.png',dpi=300, bbox_inches='tight')

    
    
 
