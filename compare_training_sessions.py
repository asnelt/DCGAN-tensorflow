#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 19 18:33:43 2017

@author: manuel
"""
import os
os.chdir('/home/manuel/DCGAN-tensorflow')
import utils


    
 
num_bins = 28
# UNIFORM FIRING RATES
# no refractory period    
dataset = 'uniform_fr'
num_classes = 1
classes_proportion = 'equal' 
ref_period = -1
epoch = 200  
num_samples = 50000

folder = 'samples_dataset_' + dataset + '_num_classes_' + str(num_classes) + '_propClasses_' + classes_proportion + \
  '_num_samples_' + str(num_samples) + '_num_bins_' + str(num_bins) + '_ref_period_' + str(ref_period) + '_iteration_0'
title =   dataset + ' numClasses:' + str(num_classes) + ' propClasses:' + classes_proportion + ' refPeriod:' + str(ref_period) 
utils.compare_trainings('/home/manuel/DCGAN-tensorflow/'+folder,title)


# refractory period = 2
dataset = 'uniform_fr'
num_classes = 1
classes_proportion = 'equal' 
ref_period = 2
epoch = 200  
num_samples = 50000
folder = 'samples_dataset_' + dataset + '_num_classes_' + str(num_classes) + '_propClasses_' + classes_proportion + \
  '_num_samples_' + str(num_samples) + '_num_bins_' + str(num_bins) + '_ref_period_' + str(ref_period) + '_iteration_0'
  
title =   dataset + ' numClasses:' + str(num_classes) + ' propClasses:' + classes_proportion + ' refPeriod:' + str(ref_period) 
utils.compare_trainings('/home/manuel/DCGAN-tensorflow/'+folder,title)



# GAUSSIAN FIRING RATES
# no refractory period
dataset = 'gaussian_fr'
num_classes = 1
classes_proportion = 'equal' 
ref_period = -1
epoch = 200  
num_samples = 50000
folder = 'samples_dataset_' + dataset + '_num_classes_' + str(num_classes) + '_propClasses_' + classes_proportion + \
  '_num_samples_' + str(num_samples) + '_num_bins_' + str(num_bins) + '_ref_period_' + str(ref_period) + '_iteration_0'
  
title =   dataset + ' numClasses:' + str(num_classes) + ' propClasses:' + classes_proportion + ' refPeriod:' + str(ref_period) 
utils.compare_trainings('/home/manuel/DCGAN-tensorflow/'+folder,title)


# refractory period = 2
dataset = 'gaussian_fr'
num_classes = 1
classes_proportion = 'equal' 
ref_period = 2
epoch = 200  
num_samples = 50000
folder = 'samples_dataset_' + dataset + '_num_classes_' + str(num_classes) + '_propClasses_' + classes_proportion + \
  '_num_samples_' + str(num_samples) + '_num_bins_' + str(num_bins) + '_ref_period_' + str(ref_period) + '_iteration_0'
  
title =   dataset + ' numClasses:' + str(num_classes) + ' propClasses:' + classes_proportion + ' refPeriod:' + str(ref_period) 
utils.compare_trainings('/home/manuel/DCGAN-tensorflow/'+folder,title)



# 2 classes / refractory period = -1
dataset = 'gaussian_fr'
num_classes = 2
classes_proportion = 'equal' 
ref_period = -1
epoch = 200  
num_samples = 50000
folder = 'samples_dataset_' + dataset + '_num_classes_' + str(num_classes) + '_propClasses_' + classes_proportion + \
  '_num_samples_' + str(num_samples) + '_num_bins_' + str(num_bins) + '_ref_period_' + str(ref_period) + '_iteration_0'
  
title =   dataset + ' numClasses:' + str(num_classes) + ' propClasses:' + classes_proportion + ' refPeriod:' + str(ref_period) 
utils.compare_trainings('/home/manuel/DCGAN-tensorflow/'+folder,title)



# 2 classes / refractory period = 2
dataset = 'gaussian_fr'
num_classes = 2
classes_proportion = 'equal' 
ref_period = 2
epoch = 200  
num_samples = 50000
folder = 'samples_dataset_' + dataset + '_num_classes_' + str(num_classes) + '_propClasses_' + classes_proportion + \
  '_num_samples_' + str(num_samples) + '_num_bins_' + str(num_bins) + '_ref_period_' + str(ref_period) + '_iteration_0'
  
title =   dataset + ' numClasses:' + str(num_classes) + ' propClasses:' + classes_proportion + ' refPeriod:' + str(ref_period) 
utils.compare_trainings('/home/manuel/DCGAN-tensorflow/'+folder,title)



# 2 classes / refractory period = -1 / classes proportion 70-30
dataset = 'gaussian_fr'
num_classes = 2
classes_proportion = '7030' 
ref_period = -1
epoch = 200  
num_samples = 50000
folder = 'samples_dataset_' + dataset + '_num_classes_' + str(num_classes) + '_propClasses_' + classes_proportion + \
  '_num_samples_' + str(num_samples) + '_num_bins_' + str(num_bins) + '_ref_period_' + str(ref_period) + '_iteration_0'
  
title =   dataset + ' numClasses:' + str(num_classes) + ' propClasses:' + classes_proportion + ' refPeriod:' + str(ref_period) 
utils.compare_trainings('/home/manuel/DCGAN-tensorflow/'+folder,title)


# 2 classes / refractory period = 2 / classes proportion 70-30
num_classes = 2
classes_proportion = '7030' 
ref_period = 2
epoch = 200  
num_samples = 50000
folder = 'samples_dataset_' + dataset + '_num_classes_' + str(num_classes) + '_propClasses_' + classes_proportion + \
  '_num_samples_' + str(num_samples) + '_num_bins_' + str(num_bins) + '_ref_period_' + str(ref_period) + '_iteration_0'
  
title =   dataset + ' numClasses:' + str(num_classes) + ' propClasses:' + classes_proportion + ' refPeriod:' + str(ref_period) 
utils.compare_trainings('/home/manuel/DCGAN-tensorflow/'+folder,title)


    
