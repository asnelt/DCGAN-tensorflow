#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 19 18:33:43 2017

@author: manuel
"""
import os
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
main_folder = '/home/manuel/DCGAN-tensorflow/'
os.chdir(main_folder)
import utils


    
pp = PdfPages(main_folder+'/training_error_summary.pdf')
num_bins = 28
# UNIFORM FIRING RATES
# no refractory period    
dataset = 'uniform_fr'
num_classes = 1
classes_proportion = 'equal' 
ref_period = -1
num_samples = 16384

folder = 'samples_dataset_' + dataset + '_num_classes_' + str(num_classes) + '_propClasses_' + classes_proportion + \
  '_num_samples_' + str(num_samples) + '_num_bins_' + str(num_bins) + '_ref_period_' + str(ref_period) + '_iteration_0'
title =   dataset + ' numClasses:' + str(num_classes) + ' propClasses:' + classes_proportion + ' refPeriod:' + str(ref_period) 
utils.compare_trainings(main_folder+folder,title)
fig = plt.figure(figsize=(8,8),dpi=250)
img = mpimg.imread(main_folder+folder+'/training_error.png')
imgplot = plt.imshow(img)
pp.savefig(fig)
plt.close()

# refractory period = 2
dataset = 'uniform_fr'
num_classes = 1
classes_proportion = 'equal' 
ref_period = 2
num_samples = 16384
folder = 'samples_dataset_' + dataset + '_num_classes_' + str(num_classes) + '_propClasses_' + classes_proportion + \
  '_num_samples_' + str(num_samples) + '_num_bins_' + str(num_bins) + '_ref_period_' + str(ref_period) + '_iteration_0'
  
title =   dataset + ' numClasses:' + str(num_classes) + ' propClasses:' + classes_proportion + ' refPeriod:' + str(ref_period) 
utils.compare_trainings(main_folder+folder,title)
fig = plt.figure(figsize=(8,8),dpi=250)
img = mpimg.imread(main_folder+folder+'/training_error.png')
imgplot = plt.imshow(img)
pp.savefig(fig)
plt.close()


# GAUSSIAN FIRING RATES
# no refractory period
dataset = 'gaussian_fr'
num_classes = 1
classes_proportion = 'equal' 
ref_period = -1
num_samples = 16384
folder = 'samples_dataset_' + dataset + '_num_classes_' + str(num_classes) + '_propClasses_' + classes_proportion + \
  '_num_samples_' + str(num_samples) + '_num_bins_' + str(num_bins) + '_ref_period_' + str(ref_period) + '_iteration_0'
  
title =   dataset + ' numClasses:' + str(num_classes) + ' propClasses:' + classes_proportion + ' refPeriod:' + str(ref_period) 
utils.compare_trainings(main_folder+folder,title)
fig = plt.figure(figsize=(8,8),dpi=250)
img = mpimg.imread(main_folder+folder+'/training_error.png')
imgplot = plt.imshow(img)
pp.savefig(fig)
plt.close()

# refractory period = 2
dataset = 'gaussian_fr'
num_classes = 1
classes_proportion = 'equal' 
ref_period = 2
num_samples = 16384
folder = 'samples_dataset_' + dataset + '_num_classes_' + str(num_classes) + '_propClasses_' + classes_proportion + \
  '_num_samples_' + str(num_samples) + '_num_bins_' + str(num_bins) + '_ref_period_' + str(ref_period) + '_iteration_0'
  
title =   dataset + ' numClasses:' + str(num_classes) + ' propClasses:' + classes_proportion + ' refPeriod:' + str(ref_period) 
utils.compare_trainings(main_folder+folder,title)
fig = plt.figure(figsize=(8,8),dpi=250)
img = mpimg.imread(main_folder+folder+'/training_error.png')
imgplot = plt.imshow(img)
pp.savefig(fig)
plt.close()


# 2 classes / refractory period = -1
dataset = 'gaussian_fr'
num_classes = 2
classes_proportion = 'equal' 
ref_period = -1
num_samples = 16384
folder = 'samples_dataset_' + dataset + '_num_classes_' + str(num_classes) + '_propClasses_' + classes_proportion + \
  '_num_samples_' + str(num_samples) + '_num_bins_' + str(num_bins) + '_ref_period_' + str(ref_period) + '_iteration_0'
  
title =   dataset + ' numClasses:' + str(num_classes) + ' propClasses:' + classes_proportion + ' refPeriod:' + str(ref_period) 
utils.compare_trainings(main_folder+folder,title)
fig = plt.figure(figsize=(8,8),dpi=250)
img = mpimg.imread(main_folder+folder+'/training_error.png')
imgplot = plt.imshow(img)
pp.savefig(fig)
plt.close()


# 2 classes / refractory period = 2
dataset = 'gaussian_fr'
num_classes = 2
classes_proportion = 'equal' 
ref_period = 2
num_samples = 16384
folder = 'samples_dataset_' + dataset + '_num_classes_' + str(num_classes) + '_propClasses_' + classes_proportion + \
  '_num_samples_' + str(num_samples) + '_num_bins_' + str(num_bins) + '_ref_period_' + str(ref_period) + '_iteration_0'
  
title =   dataset + ' numClasses:' + str(num_classes) + ' propClasses:' + classes_proportion + ' refPeriod:' + str(ref_period) 
utils.compare_trainings(main_folder+folder,title)
fig = plt.figure(figsize=(8,8),dpi=250)
img = mpimg.imread(main_folder+folder+'/training_error.png')
imgplot = plt.imshow(img)
pp.savefig(fig)
plt.close()


# 2 classes / refractory period = -1 / classes proportion 70-30
dataset = 'gaussian_fr'
num_classes = 2
classes_proportion = '7030' 
ref_period = -1
num_samples = 16384
folder = 'samples_dataset_' + dataset + '_num_classes_' + str(num_classes) + '_propClasses_' + classes_proportion + \
  '_num_samples_' + str(num_samples) + '_num_bins_' + str(num_bins) + '_ref_period_' + str(ref_period) + '_iteration_0'
  
title =   dataset + ' numClasses:' + str(num_classes) + ' propClasses:' + classes_proportion + ' refPeriod:' + str(ref_period) 
utils.compare_trainings(main_folder+folder,title)
fig = plt.figure(figsize=(8,8),dpi=250)
img = mpimg.imread(main_folder+folder+'/training_error.png')
imgplot = plt.imshow(img)
pp.savefig(fig)
plt.close()

# 2 classes / refractory period = 2 / classes proportion 70-30
num_classes = 2
classes_proportion = '7030' 
ref_period = 2
num_samples = 16384
folder = 'samples_dataset_' + dataset + '_num_classes_' + str(num_classes) + '_propClasses_' + classes_proportion + \
  '_num_samples_' + str(num_samples) + '_num_bins_' + str(num_bins) + '_ref_period_' + str(ref_period) + '_iteration_0'
  
title =   dataset + ' numClasses:' + str(num_classes) + ' propClasses:' + classes_proportion + ' refPeriod:' + str(ref_period) 
utils.compare_trainings(main_folder+folder,title)
fig = plt.figure(figsize=(8,8),dpi=250)
img = mpimg.imread(main_folder+folder+'/training_error.png')
imgplot = plt.imshow(img)
pp.savefig(fig)
plt.close()
pp.close()
    
