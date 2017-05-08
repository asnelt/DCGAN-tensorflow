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


def get_figures(main_folder,dataset,num_classes,classes_proportion,num_samples,num_bins,ref_period,pp):
    folder = 'samples_dataset_' + dataset + '_num_classes_' + str(num_classes) + '_propClasses_' + classes_proportion + \
  '_num_samples_' + str(num_samples) + '_num_bins_' + str(num_bins) + '_ref_period_' + str(ref_period) + '_iteration_0'

    title =   dataset + ' numClasses:' + str(num_classes) + ' propClasses:' + classes_proportion + ' refPeriod:' + str(ref_period) 
    best_fit_ac,best_fit_prob = utils.compare_trainings(main_folder+folder,title)
    fig = plt.figure(figsize=(8,8),dpi=250)
    img = mpimg.imread(main_folder+folder+'/training_error.png')
    plt.imshow(img)
    pp.savefig(fig)
    plt.close()
    
    #plot best ac fit
    fig = utils.plot_best_fit(best_fit_ac,folder+'/best_of_all_ac_fit')
    pp.savefig(fig)
    plt.close()
    #want to get samples from the best ac fit
    iteration = best_fit_ac['folder'][best_fit_ac['folder'].find('iteration')+10:]
    training_stage = int(int(best_fit_ac['epoch'])*num_samples/64+int(best_fit_ac['step'])+1)
    print('python3.5 main.py --dataset ' + dataset + ' --num_classes=' + str(num_classes) + ' --classes_proportion ' + classes_proportion + ' --ref_period=' + str(ref_period)\
              + ' --epoch=50 --num_samples=' + str(num_samples) + ' --iteration ' + iteration + ' --training_stage=' + str(training_stage))
    os.system('python3.5 main.py --dataset ' + dataset + ' --num_classes=' + str(num_classes) + ' --classes_proportion ' + classes_proportion + ' --ref_period=' + str(ref_period)\
                    + ' --epoch=50 --num_samples=' + str(num_samples) + ' --iteration ' + iteration + ' --training_stage=' + str(training_stage))
    
    fig = plt.figure(figsize=(8,8),dpi=250)
    img = mpimg.imread(best_fit_ac['folder']+'/fake_samples_binarized.png')
    plt.imshow(img)
    pp.savefig(fig)
    plt.close()
    
    
    
    
    fig = utils.plot_best_fit(best_fit_prob,folder+'/best_of_all_prob_fit')
    pp.savefig(fig)
    plt.close()
    #want to get samples from the best ac fit
    iteration = best_fit_prob['folder'][best_fit_prob['folder'].find('iteration')+10:]
    training_stage = int(int(best_fit_prob['epoch'])*num_samples/64+int(best_fit_prob['step'])+1)
    print('python3.5 main.py --dataset ' + dataset + ' --num_classes=' + str(num_classes) + ' --classes_proportion ' + classes_proportion + ' --ref_period=' + str(ref_period)\
              + ' --epoch=50 --num_samples=' + str(num_samples) + ' --iteration ' + iteration + ' --training_stage=' + str(training_stage))
    os.system('python3.5 main.py --dataset ' + dataset + ' --num_classes=' + str(num_classes) + ' --classes_proportion ' + classes_proportion + ' --ref_period=' + str(ref_period)\
                    + ' --epoch=50 --num_samples=' + str(num_samples) + ' --iteration ' + iteration + ' --training_stage=' + str(training_stage))
    
    fig = plt.figure(figsize=(8,8),dpi=250)
    img = mpimg.imread(best_fit_prob['folder']+'/fake_samples_binarized.png')
    plt.imshow(img)
    pp.savefig(fig)
    plt.close()
        

    
pp = PdfPages(main_folder+'/training_error_summary.pdf')
num_bins = 28
# UNIFORM FIRING RATES
# no refractory period    
dataset = 'uniform_fr'
num_classes = 1
classes_proportion = 'equal' 
ref_period = -1
num_samples = 8192#16384
get_figures(main_folder,dataset,num_classes,classes_proportion,num_samples,num_bins,ref_period,pp)

# refractory period = 2
dataset = 'uniform_fr'
num_classes = 1
classes_proportion = 'equal' 
ref_period = 2

get_figures(main_folder,dataset,num_classes,classes_proportion,num_samples,num_bins,ref_period,pp)


# GAUSSIAN FIRING RATES
# no refractory period
dataset = 'gaussian_fr'
num_classes = 1
classes_proportion = 'equal' 
ref_period = -1

get_figures(main_folder,dataset,num_classes,classes_proportion,num_samples,num_bins,ref_period,pp)


# refractory period = 2
dataset = 'gaussian_fr'
num_classes = 1
classes_proportion = 'equal' 
ref_period = 2

get_figures(main_folder,dataset,num_classes,classes_proportion,num_samples,num_bins,ref_period,pp)


# 2 classes / refractory period = -1
dataset = 'gaussian_fr'
num_classes = 2
classes_proportion = 'equal' 
ref_period = -1

get_figures(main_folder,dataset,num_classes,classes_proportion,num_samples,num_bins,ref_period,pp)

# 2 classes / refractory period = 2
dataset = 'gaussian_fr'
num_classes = 2
classes_proportion = 'equal' 
ref_period = 2

get_figures(main_folder,dataset,num_classes,classes_proportion,num_samples,num_bins,ref_period,pp)

# 2 classes / refractory period = -1 / classes proportion 70-30
dataset = 'gaussian_fr'
num_classes = 2
classes_proportion = '7030' 
ref_period = -1

get_figures(main_folder,dataset,num_classes,classes_proportion,num_samples,num_bins,ref_period,pp)


# 2 classes / refractory period = 2 / classes proportion 70-30
num_classes = 2
classes_proportion = '7030' 
ref_period = 2

get_figures(main_folder,dataset,num_classes,classes_proportion,num_samples,num_bins,ref_period,pp)


pp.close()
