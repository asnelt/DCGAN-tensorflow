#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 18 14:06:17 2017

@author: manuel
"""

import numpy as np
import utils
main_folder = '/home/manuel/DCGAN-tensorflow/'
folder_final_figures = main_folder + 'best fits'
firing_rate=0.5
num_samples=2048
num_bins=28
dataset = 'gaussian_fr'
num_classes = 1
classes_proportion = 'equal' 
ref_period = -1
iteration = '18'
title =   dataset + ' numSamp ' + str(num_samples) + ' numClass ' + str(num_classes) + ' propClass ' + classes_proportion + ' refPer ' + str(ref_period) + ' fr ' + str(firing_rate)
best_fit_ac = np.load('/home/manuel/DCGAN-tensorflow/samples_dataset_gaussian_fr_num_classes_1_propClasses_equal_num_samples_2048_num_bins_28_ref_period_-1_firing_rate_0.5_iteration_18/best_ac_fit.npz')

utils.plot_best_fit(best_fit_ac,folder_final_figures+'/'+title+'iter'+iteration+'best_of_all_ac_fit',best_fit_ac['ac_error']\
                    ,best_fit_ac['spk_mean_error'],'',num_classes,num_samples,num_bins, firing_rate,dataset,classes_proportion)