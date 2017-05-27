#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 19 09:13:12 2017

@author: manuel
"""

#this functio generates figure 3, right panel of the NIPS paper
#from matplotlib.colors import LogNorm
import matplotlib.pyplot as plt
import numpy as np
#load data
data = np.load('/home/manuel/DCGAN-tensorflow/best fits/gaussian_fr numSamp 2048 numClass 1 propClass equal refPer -1 fr 0.5iter18best_of_all_ac_fit.npz')
f,sbplt = plt.subplots(1,1,figsize=(8, 8),dpi=250)

#plot hist2d
my_cmap = plt.cm.gray
_,_,_,Image = sbplt.hist2d(data['other_samples_freq_all_log'],data['other_samples_prob_all_log'],bins=[data['edges_x'], data['edges_y']],cmap = my_cmap)#)
plt.colorbar(Image)

sbplt.plot(data['new_fake_samples_freq_log'],data['new_fake_samples_prob_log'],'xr',markersize=6)

#tune the labels so they are in the form 10^x
ticks = sbplt.get_xticks()
labels = []
for ind_tck in range(len(ticks)):
    labels.append('$10^{'+str(ticks[ind_tck]) +'}$')
   
sbplt.set_xticklabels(labels)

ticks = sbplt.get_yticks()
labels = []
for ind_tck in range(len(ticks)):
    labels.append('$10^{'+str(ticks[ind_tck]) +'}$')
   
sbplt.set_yticklabels(labels)

