#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  8 19:06:47 2017

@author: manuel
"""

import scipy.io as sio
import numpy as np 
import os
import matplotlib.pyplot as plt


def load_spikes(folder, movie, bin_size, num_bins):
    #load
    mat_contents = sio.loadmat(folder + movie + '.mat')
    #create figures folder
    folder_figures = folder + 'figures'
    if not os.path.exists(folder_figures):
        os.makedirs(folder_figures)
    
    #get spikes
    spks = mat_contents['Spikes']
    print(type(spks))
    
    duration = 0
    
    size_mat = spks.shape
    num_neurons = size_mat[0]
    
    num_movie_repetitions = size_mat[1]
    
    #find maximum spiketime that will be assumed to be the duration of one single movie repetition
    #(since trials have different numbers of spikes the function np.max is not able to compute the maximum)
    for ind_neuron in range(num_neurons):
        for ind_trial in range(num_movie_repetitions):
            trial = spks[ind_neuron][ind_trial]
            if trial.size !=0:
                trial_max = trial.max()
                duration = max([trial_max,duration])
        
    
    num_trials_per_movie_repetition = int(duration/(num_bins*bin_size))
    num_trials = int(num_trials_per_movie_repetition*num_movie_repetitions)
    
    binned_mat = np.zeros((num_neurons,num_trials, num_bins))
    maximo = 0
    minimo = 100000
    for ind_neuron in range(num_neurons):
        contador = 0
        for ind_trial in range(num_movie_repetitions):
            trial_spks = spks[ind_neuron][ind_trial]
            for ind_w in range(num_trials_per_movie_repetition):
                window_spks = trial_spks[((ind_w*bin_size*num_bins)<=trial_spks) & (((ind_w+1)*bin_size*num_bins)>trial_spks)] - ind_w*bin_size*num_bins
                
                if window_spks.size !=0:
                    maximo = np.max([maximo,np.max(window_spks)])
                    minimo = np.min([minimo,np.min(window_spks)])
                    num_spks = window_spks.size
                    for ind_spks in range(num_spks):
                        binned_mat[ind_neuron][contador][int(np.floor((window_spks[ind_spks])/bin_size))] = \
                        binned_mat[ind_neuron][contador][int(np.floor((window_spks[ind_spks])/bin_size))] + 1
                contador += 1    
 
    return binned_mat,spks



X,_ = load_spikes('/home/manuel/DATA/data/Scales/RetinalData/NeuralData/Spikes/','Movie2Exp1',1, 28)
X[X>0] = 1

print(np.shape(X))

for ind_n in range(np.shape(X)[0]):
    aux = X[ind_n,:,:]
    aux = np.mean(aux,axis=0)
    plt.plot(aux)
plt.show()
