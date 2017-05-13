#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 12 17:57:43 2017

@author: manuel
"""

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

def load_spikes(file, bin_size, num_bins,num_trials):
    duration = 20000
    #load
    mat_contents = sio.loadmat(file)
    
    spks = mat_contents['neuron_noise']
    
    assert np.shape(spks)[1]==num_trials, 'num of trials does not match'
    
    

    size_mat = spks.shape    
    num_trials = size_mat[1]
    num_samples_per_trial = int(duration/(num_bins*bin_size))
    num_samples = int(num_samples_per_trial*num_trials)
    
    binned_mat = np.zeros((1,num_samples, num_bins))
    maximo = 0
    minimo = 100000
    
    contador = 0
    for ind_trial in range(num_trials):
        trial_spks = spks[0,ind_trial]
        for ind_w in range(num_samples_per_trial):
            window_spks = trial_spks[((ind_w*bin_size*num_bins)<=trial_spks) & (((ind_w+1)*bin_size*num_bins)>trial_spks)] - ind_w*bin_size*num_bins
            
            if window_spks.size !=0:
                maximo = np.max([maximo,np.max(window_spks)])
                minimo = np.min([minimo,np.min(window_spks)])
                num_spks = window_spks.size
                for ind_spks in range(num_spks):
                    binned_mat[0,contador][int(np.floor((window_spks[ind_spks])/bin_size))] = \
                    binned_mat[0,contador][int(np.floor((window_spks[ind_spks])/bin_size))] + 1
                
                contador += 1    
 
    return binned_mat



def get_files(bin_size=1,num_bins=28):
    folder = '/home/manuel/DATA/data/Scales/auditory data Kayser/'
    #create figures folder
    folder_figures = folder + '/figures/'
    if not os.path.exists(folder_figures):
        os.makedirs(folder_figures)
    list_of_neurons = sio.loadmat(folder+'list_of_neurons.mat')
    list_of_neurons = list_of_neurons['neuron_list']
    for ind_n in range(np.shape(list_of_neurons)[0]):
        print(list_of_neurons[ind_n,0][0]+'_neuron_'+str(list_of_neurons[ind_n,1][0][0])+'_noise_'+str(list_of_neurons[ind_n,2][0][0]))
        X = load_spikes(folder+list_of_neurons[ind_n,0][0]+'_neuron_'+str(list_of_neurons[ind_n,1][0][0])+'_noise_'+str(list_of_neurons[ind_n,2][0][0])+'.mat',\
                    bin_size, num_bins,list_of_neurons[ind_n,3][0][0])
        X_reduced = X[0,:,:]
        autocorrelogram(X_reduced,folder_figures,list_of_neurons[ind_n,0][0]+'_neuron_'+str(list_of_neurons[ind_n,1][0][0])+'_noise_'+str(list_of_neurons[ind_n,2][0][0]))


def autocorrelogram(r,folder,name):
    
    #plot some samples
    num_rows = 6
    num_cols = 6
    samples_plot = r[0:num_rows*num_cols,:]
    #binnarize and plot over the probabilities
    fig,sbplt = plt.subplots(num_rows,num_cols)
    for ind_pl in range(np.shape(samples_plot)[0]):
        sbplt[int(np.floor(ind_pl/num_rows))][ind_pl%num_cols].plot(samples_plot[int(ind_pl),:])
        sbplt[int(np.floor(ind_pl/num_rows))][ind_pl%num_cols].set_ylim(0,np.max([1,np.max(samples_plot.flatten())]))
        
    
    fig.savefig(folder + '/samples_' + name + '.png',dpi=199, bbox_inches='tight')
    
    plt.close(fig)
    
    #compute average activity
    f = plt.figure()
    aux = np.mean(r,axis=0)
    plt.plot(aux)
    f.savefig(folder + '/average_activity_' + name + '.svg', bbox_inches='tight')
    #plt.show()
    plt.close(f)
    
    
    #get autocorrelogram
    lag = 10
    mean_spk_count = np.mean(np.sum(r,axis=1))
    std_spk_count = np.std(np.sum(r,axis=1))
    margin = np.zeros((r.shape[0],lag))
    #concatenate margins to then flatten the trials matrix
    r = np.hstack((margin,np.hstack((r,margin))))
    r_flat = r.flatten()
    spiketimes = np.nonzero(r_flat>0)
    ac = np.zeros(2*lag+1)
    for ind_spk in range(len(spiketimes[0])):
        spike = spiketimes[0][ind_spk]
        ac = ac + r_flat[spike-lag:spike+lag+1]
        
    f = plt.figure()
    index = np.linspace(-lag,lag,2*lag+1)
    plt.plot(index, ac)
    plt.title('mean spk-count = ' + str(round(mean_spk_count,3)) + ' (' + str(round(std_spk_count,3)) + ')')
    f.savefig(folder + '/autocorrelogram_' + name + '.png', bbox_inches='tight')
    plt.close(f)
   
    
    
    
    
        
get_files(1,28)      
      
        