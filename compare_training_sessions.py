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


def get_figures(main_folder,dataset,num_classes,classes_proportion,num_samples,num_bins,ref_period,pp,firing_rate,option=None,neuron=None):
    folder_final_figures = main_folder + 'best fits'
    
    if option==9:
        folder = 'samples' + neuron + '_num_bins_' + str(num_bins) + '_iteration_0'
        title =   neuron + ' numSamp ' + str(num_samples) 
    else:
        folder = 'samples_dataset_' + dataset + '_num_classes_' + str(num_classes) + '_propClasses_' + classes_proportion + \
        '_num_samples_' + str(num_samples) + '_num_bins_' + str(num_bins) + '_ref_period_' + str(ref_period)+ '_firing_rate_' + str(firing_rate)  + '_iteration_0'
        
        title =   dataset + ' numSamp ' + str(num_samples) + ' numClass ' + str(num_classes) + ' propClass ' + classes_proportion + ' refPer ' + str(ref_period) + ' fr ' + str(firing_rate)
    
    
    best_fit_ac,best_fit_prob = utils.compare_trainings(main_folder+folder,title)
    iteration = best_fit_ac['folder'][best_fit_ac['folder'].find('iteration')+10:]
    
    fig = plt.figure(figsize=(8,8),dpi=250)
    img = mpimg.imread(main_folder+folder+'/training_error.png')
    plt.imshow(img)
    pp.savefig(fig)
    plt.close()
    
    #plot best ac fit
    if option==3:
        #for the gaussian, no refractory period and 1 class case we will also plot the numerical probability of the samples obtained from 500000 simulated samples
        fig = utils.plot_best_fit(best_fit_ac,folder_final_figures+'/'+title+'iter'+iteration+'best_of_all_ac_fit',best_fit_ac['ac_error'],best_fit_ac['spk_mean_error'],'',\
                                  num_classes,num_samples,num_bins, firing_rate,dataset,classes_proportion)
    else:
        fig = utils.plot_best_fit(best_fit_ac,folder_final_figures+'/'+title+'iter'+iteration+'best_of_all_ac_fit',best_fit_ac['ac_error'],best_fit_ac['spk_mean_error'],'')
        
    pp.savefig(fig)
    plt.close()
    
    #now I want to get samples from the best ac fit
    training_stage = int(int(best_fit_ac['epoch'])*num_samples/64+int(best_fit_ac['step'])+1) 
    if option==9:
        command = 'python3.5 main.py --dataset kayser_data ' + ' --iteration ' + iteration + ' --training_stage=' + str(training_stage)
    else:
        command = 'python3.5 main.py --dataset ' + dataset + ' --num_classes=' + str(num_classes) + ' --classes_proportion ' + classes_proportion + ' --ref_period=' + str(ref_period)\
          + ' --firing_rate=' + str(firing_rate) + ' --epoch=50 --num_samples=' + str(num_samples) + ' --iteration ' + iteration + ' --training_stage=' + str(training_stage)
    
    print(command)
    os.system(command)
    
    fig = plt.figure(figsize=(8,8),dpi=250)
    img = mpimg.imread(best_fit_ac['folder']+'/fake_samples.png')
    plt.imshow(img)
    pp.savefig(fig)
    plt.close()
    
    
       
 
def select_experiment(all_options=False,option=3,firing_rate=0.5,num_samples=8192,num_bins=28):   
    pp = PdfPages(main_folder+'/training_error_summary.pdf')
    
    #16384
    if option==1 or all_options:
        # UNIFORM FIRING RATES
        # no refractory period    
        dataset = 'uniform_fr'
        num_classes = 1
        classes_proportion = 'equal' 
        ref_period = -1
        get_figures(main_folder,dataset,num_classes,classes_proportion,num_samples,num_bins,ref_period,pp,firing_rate)
        if not all_options:
            pp.close()
            
    
    if option==2 or all_options:
        # refractory period = 2
        dataset = 'uniform_fr'
        num_classes = 1
        classes_proportion = 'equal' 
        ref_period = 2
        get_figures(main_folder,dataset,num_classes,classes_proportion,num_samples,num_bins,ref_period,pp,firing_rate)
        if not all_options:
            pp.close()

    if option==3 or all_options:
        # GAUSSIAN FIRING RATES
        # no refractory period
        dataset = 'gaussian_fr'
        num_classes = 1
        classes_proportion = 'equal' 
        ref_period = -1
        get_figures(main_folder,dataset,num_classes,classes_proportion,num_samples,num_bins,ref_period,pp,firing_rate,option)
        if not all_options:
            pp.close()

    if option==4 or all_options:
        # refractory period = 2
        dataset = 'gaussian_fr'
        num_classes = 1
        classes_proportion = 'equal' 
        ref_period = 2
        get_figures(main_folder,dataset,num_classes,classes_proportion,num_samples,num_bins,ref_period,pp,firing_rate)
        if not all_options:
            pp.close()

    if option==5 or all_options:
        # 2 classes / refractory period = -1
        dataset = 'gaussian_fr'
        num_classes = 2
        classes_proportion = 'equal' 
        ref_period = -1
        get_figures(main_folder,dataset,num_classes,classes_proportion,num_samples,num_bins,ref_period,pp,firing_rate)
        if not all_options:
            pp.close()
        
    if option==6 or all_options:
        # 2 classes / refractory period = 2
        dataset = 'gaussian_fr'
        num_classes = 2
        classes_proportion = 'equal' 
        ref_period = 2
        get_figures(main_folder,dataset,num_classes,classes_proportion,num_samples,num_bins,ref_period,pp,firing_rate)
        if not all_options:
            pp.close()

    if option==7 or all_options:
        # 2 classes / refractory period = -1 / classes proportion 70-30
        dataset = 'gaussian_fr'
        num_classes = 2
        classes_proportion = '7030' 
        ref_period = -1
        get_figures(main_folder,dataset,num_classes,classes_proportion,num_samples,num_bins,ref_period,pp,firing_rate)
        if not all_options:
            pp.close()


    if option==8 or all_options:
        # 2 classes / refractory period = 2 / classes proportion 70-30
        num_classes = 2
        classes_proportion = '7030' 
        ref_period = 2
        get_figures(main_folder,dataset,num_classes,classes_proportion,num_samples,num_bins,ref_period,pp,firing_rate)
        if not all_options:
            pp.close()
            
    if option==9:
        num_samples = 21420
        neuron = 'AuditoryDataM03.Jg_longnat3_sites2_neuron_4_noise_1.mat'
        get_figures(main_folder,0,0,0,num_samples,num_bins,0,pp,firing_rate,option,neuron)
        ###########
        num_samples = 19992 
        neuron = 'AuditoryDataM03.Jf_longnat3_sites3_neuron_3_noise_1.mat' #
        get_figures(main_folder,0,0,0,num_samples,num_bins,0,pp,firing_rate,option,neuron)
        
        num_samples = 19992 
        neuron = 'AuditoryDataM03.Jf_longnat3_sites3_neuron_4_noise_1.mat' #
        get_figures(main_folder,0,0,0,num_samples,num_bins,0,pp,firing_rate,option,neuron)
        if not all_options:
            pp.close()
        
    
    
    if all_options:
        pp.close()


select_experiment(False,3,num_samples=2048)
#select_experiment(False,4,num_samples=2048)
#select_experiment(False,3)
#select_experiment(False,4)
#select_experiment(False,5)
#select_experiment(False,7)
#select_experiment(False,9)
#select_experiment(False,3,num_samples=1024)
#select_experiment(False,4,num_samples=1024)
