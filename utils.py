"""
Some codes from https://github.com/Newmu/dcgan_code
"""
from __future__ import division
import pprint
import numpy as np
import matplotlib.pyplot as plt
import ops
import os
import glob
import matplotlib

pp = pprint.PrettyPrinter()



def samples_statistics(r, name, parameters,d_loss=None,g_loss=None):
    #this function calculates the spike-count (mean and std), the frequency of the ground truth samples in r, 
    #the spike autocorrelogram and save all the data
    folder = parameters.sample_dir
    print('plot autocorrelogram')
    #first we get the average, std of the spike-count and the average time course
    mean_spk_count = np.mean(np.sum(r,axis=1))
    std_spk_count = np.std(np.sum(r,axis=1))
    profile_activity = np.mean(r,axis=0)
    
    #get samples probability
    if name=='real':
        r_unique = np.vstack({tuple(row) for row in r})
        num_samples = np.shape(r_unique)[0]#200
        samples = r_unique[0:num_samples,:]
        numerical_prob = np.zeros((num_samples,))
        for ind_s in range(num_samples):
            sample = samples[ind_s,:]
            sample_mat = np.tile(sample,(np.shape(r)[0] ,1))
            compare_mat = np.sum(np.abs(r-sample_mat),axis=1)
            numerical_prob[ind_s] = np.count_nonzero(compare_mat==0)/np.shape(r)[0]  
    else:
        real_data = np.load(folder + '/autocorrelogramreal.npz')
        samples = real_data['samples']
        numerical_prob = np.zeros((np.shape(samples)[0],))
        for ind_s in range(np.shape(samples)[0]):
            sample = samples[ind_s,:]
            sample_mat = np.tile(sample,(np.shape(r)[0],1))
            compare_mat = np.sum(np.abs(r-sample_mat),axis=1)
            numerical_prob[ind_s] = np.count_nonzero(compare_mat==0)/np.shape(r)[0]  
            
    #get autocorrelogram
    lag = 10
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
    f.savefig(folder + '/autocorrelogram' + name + '.svg', bbox_inches='tight')
    #plt.show()
    plt.close(f)
    if name=='real':
        data = {'mean':mean_spk_count,'std':std_spk_count,'acf':ac,'index':index,'prf_act':profile_activity,'samples':samples, 'prb_samples':numerical_prob, 'training_step':parameters.training_step}
    else:
        data = {'d_loss':d_loss,'g_loss':g_loss,'mean':mean_spk_count,'std':std_spk_count,'acf':ac,'index':index,'prf_act':profile_activity,'prb_samples':numerical_prob, 'training_step':parameters.training_step}
    
    np.savez(folder + '/autocorrelogram' + name + '.npz', **data)
    
    
def get_samples_autocorrelogram(sess, dcgan,name,parameters,d_loss,g_loss):
    #this function generate samples from the Spike-GAN in dcgan and calls the function to calculate their statistics (spike-count, autocorrelogram...)
    folder = parameters.sample_dir
    num_samples = int(2**np.log2(parameters.num_samples))
    num_trials = int(num_samples/dcgan.batch_size)
    X = np.ndarray((num_samples,int(dcgan.output_height),1))    
    for ind_tr in range(num_trials):
        z_sample = np.random.uniform(-1, 1, size=(dcgan.batch_size, dcgan.z_dim))
        X[np.arange(ind_tr*dcgan.batch_size,(ind_tr+1)*dcgan.batch_size),:,:] \
                = sess.run(dcgan.sampler, feed_dict={dcgan.z: z_sample})
    
    binarized_X = ops.binarize(X)
    binarized_X_reduced = binarized_X[:,:,0]
    
    data = {'fake_samples':binarized_X_reduced}
    np.savez(folder + '/fake_samples.npz', **data)
    #compute statistics
    samples_statistics(binarized_X_reduced,name,parameters,d_loss,g_loss)  
    
    #plot average activity
    f = plt.figure()
    aux = np.mean(binarized_X_reduced,axis=0)
    plt.plot(aux)
    f.savefig(folder + '/average_activity' + name + '.svg', bbox_inches='tight')
    #plt.show()
    plt.close(f)
      
def get_samples(sess,dcgan,folder):
    #this function generate samples from the Spike-GAN in dcgan and calls the function to plot them
    z_sample = np.random.uniform(-1, 1, size=(dcgan.batch_size, dcgan.z_dim))
    samples_plot = sess.run(dcgan.sampler, feed_dict={dcgan.z: z_sample})
    
    plot_samples(samples_plot,folder,'fake')
    
def plot_samples(samples_plot,folder,name):  
    #this function plots the samples in samples_plot in two different formats
    num_rows = 6
    num_cols = 6
    samples_plot = samples_plot[0:num_rows*num_cols,:]
    num_bins = np.shape(samples_plot)[1]
    #binnarize and plot over the probabilities
    samples_plot_bin = ops.binarize(samples_plot)
    fig,sbplt = plt.subplots(num_rows,num_cols)
    for ind_pl in range(np.shape(samples_plot_bin)[0]):
        sbplt[int(np.floor(ind_pl/num_rows))][ind_pl%num_cols].plot(samples_plot_bin[int(ind_pl),:])
        sbplt[int(np.floor(ind_pl/num_rows))][ind_pl%num_cols].set_ylim(0,np.max([1,np.max(samples_plot.flatten())]))
        #sbplt[int(np.floor(ind_pl/num_rows))][ind_pl%num_cols].axis('off')
    
    for ind_pl in range(np.shape(samples_plot)[0]):
        sbplt[int(np.floor(ind_pl/num_rows))][ind_pl%num_cols].plot(samples_plot[int(ind_pl),:],'--')
        sbplt[int(np.floor(ind_pl/num_rows))][ind_pl%num_cols].set_ylim(0,np.max([1,np.max(samples_plot.flatten())]))
        sbplt[int(np.floor(ind_pl/num_rows))][ind_pl%num_cols].axis('off')
    
    
    fig.savefig(folder + '/' + name + '_samples.svg',dpi=199, bbox_inches='tight')
    fig.savefig(folder + '/' + name + '_samples.png',dpi=199, bbox_inches='tight')
    plt.close(fig)
    
    #here we plot the rasters together with the probabilities (NIPS paper, Fig. 2, top row)
    fig = plt.figure(figsize=(8,4),dpi=250)
    num_samples_raster = 6
    for ind_r in range(num_rows):
        plt.subplot(num_rows,1,ind_r+1)
        for ind_pl in range(num_samples_raster):
            spiketimes = np.nonzero(samples_plot_bin[ind_pl+num_samples_raster*ind_r,:]!=0)[0] + ind_pl*num_bins+2
            for ind_spk in range(len(spiketimes)):
                plt.plot(spiketimes[ind_spk]*np.ones((2,)),[1.1,1.2],'b')

            plt.plot(np.arange(ind_pl*num_bins,(ind_pl+1)*num_bins)+2,samples_plot[ind_pl+num_samples_raster*ind_r,:],'g')
            
        plt.ylim(0,1.5)
        plt.xlim(0,num_samples_raster*num_bins)
        plt.xlim(0,num_samples_raster*num_bins)
        plt.axis('off')
    fig.savefig(folder + '/' + name + '_rasters.svg',dpi=199, bbox_inches='tight')
    fig.savefig(folder + '/' + name + '_rasters.png',dpi=199, bbox_inches='tight')

def evaluate_training(folder,sbplt,ind):
    #this function evaluates the error in the spike-count, average time course and spike autocorrelogram for all training steps
    #and identifies the training step corresponding to the best autocorrelogram fit
    mycwd = os.getcwd()
    os.chdir(folder)
    real_data = np.load('autocorrelogramreal.npz')
    real_acf = real_data['acf']
    if np.max(real_acf)>0:
        real_acf = real_acf/np.max(real_acf)
    real_spkC_mean = real_data['mean']
    real_spkC_std = real_data['std']
    real_spkC_prf_act = real_data['prf_act']
    real_probs_samples = real_data['prb_samples']
    training_step = real_data['training_step']
    
    best_ac_fit = {}
    best_ac_fit['acf'] = real_acf
    best_ac_fit['mean'] = real_data['mean']
    best_ac_fit['std'] = real_data['std']
    best_ac_fit['prf_act'] = real_data['prf_act']  
    best_ac_fit['prb_samples'] = real_data['prb_samples']  
    best_ac_fit['folder'] = folder
    best_ac_fit['iteration'] = folder[folder.find('iteration')+10:]
    best_mean_prob_fit = best_ac_fit.copy()
   
       
    files = glob.glob("autocorrelogramtrain_*.npz")
    error_ac = np.empty((len(files),))
    spkC_mean = np.empty((len(files),))
    spkC_std = np.empty((len(files),))
    error_spkC_prf_act = np.empty((len(files),))
    error_probs_samples = np.empty((len(files),))
    train_step = np.empty((len(files),))
    training_probs_mat = np.empty((len(files),len(real_probs_samples)))
    min_error_ac = 1000000
    min_error_mean_prob = 1000000
    for ind_f in range(len(files)):
        #get epoch and step from name
        name = files[ind_f]  
        find_us = name.find('_')
        find_us2 = name[find_us+1:].find('_')
        find_dot = name.find('.')
        
        epoch = (name[find_us+1:find_us+find_us2+1])
        step = name[find_us+find_us2+2:find_dot]
        train_step[ind_f] = int(epoch+step)
        training_data = np.load(name)
        #acticity profile
        training_spkC_prf_act = training_data['prf_act']
        error_spkC_prf_act[ind_f] = np.sum(np.abs(real_spkC_prf_act-training_spkC_prf_act))
        #autocorrelogram
        training_acf = training_data['acf']
        if np.max(training_acf)>0:
            training_acf = training_acf/np.max(training_acf)
        error_ac[ind_f] = np.sum(np.abs(real_acf-training_acf))
        spkC_mean[ind_f] = np.abs(real_spkC_mean-training_data['mean'])
        #spikeCount std
        spkC_std[ind_f] = np.abs(real_spkC_std-training_data['std'])
         
        #mean probability of samples
        training_probs_mat[ind_f,:] = np.abs(real_probs_samples-training_data['prb_samples'])#np.abs(real_probs_samples-training_data['prb_samples'])/real_probs_samples
        
        prob_logs = training_data['prb_samples']*np.log2(training_data['prb_samples']/real_probs_samples)
        prob_logs = np.delete(prob_logs,np.nonzero(training_data['prb_samples']==0))
        error_probs_samples[ind_f] = np.sum(prob_logs)+(1-np.sum(training_data['prb_samples']))*np.log2((1-np.sum(training_data['prb_samples']))/np.min(real_probs_samples))
        
       
        if min_error_ac>error_ac[ind_f]:
            min_error_ac = error_ac[ind_f]
            best_ac_fit['acf_fake'] = training_acf
            best_ac_fit['mean_fake'] = training_data['mean']
            best_ac_fit['std_fake'] = training_data['std']
            best_ac_fit['prf_act_fake'] = training_data['prf_act']
            best_ac_fit['prob_samples_fake'] = training_data['prb_samples']
            best_ac_fit['epoch'] = epoch
            best_ac_fit['step'] = step
            best_ac_fit['error'] = min_error_ac
        if min_error_mean_prob>error_probs_samples[ind_f]:
            min_error_mean_prob = error_probs_samples[ind_f]
            best_mean_prob_fit['acf_fake'] = training_acf
            best_mean_prob_fit['mean_fake'] = training_data['mean']
            best_mean_prob_fit['std_fake'] = training_data['std']
            best_mean_prob_fit['prf_act_fake'] = training_data['prf_act']
            best_mean_prob_fit['prob_samples_fake'] = training_data['prb_samples']
            best_mean_prob_fit['epoch'] = epoch
            best_mean_prob_fit['step'] = step
            best_mean_prob_fit['error'] = min_error_mean_prob
    
   
    #sort traces
    indices = np.argsort(train_step)
    error_ac = np.array(error_ac)[indices]
    spkC_mean = np.array(spkC_mean)[indices]
    spkC_std = np.array(spkC_std)[indices]
    error_spkC_prf_act = np.array(error_spkC_prf_act)[indices]
    error_probs_samples = np.array(error_probs_samples)[indices]
    training_probs_mat  = np.array(training_probs_mat)[indices]
    
    #plot prob of each real sample
    maximo = np.max(np.concatenate((real_probs_samples,training_probs_mat.flatten()),axis=0))
    minimo = np.min(np.concatenate((real_probs_samples,training_probs_mat.flatten()),axis=0))
    f = plt.figure()
    plt.subplot(2,1,1)
    plt.imshow(training_probs_mat,vmin=minimo,vmax=maximo,aspect='auto')
    plt.colorbar()
    plt.subplot(2,1,2)
    real_probs_samples = np.expand_dims(real_probs_samples,axis=0)
    plt.imshow(real_probs_samples,vmin=minimo,vmax=maximo,aspect='auto')
    plt.colorbar()
   
    f.savefig('probs_mat.png',dpi=600, bbox_inches='tight')
    plt.close()
    
    #put traces on best fit dictionary 
    best_ac_fit['spk_mean_error'] = spkC_mean
    best_ac_fit['ac_error'] = error_ac
    best_mean_prob_fit['spk_mean_error'] = spkC_mean
    best_mean_prob_fit['ac_error'] = error_ac
    
    #plot training error traces
    sbplt[0][0].plot(error_ac)
    sbplt[0][0].set_title('AC error')
    sbplt[0][0].set_xlabel('training step (' + str(training_step) + ' batches)')
    sbplt[0][0].set_ylabel('L1 distance')
    sbplt[0][1].plot(spkC_mean)
    #sbplt[0][1].plot(np.ones(len(files),)*real_spkC_mean)
    sbplt[0][1].set_title('spk-count mean error')
    sbplt[0][1].set_xlabel('training step (' + str(training_step) + ' batches)')
    sbplt[0][1].set_ylabel('absolute difference')
    sbplt[1][0].plot(spkC_std)
    #sbplt[1][0].plot(np.ones(len(files),)*real_spkC_std)
    sbplt[1][0].set_title('spk-count std error')
    sbplt[1][0].set_xlabel('training step (' + str(training_step) + ' batches)')
    sbplt[1][0].set_ylabel('absolute difference')
    sbplt[1][1].plot(error_probs_samples,label=best_ac_fit['iteration'])
    sbplt[1][1].set_title('samples mean probability error')
    sbplt[1][1].set_xlabel('training step (' + str(training_step) + ' batches)')
    sbplt[1][1].set_ylabel('absolute difference')
    
    #save training error traces only for the current simulation
    f,sbplt2 = plt.subplots(2,1,figsize=(8, 8),dpi=250)
    
    left  = 0.125  # the left side of the subplots of the figure
    right = 0.9    # the right side of the subplots of the figure
    bottom = 0.1   # the bottom of the subplots of the figure
    top = 0.9      # the top of the subplots of the figure
    wspace = 0.4   # the amount of width reserved for blank space between subplots
    hspace = 0.4   # the amount of height reserved for white space between subplots

    matplotlib.rcParams.update({'font.size': 8})
    plt.subplots_adjust(left=left, bottom=bottom, right=right, top=top, wspace=wspace, hspace=hspace)
    sbplt2[1].plot(error_ac)
    sbplt2[1].set_title('AC error')
    sbplt2[1].set_xlabel('training step (' + str(training_step) + ' batches)')
    sbplt2[1].set_ylabel('L1 distance')
    sbplt2[0].plot(spkC_mean)
    #sbplt2[0].plot(np.ones(len(files),)*real_spkC_mean)
    sbplt2[0].set_title('spk-count mean error')
    sbplt2[0].set_xlabel('training step (' + str(training_step) + ' batches)')
    sbplt2[1].set_ylabel('absolute difference')
    f.savefig('training error traces alone.svg',dpi=600, bbox_inches='tight')
    
    
    
    #plot best fits
    plot_best_fit(best_ac_fit,'best_ac_fit',error_ac,spkC_mean,training_step)
    plt.close()
    
    plot_best_fit(best_mean_prob_fit,'best_mean_prob_fit',error_ac,spkC_mean,training_step)
    plt.close()
    
    plt.close()
    
    os.chdir(mycwd)
    
    return(best_ac_fit,best_mean_prob_fit)
    
def compare_trainings(folder,title):
    #this function goes over all Spike-GAN instances corresponding to a given configuration and returns the best of them
    print('--------------------------------------------------')
    print(folder)
    find_aux = folder.find('iteration')
    files = glob.glob(folder[0:find_aux]+'*')
    #figure for all training error across epochs (supp. figure 2)
    f,sbplt = plt.subplots(2,2,figsize=(8, 8),dpi=250)
    
    left  = 0.125  # the left side of the subplots of the figure
    right = 0.9    # the right side of the subplots of the figure
    bottom = 0.1   # the bottom of the subplots of the figure
    top = 0.9      # the top of the subplots of the figure
    wspace = 0.4   # the amount of width reserved for blank space between subplots
    hspace = 0.4   # the amount of height reserved for white space between subplots

    matplotlib.rcParams.update({'font.size': 8})
    plt.subplots_adjust(left=left, bottom=bottom, right=right, top=top, wspace=wspace, hspace=hspace)
    min_error_ac = 100000000
    min_error_prob = 100000000
    for ind_f in range(len(files)):
        print(files[ind_f])
        best_candidate_ac,best_candidate_prob = evaluate_training(files[ind_f],sbplt,ind_f)
        if min_error_ac>best_candidate_ac['error']:
            min_error_ac = best_candidate_ac['error']
            best_fit_ac = best_candidate_ac.copy()
        if min_error_prob>best_candidate_prob['error']:
            min_error_prob = best_candidate_prob['error']
            best_fit_prob = best_candidate_prob.copy()

    plt.legend(shadow=True, fancybox=True)
    plt.suptitle(title)
    plt.show()
    
    f.savefig(folder+'/training_error.png',dpi=300, bbox_inches='tight')
    f.savefig(folder+'/training_error.svg',dpi=300, bbox_inches='tight')
    plt.close()
    
    
    return(best_fit_ac,best_fit_prob)
    
    
def plot_best_fit(data,name,error_ac,spkC_mean,training_step,
                  num_classes=None,num_samples=None,num_bins=None, firing_rate=None,dataset=None,classes_proportion=None):
    #this function plots most of the figures used in the NIPS paper for the best result Spike-instances
    lag = 10
    fake_mean = float(data['mean_fake'])    
    fake_std = float(data['std_fake'])
    real_mean = float(data['mean'])
    real_std = float(data['std'])
    left  = 0.125  # the left side of the subplots of the figure
    right = 0.9    # the right side of the subplots of the figure
    bottom = 0.1   # the bottom of the subplots of the figure
    top = 0.9      # the top of the subplots of the figure
    wspace = 0.4   # the amount of width reserved for blank space between subplots
    hspace = 0.4   # the amount of height reserved for white space between subplots
    
    
    
    if num_classes!=None:
        #only for the paramater configuration shown in figure 2, middle row (gaussian, 1 class, no refractory period)
        plot_probabilities(data,name,firing_rate,num_classes,num_bins,num_samples,classes_proportion)
        
    
    #plot training error traces (this plot corresponds to figures 2 (middle and bottom rows), 4 and 5 in the NIPS paper)
    f,sbplt2 = plt.subplots(1,2,figsize=(8, 2),dpi=250)
    matplotlib.rcParams.update({'font.size': 8})
    plt.subplots_adjust(left=left, bottom=bottom, right=right, top=top, wspace=wspace, hspace=hspace)
    index = np.linspace(-lag,lag,2*lag+1)   
    sbplt2[0].plot(spkC_mean,'k')
    sbplt2[0].set_title('spk-count mean error')
    sbplt2[0].set_xlabel('training epoch')
    sbplt2[0].set_ylabel('absolute difference')
    sbplt2[1].plot(error_ac,'k')
    sbplt2[1].set_title('AC error')
    sbplt2[1].set_xlabel('training epoch')
    sbplt2[1].set_ylabel('L1 distance')
    
    #inset for spike autocorrelogram fit
    plt.axes([.65, .42, .2, .4])
    plt.plot(index,data['acf'],'b',label='real')
    plt.plot(index,data['acf_fake'],'--r',label='fake')
   
    plt.xlabel('time')
    #plt.ylabel('proportion of spikes')
    
    #inset for average time course fit
    plt.axes([.2, .42, .2, .4])
    plt.plot(data['prf_act']*1000,'b',label='real')
    plt.plot(data['prf_act_fake']*1000,'r',label='fake')
    plt.title('mean spk-count: ' + "{0:.2f}".format(fake_mean) + ' (' + "{0:.2f}".format(fake_std) + '). Real: ' + "{0:.2f}".format(real_mean) + ' (' + "{0:.2f}".format(real_std) + ')')
    plt.xlabel('time')
    #plt.ylabel('mean firing rate (Hz)')
    maximo = 1000*np.max(np.concatenate([data['prf_act'],data['prf_act_fake']]))
    plt.ylim(0,maximo+maximo/10)
    plt.suptitle('iteration ' + str(data['iteration']) + ' epoch ' + str(data['epoch']) + ' step ' + str(data['step'])) 
    #plt.legend(shadow=True, fancybox=True)
    plt.show()
    f.savefig(name + '.png',dpi=300, bbox_inches='tight')
    f.savefig(name + '.svg',dpi=300, bbox_inches='tight') 
    plt.close()
    np.savez(name + '.npz', **data)
    return(f)
   

        
def plot_probabilities(data,name,firing_rate,num_classes,num_bins,num_samples,classes_proportion):
    left  = 0.125  # the left side of the subplots of the figure
    right = 0.9    # the right side of the subplots of the figure
    bottom = 0.1   # the bottom of the subplots of the figure
    top = 0.9      # the top of the subplots of the figure
    wspace = 0.4   # the amount of width reserved for blank space between subplots
    hspace = 0.4   # the amount of height reserved for white space between subplots
    f,sbplt = plt.subplots(1,3,figsize=(8, 3),dpi=250)
    matplotlib.rcParams.update({'font.size': 8})
    plt.subplots_adjust(left=left, bottom=bottom, right=right, top=top, wspace=wspace, hspace=hspace)
    folder = str(data['folder'])
    #calculate the numerical probabilities of the ground truth dataset samples 
    real_samples = np.load(folder + '/autocorrelogramreal.npz')
    real_samples = real_samples['samples']
    aux = np.load('/home/manuel/DCGAN-tensorflow/samples/numerical_probs_num_samples_500000.npz')
    theoretical_probs = aux['numerical_prob']
    theoretical_samples = aux['diff_samples']
    real_samples_prob = np.zeros((np.shape(real_samples)[0],))
    for ind_s in range(np.shape(real_samples)[0]):
        sample = real_samples[ind_s,:]
        sample_mat = np.tile(sample,(np.shape(theoretical_samples)[0],1))
        compare_mat = np.sum(np.abs(theoretical_samples-sample_mat),axis=1)
        if np.count_nonzero(compare_mat==0)==1:
            real_samples_prob[ind_s] = theoretical_probs[np.nonzero(compare_mat==0)]
        else: 
            assert np.count_nonzero(compare_mat==0)==0
    
    #now plot the frequencies of the same samples in the ground truth and generated datasets against their numerical probabilities       
    sbplt[0].loglog(data['prob_samples_fake'],real_samples_prob,'xr',basex=10)
    sbplt[0].loglog(data['prb_samples'],real_samples_prob,'+b',basex=10)
    equal_line =   np.linspace(0.0005,0.05,10000)
    sbplt[0].loglog(equal_line,equal_line,basex=10)
    sbplt[0].set_xlabel('frequencies of samples in real dataset')
    sbplt[0].set_ylabel('theoretical probabilities')
    sbplt[0].set_title(str(np.sum(data['prob_samples_fake'])))
    
    
    #we will now compute the numerical prob of the samples generated by Spike-gan that are not in the original ground truth dataset
    #load samples generated by the GAN
    fake_samples = np.load(folder + '/fake_samples.npz')
    fake_samples = fake_samples['fake_samples']
    #get the new and impossible samples
    fake_samples_prob,fake_samples_freq,new_sample = get_new_and_impossible_samples(fake_samples,theoretical_samples,real_samples,theoretical_probs)
    new_fake_samples = fake_samples[new_sample==1,:]
    impossible_fake_samples = fake_samples[fake_samples_prob==0,:]
    
    #get more real datasets from the underlying distribution and compute the numerical prob of the samples 
    #present in these new real datasets that are not in the original ground truth dataset
    num_real_dataset = 1000
    equal_line =   np.linspace(0.0005,0.005,10000)
    num_of_impossible_samples = np.zeros((num_real_dataset,))
    counter = 0
    other_samples_prob_all = np.zeros((num_real_dataset*num_samples,))
    other_samples_freq_all = np.zeros((num_real_dataset*num_samples,))
    counter_num_other_samples = 0
    for ind_f in range(num_real_dataset):
        #here we get more samples
        other_real_samples = get_more_real_samples_gaussian_no_refrPer(firing_rate,num_classes,num_bins,num_samples,classes_proportion)
        #calculate the numerical probs and frequencies of the samples as well as whether they are new
        other_samples_prob,other_samples_freq,other_new_sample = get_new_and_impossible_samples(other_real_samples,theoretical_samples,real_samples,theoretical_probs)

        num_of_impossible_samples[counter] = np.sum(other_samples_freq[other_samples_prob==0])
        
        other_samples_prob_all[counter_num_other_samples:counter_num_other_samples+np.count_nonzero(other_new_sample==1)] = other_samples_prob[other_new_sample==1]
        other_samples_freq_all[counter_num_other_samples:counter_num_other_samples+np.count_nonzero(other_new_sample==1)] = other_samples_freq[other_new_sample==1]
        counter = counter + 1
        counter_num_other_samples = counter_num_other_samples+np.count_nonzero(other_new_sample==1)
    
    
    #in these two vectors we have the numerical probs and freq of the new samples (not present in the original ground truth sample)
    other_samples_prob_all = other_samples_prob_all[0:counter_num_other_samples]
    other_samples_freq_all = other_samples_freq_all[0:counter_num_other_samples]
    
    #we get rid of the freqs for which the numerical prob is zero (so we can compute the logs)
    other_samples_freq_all = np.delete(other_samples_freq_all,np.nonzero(other_samples_prob_all==0))
    other_samples_prob_all = np.delete(other_samples_prob_all,np.nonzero(other_samples_prob_all==0))
    
    #compute the logs of the probs and freq
    other_samples_prob_all_log = np.log10(other_samples_prob_all)
    other_samples_freq_all_log = np.log10(other_samples_freq_all) 
    
    #now we want to get the bins for the hist2d
    aux = np.unique(other_samples_freq_all_log)
    bin_size = 2*np.min(np.diff(aux))
    edges_x = np.unique(np.concatenate((aux-bin_size/2,aux+bin_size/2)))
    
    edges_y = np.linspace(np.min(other_samples_prob_all_log)-0.1,np.max(other_samples_prob_all_log)+0.1,10)
    my_cmap = plt.cm.gray
    _,_,_,Image = sbplt[1].hist2d(other_samples_freq_all_log,other_samples_prob_all_log,bins=[edges_x, edges_y],cmap = my_cmap)#
    plt.colorbar(Image)
    
    #here we do the same as for the surrogate real datasets but for the generated dataset
    new_fake_samples_freq = fake_samples_freq[new_sample==1]
    new_fake_samples_prob = fake_samples_prob[new_sample==1]
    
    new_fake_samples_freq = np.delete(new_fake_samples_freq,np.nonzero(new_fake_samples_prob==0))
    new_fake_samples_prob = np.delete(new_fake_samples_prob,np.nonzero(new_fake_samples_prob==0))
    
    new_fake_samples_prob_log = np.log10(new_fake_samples_prob)
    new_fake_samples_freq_log = np.log10(new_fake_samples_freq)
     
    #transalate ticks to the 10^x format      
    sbplt[1].plot(new_fake_samples_freq_log,new_fake_samples_prob_log,'xr',markersize=2)
    ticks = sbplt[1].get_xticks()
    labels = []
    for ind_tck in range(len(ticks)):
        labels.append('$10^{'+str(ticks[ind_tck]) +'}$')
   
    sbplt[1].set_xticklabels(labels)
    
    ticks = sbplt[1].get_yticks()
    labels = []
    for ind_tck in range(len(ticks)):
        labels.append('$10^{'+str(ticks[ind_tck]) +'}$')
   
    sbplt[1].set_yticklabels(labels)


    #here we plot the histogram of the frequency of samples with numerical prob = 0
    sbplt[2].hist(num_of_impossible_samples)  
    sbplt[2].plot(np.sum(fake_samples_freq[fake_samples_prob==0])*np.ones((10,1)),np.arange(10),'r')
    #For some reason, python gives an error when trying to save this figure
    #    f.savefig(name + 'samples_probabilities.png',dpi=300, bbox_inches='tight')
    #    f.savefig(name + 'samples_probabilities.svg',dpi=300, bbox_inches='tight') 
    #    plt.close()
    
    #save data
    impossible_samples_probs = {'other_samples_freq_all_log':other_samples_freq_all_log,'other_samples_prob_all_log':other_samples_prob_all_log,\
            'edges_x':edges_x,'edges_y':edges_y,'new_fake_samples_freq_log':new_fake_samples_freq_log,'new_fake_samples_prob_log':new_fake_samples_prob_log}
    np.savez(name + 'new.npz', **impossible_samples_probs)
        

    #plot some new and impossible samples examples
    num_rows = 6
    num_cols = 6
    fig,sbplt = plt.subplots(num_rows,num_cols)
    for ind_pl in range(int(num_rows*num_rows)):
        if ind_pl<num_rows*num_cols/2:
            sbplt[int(np.floor(ind_pl/num_rows))][ind_pl%num_cols].plot(new_fake_samples[int(ind_pl),:],'b')
            sbplt[int(np.floor(ind_pl/num_rows))][ind_pl%num_cols].axis('off')
        else:
            sbplt[int(np.floor(ind_pl/num_rows))][ind_pl%num_cols].plot(impossible_fake_samples[int(ind_pl),:],'r')
            sbplt[int(np.floor(ind_pl/num_rows))][ind_pl%num_cols].axis('off')
    
    fig.savefig(name + 'new_and_impossible_samples.png',dpi=300, bbox_inches='tight')
    fig.savefig(name + 'new_and_impossible_samples.svg',dpi=300, bbox_inches='tight') 
    plt.close()

def get_new_and_impossible_samples(fake_samples,theoretical_samples,real_samples,theoretical_probs):
    #this function calculates the numerical probs and frequencies of the samples contained in fake_samples as well as whether they are new
    fake_samples_prob = np.zeros((np.shape(fake_samples)[0],))
    fake_samples_freq = np.zeros((np.shape(fake_samples)[0],))
    new_sample =  np.zeros((np.shape(fake_samples)[0],))
    for ind_s in range(np.shape(fake_samples)[0]):
        #get theoretical probability of sample
        sample = fake_samples[ind_s,:]
        sample_mat = np.tile(sample,(np.shape(theoretical_samples)[0],1))
        compare_mat = np.sum(np.abs(theoretical_samples-sample_mat),axis=1)
        if np.count_nonzero(compare_mat==0)==1:
            fake_samples_prob[ind_s] = theoretical_probs[np.nonzero(compare_mat==0)]
        else:
            assert np.count_nonzero(compare_mat==0)==0
    
        #calculate frequency of the sample in the fake dataset
        sample_mat = np.tile(sample,(np.shape(fake_samples)[0],1))
        compare_mat = np.sum(np.abs(fake_samples-sample_mat),axis=1)
        fake_samples_freq[ind_s] = np.count_nonzero(compare_mat==0)/np.shape(fake_samples)[0]  
    
        #check whether the sample was already in the real samples dataset
        sample_mat = np.tile(sample,(np.shape(real_samples)[0],1))
        compare_mat = np.sum(np.abs(real_samples-sample_mat),axis=1)
        new_sample[ind_s] = np.count_nonzero(compare_mat==0)==0

    return(fake_samples_prob,fake_samples_freq,new_sample)

def get_more_real_samples_gaussian_no_refrPer(firing_rate=0.5,num_classes=1,num_bins=28,num_samples=2048,classes_proportion='equal'):
    #this function creates more samples coming from the underlying distribution using the parameters specified in the input
    
    noise = 0.01*firing_rate
    margin = 6 #num bins from the middle one that the response peaks will span (see line 389)
    std_resp = 4 #std of the gaussian defining the firing rates
    #check the proportion of the two classes
    if classes_proportion=='equal':
        mat_prop_classes = np.ones((num_classes,))/num_classes
    elif classes_proportion=='7030':    
        mat_prop_classes = [0.7,0.3]
    else:
        raise ValueError("Unknown dataset '" + classes_proportion + "'")
        
        
    t = np.arange(num_bins)
    
    peaks1 = np.linspace(int(num_bins/2)-margin,int(num_bins/2)+margin,num_classes)
    X =np.zeros((num_samples,num_bins))
    
    for ind in range(num_samples):
        stim = np.random.choice(num_classes,p=mat_prop_classes)
        fr = firing_rate*np.exp(-(t-peaks1[stim])**2/std_resp**2) + np.random.normal(0,noise,(1,num_bins))
        fr[fr<0] = 0
        r = fr > np.random.random(fr.shape)
        r = r.astype(float)
        r[r>0] = 1
    
        X[ind,:] = r
        
    return(X)

def probability_data(data,num_classes,num_samples,num_bins, firing_rate,dataset,classes_proportion):
    #this function calculates the theoretical probability of a dataset with respect to the 
    #distribution defined by the input parameters. The function was not used to obtained the results
    #shown in the NIPS paper.
    num_classes = num_classes
    num_samples = int(np.shape(data)[0])
    num_bins = num_bins
    firing_rate = firing_rate
    if dataset=='gaussian_fr':
        margin = 6 #num bins from the middle one that the response peaks will span (see line 389)
        std_resp = 4 #std of the gaussian defining the firing rates
        if classes_proportion=='equal':
            mat_prop_classes = np.ones((num_classes,))/num_classes
        elif classes_proportion=='7030':    
            mat_prop_classes = [0.7,0.3]
        else:
            raise ValueError("Unknown dataset '" + classes_proportion + "'")
             
            
        t = np.arange(num_bins)
        
        peaks1 = np.linspace(int(num_bins/2)-margin,int(num_bins/2)+margin,num_classes)
        
        probs_mat = np.zeros((num_classes,num_samples))
        #calculate the probability of getting each sample and average it
        for ind in range(num_classes):
            fr = firing_rate*np.exp(-(t-peaks1[ind])**2/std_resp**2)
            fr_noSpk = np.tile(fr,(num_samples,1))
            fr_Spk = np.tile(fr,(num_samples,1))
            fr_noSpk[data==1] = 0
            prob_noSpk_bins = np.prod(1-fr_noSpk,axis=1)
            fr_Spk[data==0] = 1
            prob_Spk_bins = np.prod(fr_Spk,axis=1)
            probs_mat[ind,:] = prob_noSpk_bins*prob_Spk_bins*mat_prop_classes[ind]
            
        probData = np.sum(probs_mat,axis=0)
        #probData = probData/np.sum(probData)  
            
    elif dataset=='uniform_fr':
        fr_noSpk =np.zeros((num_samples,num_bins)) + firing_rate
        fr_Spk =np.zeros((num_samples,num_bins)) + firing_rate
        fr_noSpk[data==1] = 0
        prob_noSpk_bins = np.prod(1-fr_noSpk,axis=1)
        fr_Spk[data==0] = 1
        prob_Spk_bins = np.prod(fr_Spk,axis=1)
        probData = prob_noSpk_bins*prob_Spk_bins
        
    return(probData)    