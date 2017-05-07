"""
Some codes from https://github.com/Newmu/dcgan_code
"""
from __future__ import division
import math
import pprint
import scipy.misc
import numpy as np
import matplotlib.pyplot as plt
import ops
import os
import glob
import matplotlib

pp = pprint.PrettyPrinter()

get_stddev = lambda x, k_h, k_w: 1/math.sqrt(k_w*k_h*x.get_shape()[-1])

def get_image(image_path, input_height, input_width,
              resize_height=64, resize_width=64,
              is_crop=True, is_grayscale=False):
  image = imread(image_path, is_grayscale)
  return transform(image, input_height, input_width,
                   resize_height, resize_width, is_crop)

def save_images(images, size, image_path):
  return imsave(inverse_transform(images), size, image_path)

def imread(path, is_grayscale = False):
  if (is_grayscale):
    return scipy.misc.imread(path, flatten = True).astype(np.float)
  else:
    return scipy.misc.imread(path).astype(np.float)

def merge_images(images, size):
  return inverse_transform(images)

def merge(images, size):
  h, w = images.shape[1], images.shape[2]
  img = np.zeros((h * size[0], w * size[1], 3))
  for idx, image in enumerate(images):
    i = idx % size[1]
    j = idx // size[1]
    img[j*h:j*h+h, i*w:i*w+w, :] = image
  return img

def imsave(images, size, path):
  return scipy.misc.imsave(path, merge(images, size))

def center_crop(x, crop_h, crop_w,
                resize_h=64, resize_w=64):
  if crop_w is None:
    crop_w = crop_h
  h, w = x.shape[:2]
  j = int(round((h - crop_h)/2.))
  i = int(round((w - crop_w)/2.))
  return scipy.misc.imresize(
      x[j:j+crop_h, i:i+crop_w], [resize_h, resize_w])

def transform(image, input_height, input_width, 
              resize_height=64, resize_width=64, is_crop=True):
  if is_crop:
    cropped_image = center_crop(
      image, input_height, input_width, 
      resize_height, resize_width)
  else:
    cropped_image = scipy.misc.imresize(image, [resize_height, resize_width])
  return np.array(cropped_image)/127.5 - 1.

def inverse_transform(images):
  return (images+1.)/2.

def to_json(output_path, *layers):
  with open(output_path, "w") as layer_f:
    lines = ""
    for w, b, bn in layers:
      layer_idx = w.name.split('/')[0].split('h')[1]

      B = b.eval()

      if "lin/" in w.name:
        W = w.eval()
        depth = W.shape[1]
      else:
        W = np.rollaxis(w.eval(), 2, 0)
        depth = W.shape[0]

      biases = {"sy": 1, "sx": 1, "depth": depth, "w": ['%.2f' % elem for elem in list(B)]}
      if bn != None:
        gamma = bn.gamma.eval()
        beta = bn.beta.eval()

        gamma = {"sy": 1, "sx": 1, "depth": depth, "w": ['%.2f' % elem for elem in list(gamma)]}
        beta = {"sy": 1, "sx": 1, "depth": depth, "w": ['%.2f' % elem for elem in list(beta)]}
      else:
        gamma = {"sy": 1, "sx": 1, "depth": 0, "w": []}
        beta = {"sy": 1, "sx": 1, "depth": 0, "w": []}

      if "lin/" in w.name:
        fs = []
        for w in W.T:
          fs.append({"sy": 1, "sx": 1, "depth": W.shape[0], "w": ['%.2f' % elem for elem in list(w)]})

        lines += """
          var layer_%s = {
            "layer_type": "fc", 
            "sy": 1, "sx": 1, 
            "out_sx": 1, "out_sy": 1,
            "stride": 1, "pad": 0,
            "out_depth": %s, "in_depth": %s,
            "biases": %s,
            "gamma": %s,
            "beta": %s,
            "filters": %s
          };""" % (layer_idx.split('_')[0], W.shape[1], W.shape[0], biases, gamma, beta, fs)
      else:
        fs = []
        for w_ in W:
          fs.append({"sy": 5, "sx": 5, "depth": W.shape[3], "w": ['%.2f' % elem for elem in list(w_.flatten())]})

        lines += """
          var layer_%s = {
            "layer_type": "deconv", 
            "sy": 5, "sx": 5,
            "out_sx": %s, "out_sy": %s,
            "stride": 2, "pad": 1,
            "out_depth": %s, "in_depth": %s,
            "biases": %s,
            "gamma": %s,
            "beta": %s,
            "filters": %s
          };""" % (layer_idx, 2**(int(layer_idx)+2), 2**(int(layer_idx)+2),
               W.shape[0], W.shape[3], biases, gamma, beta, fs)
    layer_f.write(" ".join(lines.replace("'","").split()))

def make_gif(images, fname, duration=2, true_image=False):
  import moviepy.editor as mpy

  def make_frame(t):
    try:
      x = images[int(len(images)/duration*t)]
    except:
      x = images[-1]

    if true_image:
      return x.astype(np.uint8)
    else:
      return ((x+1)/2*255).astype(np.uint8)

  clip = mpy.VideoClip(make_frame, duration=duration)
  clip.write_gif(fname, fps = len(images) / duration)

def samples_statistics(r, name, parameters,d_loss=None,g_loss=None):
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
    f.savefig(folder + '/autocorrelogram' + name + '.png', bbox_inches='tight')
    #plt.show()
    plt.close(f)
    if name=='real':
        data = {'mean':mean_spk_count,'std':std_spk_count,'acf':ac,'index':index,'prf_act':profile_activity,'samples':samples, 'prb_samples':numerical_prob, 'training_step':parameters.training_step}
    else:
        data = {'d_loss':d_loss,'g_loss':g_loss,'mean':mean_spk_count,'std':std_spk_count,'acf':ac,'index':index,'prf_act':profile_activity,'prb_samples':numerical_prob, 'training_step':parameters.training_step}
    
    np.savez(folder + '/autocorrelogram' + name + '.npz', **data)
    
    
def get_samples_autocorrelogram(sess, dcgan,name,parameters,d_loss,g_loss):
    folder = parameters.sample_dir
    num_samples = int(2**np.log2(parameters.num_samples))
    num_trials = int(num_samples/dcgan.batch_size)
    X = np.ndarray((num_samples,int(dcgan.output_height),1))    
    for ind_tr in range(num_trials):
        z_sample, = np.random.uniform(-1, 1, size=(dcgan.batch_size, dcgan.z_dim))
        X[np.arange(ind_tr*dcgan.batch_size,(ind_tr+1)*dcgan.batch_size),:,:] \
                = sess.run(dcgan.sampler, feed_dict={dcgan.z: z_sample})
    
    binarized_X = ops.binarize(X)
    binarized_X_reduced = binarized_X[:,:,0]
    
    #compute statistics
    samples_statistics(binarized_X_reduced,name,parameters,d_loss,g_loss)  
    
    
    #compute average activity
    f = plt.figure()
    aux = np.mean(binarized_X_reduced,axis=0)
    plt.plot(aux)
    f.savefig(folder + '/average_activity' + name + '.png', bbox_inches='tight')
    #plt.show()
    plt.close(f)
      
def get_samples(sess,dcgan,folder):    
    z_sample = np.random.uniform(-1, 1, size=(dcgan.batch_size, dcgan.z_dim))
    samples_plot = sess.run(dcgan.sampler, feed_dict={dcgan.z: z_sample})
    samples_plot = ops.binarize(samples_plot)
    fig,sbplt = plt.subplots(8,8)
    for ind_pl in range(np.shape(samples_plot)[0]):
        sbplt[int(np.floor(ind_pl/8))][ind_pl%8].plot(samples_plot[int(ind_pl),:])
        sbplt[int(np.floor(ind_pl/8))][ind_pl%8].axis('off')
    fig.savefig(folder + '/fake_samples_binarized.png',dpi=199, bbox_inches='tight')
    #plt.show()
    plt.close(fig)

def evaluate_training(folder,sbplt,ind):
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
        #autocorrelogram
        training_acf = training_data['acf']
        if np.max(training_acf)>0:
            training_acf = training_acf/np.max(training_acf)
        error_ac[ind_f] = np.sum(np.abs(real_acf-training_acf))
        #spikeCount mean
        spkC_mean[ind_f] = np.abs(real_spkC_mean-training_data['mean'])
        #spikeCount std
        spkC_std[ind_f] = np.abs(real_spkC_std-training_data['std'])
         
        #acticity profile
        training_spkC_prf_act = training_data['prf_act']
        error_spkC_prf_act[ind_f] = np.sum(np.abs(real_spkC_prf_act-training_spkC_prf_act))
        
        #mean probability of samples
        training_probs_mat[ind_f,:] = np.abs(real_probs_samples-training_data['prb_samples'])#np.abs(real_probs_samples-training_data['prb_samples'])/real_probs_samples
        error_probs_samples[ind_f] = np.sum(np.abs(real_probs_samples-training_data['prb_samples']))/len(real_probs_samples)

        
       
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
    
    #plot best fits
    plot_best_fit(best_ac_fit,'best_ac_fit')
    plt.close()
    plot_best_fit(best_mean_prob_fit,'best_mean_prob_fit')
    plt.close()
    
    indices = np.argsort(train_step)
    error_ac = np.array(error_ac)[indices]
    spkC_mean = np.array(spkC_mean)[indices]
    spkC_std = np.array(spkC_std)[indices]
    error_spkC_prf_act = np.array(error_spkC_prf_act)[indices]
    error_probs_samples = np.array(error_probs_samples)[indices]
    training_probs_mat  = np.array(training_probs_mat)[indices]
    
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
    sbplt[1][1].plot(error_probs_samples,label=str(ind))
    sbplt[1][1].set_title('samples mean probability error')
    sbplt[1][1].set_xlabel('training step (' + str(training_step) + ' batches)')
    sbplt[1][1].set_ylabel('absolute difference')
    os.chdir(mycwd)
    
    return(best_ac_fit,best_mean_prob_fit)
    
def compare_trainings(folder,title):
    print('--------------------------------------------------')
    print(folder)
    find_aux = folder.find('iteration')
    files = glob.glob(folder[0:find_aux]+'*')
    #figure for training error across epochs
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
    
    
def plot_best_fit(data,name):
    left  = 0.125  # the left side of the subplots of the figure
    right = 0.9    # the right side of the subplots of the figure
    bottom = 0.1   # the bottom of the subplots of the figure
    top = 0.9      # the top of the subplots of the figure
    wspace = 0.4   # the amount of width reserved for blank space between subplots
    hspace = 0.4   # the amount of height reserved for white space between subplots
    f,sbplt2 = plt.subplots(1,2,figsize=(8, 8),dpi=250)
    matplotlib.rcParams.update({'font.size': 8})
    plt.subplots_adjust(left=left, bottom=bottom, right=right, top=top, wspace=wspace, hspace=hspace)
    sbplt2[0].plot(data['acf'],label='real')
    sbplt2[0].plot(data['acf_fake'],'--',label='fake')
    sbplt2[0].set_title('AC function')
    sbplt2[0].set_xlabel('time')
    sbplt2[0].set_ylabel('proportion of spikes')
    sbplt2[1].plot(data['prf_act'],label='real')
    sbplt2[1].plot(data['prf_act_fake'],label='fake')
    sbplt2[1].set_title('time course average')
    sbplt2[1].set_xlabel('time')
    sbplt2[1].set_ylabel('average firing rate')
    sbplt2[1].set_ylim(0,0.12)
    plt.suptitle(name)
    plt.legend(shadow=True, fancybox=True)
    plt.show()
    f.savefig(name + '.png',dpi=300, bbox_inches='tight')
    f.savefig(name + '.svg',dpi=300, bbox_inches='tight') 
    np.savez(name + '.npz', **data)
    return(f)
    
def probability_data(parameters,data):
    num_classes = parameters.num_classes
    num_samples = int(np.shape(data)[0])
    num_bins = parameters.num_bins
    firing_rate = 0.1
    if parameters.dataset=='gaussian_fr':
        margin = 6 #num bins from the middle one that the response peaks will span (see line 389)
        std_resp = 4 #std of the gaussian defining the firing rates
        if parameters.classes_proportion=='equal':
            mat_prop_classes = np.ones((num_classes,))/num_classes
        elif parameters.classes_proportion=='7030':    
            mat_prop_classes = [0.7,0.3]
        else:
            raise ValueError("Unknown dataset '" + parameters.classes_proportion + "'")
             
            
        t = np.arange(num_bins)
        
        peaks1 = np.linspace(int(num_bins/2)-margin,int(num_bins/2)+margin,num_classes)
        
        
        probs_mat = np.zeros((num_classes,))
        for ind in range(num_classes):
            fr = firing_rate*np.exp(-(t-peaks1[ind])**2/std_resp**2)
            fr_noSpk = np.tile(fr,(num_samples,1))
            fr_Spk = np.tile(fr,(num_samples,1))
            fr_noSpk[data==1] = 0
            prob_noSpk_bins = np.prod(1-fr_noSpk,axis=1)
            fr_Spk[data==0] = 1
            prob_Spk_bins = np.prod(fr_Spk,axis=1)
            probs_mat[ind] = prob_noSpk_bins*prob_Spk_bins
            
        probData = np.sum(mat_prop_classes*probs_mat)
          
            
    elif parameters.dataset=='uniform_fr':
        fr_noSpk =np.zeros((num_samples,num_bins)) + firing_rate
        fr_Spk =np.zeros((num_samples,num_bins)) + firing_rate
        fr_noSpk[data==1] = 0
        prob_noSpk_bins = np.prod(1-fr_noSpk,axis=1)
        fr_Spk[data==0] = 1
        prob_Spk_bins = np.prod(fr_Spk,axis=1)
        probData = prob_noSpk_bins*prob_Spk_bins
        
    return(probData)    
        
        