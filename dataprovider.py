from utils import spk_autocorrelogram
import numpy as np
import matplotlib.pyplot as plt


class DataProvider(object):

    def __init__(self, parameters=None):
        self.data, self.labels = generate_spike_trains(parameters)
        

    def visualize(self):
        pass

def generate_spike_trains(parameters):
    #create artificial data
    num_classes = parameters.num_classes
    num_samples = parameters.num_samples
    num_bins = parameters.num_bins
    refr_per = parameters.ref_period
    firing_rate = 0.1
    
    
    if parameters.dataset=='gaussian_fr':
        noise = 0.01*firing_rate
        margin = 6 #num bins from the middle one that the response peaks will span (see line 389)
        std_resp = 4 #std of the gaussian defining the firing rates
        t = np.arange(num_bins)
        
        peaks1 = np.linspace(int(num_bins/2)-margin,int(num_bins/2)+margin,num_classes)
        peaks1 = np.tile(peaks1, (1,int(np.round(num_samples/num_classes)))).transpose()
        stims = np.unique(peaks1)
        X =np.zeros((peaks1.size,num_bins,1))
        y =np.zeros((peaks1.size,num_classes))
        if parameters.visualize_data:
            if num_classes>1:
                fig,sbplt = plt.subplots(1,num_classes)
            else:
                fig = plt.figure()
        counter = np.zeros((1,num_classes))
        for ind in range(peaks1.size):
            stim = np.nonzero(stims==peaks1[ind])
            stim = int(stim[0])
            fr = firing_rate*np.exp(-(t-peaks1[ind])**2/std_resp**2) + np.random.normal(0,noise,(1,num_bins))
            fr[fr<0] = 0
            r = np.random.poisson(fr)
            r[r>0] = 1
                
            X[ind,:,0] = r
            y[ind,stim] = 1
            counter[0][stim] = counter[0][stim] + 1
            if parameters.visualize_data:
                if counter[0][stim]==1:
                    if num_classes>1:
                        sbplt[stim].plot(fr[0],linewidth=4.0)
                    else:
                        plt.plot(fr[0],linewidth=4.0)
                if counter[0][stim]<=10:
                    if num_classes>1:
                        sbplt[stim].plot(r[0])
                        sbplt[stim].axis('off')
                    else:
                        plt.plot(r[0])
                        plt.axis('off')
                    
                    
    elif parameters.dataset=='uniform_fr':
        X =np.random.poisson(np.zeros((num_samples,num_bins,1)) + firing_rate)
        X[X>0] = 1
        y = np.ones((num_samples, 1))
        counter = np.zeros((1,num_classes))
        counter[0] = num_samples   
        if parameters.visualize_data:
            fig = plt.figure()
            samples_plot = X[0:9,:,0]
            plt.plot(np.transpose(samples_plot))
    else:
        raise ValueError("Unknown dataset '" + parameters.dataset + "'")
        
    show_real_samples = True#False
    if parameters.visualize_data:
        plt.show()

        fig.savefig(parameters.sample_dir + '/real_samples.png',dpi=199, bbox_inches='tight')
        plt.close(fig)

        f,sbplt = plt.subplots(1,2)
        sbplt[0].plot(counter[0])
        sbplt[1].imshow(y[1:100,:])
    
        if show_real_samples:
            plt.show()
    
        f.savefig(parameters.sample_dir + '/stim_tags.png', bbox_inches='tight')
        plt.close(f)
    
    #impose refractory period
    if refr_per>=0:
        X = refractory_period(refr_per,X)  
        
    #get autocorrelogram
    X_reduced = X[:,:,0]
    spk_autocorrelogram(X_reduced,'real', parameters.sample_dir)
    
    #compute average activity
    f = plt.figure()
    aux = np.mean(X_reduced,axis=0)
    plt.plot(aux)
    f.savefig(parameters.sample_dir + '/average_activity_real.png', bbox_inches='tight')
    plt.show()
    plt.close(f)
    
    X = X-np.min(X)
    # Shuffle samples
    seed = 547
    np.random.seed(seed)
    np.random.shuffle(X)
    np.random.seed(seed)
    np.random.shuffle(y)
  
    return X/X.max(), y

def refractory_period(refr_per, r):
    print('imposing refractory period of ' + str(refr_per))
    r = r[:,:,0]
    margin = np.zeros((r.shape[0],refr_per))
    r = np.hstack((margin,np.hstack((r,margin))))
    r_flat = r.flatten()
    spiketimes = np.nonzero(r_flat>0)
    spiketimes = np.sort(spiketimes)
    isis = np.diff(spiketimes)
    too_close = np.nonzero(isis<=refr_per)
    while len(too_close[0])>0:
        spiketimes = np.delete(spiketimes,too_close[0][0]+1)
        isis = np.diff(spiketimes)
        too_close = np.nonzero(isis<=refr_per)
        
    r_flat = np.zeros(r_flat.shape)
    r_flat[spiketimes] = 1
    r = np.reshape(r_flat,r.shape)
    r = r[:,refr_per:-refr_per]
    r = np.expand_dims(r,2)
    return r
