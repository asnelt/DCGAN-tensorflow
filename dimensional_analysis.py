#!/env/python
import os

from subprocess import call
import numpy as np
import tensorflow as tf

import utils
from model import DCGAN

folder_name = "samples_dataset_gaussian_fr_num_classes_1_propClasses_equal_num_samples_2048_num_bins_28_ref_period_-1_firing_rate_0.5_iteration_18"

def generate_fake():
    call(["python3.5", "main.py", "--dataset", "gaussian_fr", "--num_classes=1", "--classes_proportion", "equal", "--ref_period=-1", "--firing_rate=0.5", "--epoch=50", "--num_samples=2048", "--iteration", "18", "--training_stage=801"])
    temp = np.load(os.path.join(folder_name, "fake_samples.npz"))
    return temp['fake_samples'].astype(np.bool)

def generate_real():
    return(utils.get_more_real_samples_gaussian_no_refrPer(num_samples=2048).astype(np.bool))

def generate_random(n_samples=2048, output_height=28):
    run_config = tf.ConfigProto()
    run_config.gpu_options.allow_growth=True
    samples = np.empty((n_samples, output_height));
    with tf.Session(config=run_config) as sess:
        dcgan = DCGAN(
            sess,
            dataset_name="gaussian_fr",
            checkpoint_dir="checkpoint_random",
            sample_dir="sample_random")
        tf.global_variables_initializer().run()
        n_passes = int(n_samples/dcgan.batch_size)
        for n in range(n_passes):
            z_sample = np.random.uniform(-1, 1, size=(dcgan.batch_size, dcgan.z_dim))
            samples[n*dcgan.batch_size:(n+1)*dcgan.batch_size] = np.squeeze(sess.run(dcgan.sampler, feed_dict={dcgan.z: z_sample}))

    return samples

def frequency_table(a):
    num_samples = a.shape[0]
    b = np.ascontiguousarray(a).view(np.dtype((np.void, a.dtype.itemsize * a.shape[1])))
    unique_b, idx, counts = np.unique(b, return_index=True, return_counts=True)
    unique_a = a[idx]
    frequencies = counts/num_samples
    return dict([(key.tostring(), value) for (key, value) in zip(unique_a, frequencies)])
    
def jensen_shannon_distance(p, q):
    for key in q.keys() - p.keys():
        p[key] = 0
    for key in p.keys() - q.keys():
        q[key] = 0
    
    # note: this can obviously be optimised by getting rid of the for loop
    divergence = 0
    for key in p.keys():
        px = p[key]
        qx = q[key]
        mx = (px+qx)/2
        divergence += (kl_term(px,mx)+kl_term(qx,mx))/2
    
    return np.sqrt(divergence)
        
def kl_term(px, qx):
    if px>0:
        return px * np.log2(px/qx)
    else:
        return 0
    
def full_distance_matrix(n_real, n_fake):
    real = [frequency_table(generate_real()) for each in range(n_real)]
    fake = [frequency_table(generate_fake()) for each in range(n_fake)]
    print("Done generating stuff")
    return pairwise_distances(real + fake)

    
def pairwise_distances(ps):
    n_points = len(ps)
    distances = np.zeros(int(n_points*(n_points-1)/2))
    idx = 0
    for i in range(n_points):
        for j in range(i+1, n_points):
            distances[idx] = jensen_shannon_distance(ps[i], ps[j])
            idx+=1
    return distances
    

