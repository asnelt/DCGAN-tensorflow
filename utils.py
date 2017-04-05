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

def spk_autocorrelogram(r, name):
    print('plot autocorrelogram')
    #first we get the average and std of the spike-count 
    mean_spk_count = np.mean(np.sum(r,axis=1))
    std_spk_count = np.std(np.sum(r,axis=1))
    lag = 10
    margin = np.zeros((r.shape[0],lag))
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
    f.savefig('samples/autocorrelogram' + name + '.png', bbox_inches='tight')
    plt.show()
    plt.close(f)
    data = {'mean':mean_spk_count,'std':std_spk_count,'acf':ac,'index':index}
    np.savez('samples/autocorrelogram' + name + '.npz', **data)
    
    
def get_samples_autocorrelogram(sess, dcgan,name):
    num_samples = int(2**15)
    num_trials = int(num_samples/dcgan.batch_size)
    X = np.ndarray((num_samples,int(dcgan.output_height),1))    
    for ind_tr in range(num_trials):
        z_sample = np.random.uniform(-1, 1, size=(dcgan.batch_size, dcgan.z_dim))
        X[np.arange(ind_tr*dcgan.batch_size,(ind_tr+1)*dcgan.batch_size),:,:] \
                = sess.run(dcgan.sampler, feed_dict={dcgan.z: z_sample})
    
    binarized_X = ops.binarize(X)
    binarized_X_reduced = binarized_X[:,:,0]
    
    spk_autocorrelogram(binarized_X_reduced,name)  
    
    
def get_samples(sess,dcgan):    
    z_sample = np.random.uniform(-1, 1, size=(dcgan.batch_size, dcgan.z_dim))
    samples_plot = sess.run(dcgan.sampler, feed_dict={dcgan.z: z_sample})
    samples_plot = ops.binarize(samples_plot)
    fig,sbplt = plt.subplots(8,8)
    for ind_pl in range(np.shape(samples_plot)[0]):
        sbplt[int(np.floor(ind_pl/8))][ind_pl%8].plot(samples_plot[int(ind_pl),:])
        sbplt[int(np.floor(ind_pl/8))][ind_pl%8].axis('off')
        #fig.suptitle("[Sample] d_loss: %.8f, g_loss: %.8f" % (d_loss, g_loss))
    fig.savefig('samples/fake_samples_binarized.png',dpi=199, bbox_inches='tight')
    plt.show()
    plt.close(fig)
