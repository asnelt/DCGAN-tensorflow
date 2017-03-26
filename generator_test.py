import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
tf.reset_default_graph()

class batch_norm(object):
  def __init__(self, epsilon=1e-5, momentum = 0.9, name="batch_norm"):
    with tf.variable_scope(name):
      self.epsilon  = epsilon
      self.momentum = momentum
      self.name = name

  def __call__(self, x, train=True):
    return tf.contrib.layers.batch_norm(x,
                      decay=self.momentum, 
                      updates_collections=None,
                      epsilon=self.epsilon,
                      scale=True,
                      is_training=train,
                      scope=self.name)

def conv_cond_concat(x, y):
  """Concatenate conditioning vector on feature map axis."""
  x_shapes = x.get_shape()
  y_shapes = y.get_shape()
  return tf.concat_v2([x, y*tf.ones([x_shapes[0], x_shapes[1], y_shapes[2]])], 2)

def deconv1d(input_, output_shape,
       k_h=5, d_h=2, stddev=0.02,
       name="deconv2d"):
  with tf.variable_scope(name):
    # filter : [height, width, output_channels, in_channels]
    print('kernel size')
    print([k_h, 1, output_shape[-1], input_.get_shape()[-1]])
    w = tf.get_variable('w', [k_h, 1, output_shape[-1], input_.get_shape()[-1]],
              initializer=tf.random_normal_initializer(stddev=stddev))
    
    aux_input = tf.expand_dims(input_,axis=2)
    deconv = tf.nn.conv2d_transpose(aux_input, w, output_shape=output_shape,strides=[1, d_h, 1, 1])


    biases = tf.get_variable('biases', [output_shape[-1]], initializer=tf.constant_initializer(0.0))
    deconv = tf.reshape(tf.nn.bias_add(deconv, biases), deconv.get_shape())
   
    return deconv[:,:,0,:]


def nn_resize(input_, output_shape,
       k_h=5, d_h=1, stddev=0.02,
       name="deconv2d"):
  with tf.variable_scope(name):
    # filter : [height, width, output_channels, in_channels]
    w = tf.get_variable('w', [k_h, input_.get_shape()[-1], output_shape[-1]],
              initializer=tf.truncated_normal_initializer(stddev=stddev))   
    print('ooooooooooooooooooooooooooooooooooooo')
    print('kernel size')
    print(w.get_shape())
    aux_input = tf.expand_dims(input_,axis=2)
    print('input size')
    print(input_.get_shape())
    print('aux input size')
    print(aux_input.get_shape())
    resized_image = tf.image.resize_images(aux_input, output_shape[1:3], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    resized_image = resized_image[:,:,0,:]
    print('resized image shape')
    print(resized_image.get_shape())
    deconv = tf.nn.conv1d(resized_image, w, stride=d_h, padding='SAME')
    print('deconv size')
    print(deconv.get_shape())
    print('ooooooooooooooooooooooooooooooooooooo')
    biases = tf.get_variable('biases', [output_shape[-1]], initializer=tf.constant_initializer(0.0))
    deconv = tf.reshape(tf.nn.bias_add(deconv, biases), deconv.get_shape())
   
    return deconv

     
def lrelu(x, leak=0.2, name="lrelu"):
  return tf.maximum(x, leak*x)

def linear(input_, output_size, scope=None, stddev=0.02, bias_start=0.0, with_w=False):
  shape = input_.get_shape().as_list()

  with tf.variable_scope(scope or "Linear"):
    matrix = tf.get_variable("Matrix", [shape[1], output_size], tf.float32,
                 tf.random_normal_initializer(stddev=stddev))
    bias = tf.get_variable("bias", [output_size],
      initializer=tf.constant_initializer(bias_start))
    if with_w:
      return tf.matmul(input_, matrix) + bias, matrix, bias
    else:
      return tf.matmul(input_, matrix) + bias



batch_size = 64
z = tf.Variable(tf.random_uniform([batch_size, 100]), name="var")

aux = np.random.choice(2, batch_size)
y = np.zeros((batch_size, 2))
y[np.arange(batch_size), aux] = 1
          
y_one_hot = tf.to_float(tf.constant(y))
          
output_height = 28


y_dim = 2
gf_dim = 1
gfc_dim = 1024
output_depth = 1
g_bn0 = batch_norm(name='g_bn0')
g_bn1 = batch_norm(name='g_bn1')
g_bn2 = batch_norm(name='g_bn2')
s_h = output_height
s_h2, s_h4 = int(s_h/2), int(s_h/4)
  
print('z')  
print(z.get_shape())
print('y')  
print(y_one_hot.get_shape())
#labels
yb = tf.reshape(y_one_hot, [batch_size, 1, y_dim])
#z is concatenated with the labels 
z = tf.concat_v2([z, y_one_hot], 1)

#first layer (linear + batch norm. + relu)
h0_linear = linear(z, gfc_dim, 'g_h0_lin')
h0_activity = tf.nn.relu((h0_linear))
print('h0.get_shape()')
print(h0_activity.get_shape())
h0 = tf.concat_v2([h0_activity, y_one_hot], 1)
      
#first layer (linear + batch norm. + relu)
h1_linear = linear(h0, gf_dim*s_h4, 'g_h1_lin')
h1_norm = g_bn1(h1_linear)
h1_activity = tf.nn.relu(h1_norm)
print('h1_aux.get_shape()')
print(h1_activity.get_shape())
h1 = tf.reshape(h1_activity, [batch_size, s_h4, gf_dim])
print('h1.get_shape() reshaped')
print(h1.get_shape())
h1 = conv_cond_concat(h1, yb)

#third layer (deconv2d + batch norm. + relu)
h2_deconv = nn_resize(h1,[batch_size, s_h2, 1, gf_dim ], name='g_h2')
h2_norm = g_bn2(h2_deconv)
h2_activity = tf.nn.relu(h2_norm)
print('h2.get_shape()')
print(h2_activity.get_shape())
h2 = conv_cond_concat(h2_activity, yb)

#third layer (deconv2d + signmoid, no batch norm.)
h3_deconv = nn_resize(h2, [batch_size, s_h, 1, output_depth], name='g_h3')
h3_activity = tf.nn.sigmoid(h3_deconv)
print('h3.get_shape()')
print(h3_activity.get_shape())


sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())
f,sbplt = plt.subplots(4,3)  

sbplt[0][0].plot(np.transpose(z.eval()))
sbplt[0][0].set_title('z')
sbplt[0][1].plot(np.transpose(h0_linear.eval()))
sbplt[0][1].set_title('h0 linear')
sbplt[0][2].plot(np.transpose(h0_activity.eval()))
sbplt[0][2].set_title('h0 activity')

sbplt[1][0].plot(np.transpose(h1_linear.eval()))
sbplt[1][0].set_title('h1 linear')
sbplt[1][1].plot(np.transpose(h1_norm.eval()))
sbplt[1][1].set_title('h1 norm')
sbplt[1][2].plot(np.transpose(h1_activity.eval()))
sbplt[1][2].set_title('h1 activity')

samples_plot = h2_deconv[:,:,0]
sbplt[2][0].plot(np.transpose(samples_plot.eval()))
sbplt[2][0].set_title('h2 deconv')
samples_plot = h2_norm[:,:,0]
sbplt[2][1].plot(np.transpose(samples_plot.eval()))
sbplt[2][1].set_title('h2 norm')
samples_plot = h2_activity[:,:,0]
sbplt[2][2].plot(np.transpose(samples_plot.eval()))
sbplt[2][2].set_title('h2 activity')

samples_plot = h3_deconv[:,:,0]
sbplt[3][0].plot(np.transpose(samples_plot.eval()))
sbplt[3][0].set_title('h3 deconv')
samples_plot = h3_activity[:,:,0]
sbplt[3][1].plot(np.transpose(samples_plot.eval()))
sbplt[3][1].set_title('h3 norm')
plt.show()
