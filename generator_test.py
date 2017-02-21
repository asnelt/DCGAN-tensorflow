import tensorflow as tf
import ops 
import numpy as np
import matplotlib.pyplot as plt
z = tf.Variable(tf.random_uniform([64, 100]), name="var")
batch_size = 64
aux = np.random.choice(2, batch_size)
y = np.zeros((batch_size, 2))
y[np.arange(batch_size), aux] = 1
          
y_one_hot = tf.to_float(tf.constant(y))
          
output_height = 28
output_width = 1

y_dim = 2
gf_dim = 64
gfc_dim = 1024
output_depth = 1
g_bn0 = ops.batch_norm(name='g_bn0')
g_bn1 = ops.batch_norm(name='g_bn1')
g_bn2 = ops.batch_norm(name='g_bn2')
s_h, s_w = output_height, output_width
s_h2, s_h4 = int(s_h/2), int(s_h/4)
s_w2, s_w4 = int(s_w), int(s_w)
      
print('z')  
print(z.get_shape())
print('y')  
print(y_one_hot.get_shape())
#labels
yb = tf.reshape(y_one_hot, [batch_size, 1, 1, y_dim])
#z is concatenated with the labels 
#z = tf.concat_v2([z, y_one_hot], 1)

#first layer (linear + batch norm. + relu)
h0 = tf.nn.relu(g_bn0(ops.linear(z, gfc_dim, 'g_h0_lin')))
print('h0.get_shape()')
print(h0.get_shape())
#h0 = tf.concat_v2([h0, y_one_hot], 1)
      
#first layer (linear + batch norm. + relu)
h1 = tf.nn.relu(g_bn1(ops.linear(h0, gf_dim*2*s_h4*s_w4, 'g_h1_lin')))
print('h1.get_shape()')
print(h1.get_shape())
h1 = tf.reshape(h1, [batch_size, s_h4, s_w4, gf_dim * 2])
print('h1.get_shape() reshaped')
print(h1.get_shape())
#h1 = ops.conv_cond_concat(h1, yb)

#third layer (deconv2d + batch norm. + relu)
h2 = tf.nn.relu(g_bn2(ops.deconv2d(h1,[batch_size, s_h2, s_w2, gf_dim * 2], name='g_h2')))
print('h2.get_shape()')
print(h2.get_shape())
#h2 = ops.conv_cond_concat(h2, yb)

#third layer (deconv2d + signmoid, no batch norm.)
h3 = tf.nn.sigmoid(ops.deconv2d(h2, [batch_size, s_h, s_w, output_depth], name='g_h3'))
print('h3.get_shape()')
print(h3.get_shape())


sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())
fig = plt.figure()
samples_plot = h3[:,:,0,0]
print(samples_plot.get_shape())
plt.plot(np.transpose(samples_plot.eval()))
plt.show()
