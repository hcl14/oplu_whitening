
from __future__ import print_function
import numpy as np
import random
#import h5py #matlab v7.3 files

import sys

import os

#cpu version
#os.environ['CUDA_VISIBLE_DEVICES'] = '-1'


import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# create a file handler
handler = logging.FileHandler('oplu.log',mode='w')
handler.setLevel(logging.DEBUG)

# create a logging format
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)

# add the handlers to the logger
logger.addHandler(handler)




np.random.seed(1001)

# Import MNIST data


augmented_datasets_available = 100

    
#mat_contents =  h5py.File('matlab_mnist/affNIST.mat')

#X2 = np.array(mat_contents['all_images_val'],dtype=np.float32) # (10000, 1600) FLOAT32
#T2 = np.array(mat_contents['all_labels_val'],dtype=np.float32) # (10000, 10) 

# Import MNIST data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

X2 = mnist.test.images
T2 = mnist.test.labels

# clear space
mat_contents = None


#X1,T1 = None,None

X1 = mnist.train.images
T1 = mnist.train.labels


def load_epoch(epoch):
    
    '''
    
    if epoch>=augmented_datasets_available:
        epoch=epoch%augmented_datasets_available
  
    mat_contents = h5py.File('augdata/augmented'+str(epoch)+'.h5')
    
    #load new data
    
    
    X2 = np.array(mat_contents['all_images_train'],dtype=np.float32) # (1600000,1600) #already in Float32 from data augmenter!
    
    global X1,T1
    
    X1 = np.random.normal(0,1,X2.shape)
    
    
    T1 = np.array(mat_contents['all_labels_train'],dtype=np.float32) # (1600000, 10)
    '''
    
    # clear space
    mat_contents = None
    
    

load_epoch(0)


def get_batch_train(idx):
    batch_x = X1[idx, :]
    batch_y = T1[idx, :]
    
    return batch_x, batch_y

print('Shapes:')
print('training:')
print(X1.shape)
print(T1.shape)
print('testing:')
print(X2.shape)
print(T2.shape)

print('---------')













import tensorflow as tf
from tensorflow.python.client import timeline


# not needed - it's the shape of the image
#X1 = tf.contrib.layers.batch_norm(X1, fused=True, data_format='NCHW') #NCHW is the optimal format to use when training on NVIDIA GPUs using cuDNN.
#X2 = tf.contrib.layers.batch_norm(X2, fused=True, data_format='NCHW') #NCHW is the optimal format to use when training on NVIDIA GPUs using cuDNN.


#from adamax import *

# Parameters
learning_rate = 0.005
learning_rate_decay = 0.99

momentum = 0.3

training_epochs = 100
batch_size = 1
display_step = 1

# Network Parameters
n_hidden_1 = 128 # 1st layer number of neurons
n_hidden_2 = 128 # 2nd layer number of neurons
n_hidden_3 = 128 # 3rd layer
n_hidden_4 = 128
n_hidden_5 = 128
n_hidden_6 = 128
n_hidden_7 = 128
n_hidden_8 = 128
n_hidden_9 = 128
n_hidden_10 = 128
n_hidden_11 = 128
n_hidden_12 = 128
n_hidden_13 = 128
n_hidden_14 = 128
n_hidden_15 = 128
n_hidden_16 = 128
n_hidden_17 = 128
n_hidden_18 = 128
n_hidden_19 = 128
n_hidden_20 = 128

n_input = X1.shape[1]
n_classes = T1.shape[1]







# tf Graph input
X = tf.placeholder("float", [None, n_input])
Y = tf.placeholder("float", [None, n_classes])
tf_learning_rate = tf.placeholder(tf.float32, shape=[])  # need to be a placeholder if we want to apply new learning rate on each step

b = np.array([])

# function  that initializes orthogonal matrix
def ort_initializer(shape, dtype=tf.float32):
    
      a=shape[0]
      b=shape[1]
      
      if a>b:
          
          A = np.random.normal(0,1,(a,a))
          q, _ = np.linalg.qr(A)  # q -orthonormal
          q = q[:,:b] # cut q
          
      else:
          A = np.random.normal(0,1,(b,b))
          q, _ = np.linalg.qr(A)  # q -orthonormal
          q = q[:a,:] # cut q
          
      return tf.constant(q,dtype=dtype)


# compute ||W*W^T - I||L2 for a matrix
def ort_discrepancy(matrix):
    
    wwt = tf.matmul(matrix, matrix, transpose_a=True)
    
    rows = tf.shape(wwt)[0]
    cols = tf.shape(wwt)[1]
    
    return tf.norm((wwt - tf.eye(rows,cols)),ord='euclidean') #/tf.to_float(rows*cols)


def tf_gram_schmidt(vectors):
    # add batch dimension for matmul
    basis = tf.expand_dims(vectors[0,:]/tf.norm(vectors[0,:]),0)
    for i in range(1,vectors.get_shape()[0].value):
        v = vectors[i,:]
        # add batch dimension for matmul
        v = tf.expand_dims(v,0) 
        w = v - tf.matmul(tf.matmul(v, basis, transpose_b=True), basis)
         # I assume that my matrix is close to orthogonal
        basis = tf.concat([basis, w/tf.norm(w)],axis=0)
    return basis
# Store layers weight & bias

weights = {
    'h1': tf.Variable(ort_initializer([n_input, n_hidden_1])),
    'h2': tf.Variable(ort_initializer([n_hidden_1, n_hidden_2])),
    'h3': tf.Variable(ort_initializer([n_hidden_2, n_hidden_3])),
    'h4': tf.Variable(ort_initializer([n_hidden_3, n_hidden_4])),
    'h5': tf.Variable(ort_initializer([n_hidden_4, n_hidden_5])),
    'h6': tf.Variable(ort_initializer([n_hidden_5, n_hidden_6])),
    'h7': tf.Variable(ort_initializer([n_hidden_6, n_hidden_7])),
    'h8': tf.Variable(ort_initializer([n_hidden_7, n_hidden_8])),
    'h9': tf.Variable(ort_initializer([n_hidden_8, n_hidden_9])),
    'h10': tf.Variable(ort_initializer([n_hidden_9, n_hidden_10])),
    'h11': tf.Variable(ort_initializer([n_hidden_10, n_hidden_11])),
    'h12': tf.Variable(ort_initializer([n_hidden_11, n_hidden_12])),
    'h13': tf.Variable(ort_initializer([n_hidden_12, n_hidden_13])),
    'h14': tf.Variable(ort_initializer([n_hidden_13, n_hidden_14])),
    'h15': tf.Variable(ort_initializer([n_hidden_14, n_hidden_15])),
    'h16': tf.Variable(ort_initializer([n_hidden_15, n_hidden_16])),
    'h17': tf.Variable(ort_initializer([n_hidden_16, n_hidden_17])),
    'h18': tf.Variable(ort_initializer([n_hidden_17, n_hidden_18])),
    'h19': tf.Variable(ort_initializer([n_hidden_18, n_hidden_19])),
    'h20': tf.Variable(ort_initializer([n_hidden_19, n_hidden_20])),
    
    'out': tf.Variable(ort_initializer([n_hidden_20, n_classes]))
}




grads = {
    'h1': tf.Variable(tf.zeros([n_input, n_hidden_1])),
    'h2': tf.Variable(tf.zeros([n_hidden_1, n_hidden_2])),
    'h3': tf.Variable(tf.zeros([n_hidden_2, n_hidden_3])),
    'h4': tf.Variable(tf.zeros([n_hidden_3, n_hidden_4])),
    'h5': tf.Variable(tf.zeros([n_hidden_4, n_hidden_5])),
    'h6': tf.Variable(tf.zeros([n_hidden_5, n_hidden_6])),
    'h7': tf.Variable(tf.zeros([n_hidden_6, n_hidden_7])),
    'h8': tf.Variable(tf.zeros([n_hidden_7, n_hidden_8])),
    'h9': tf.Variable(tf.zeros([n_hidden_8, n_hidden_9])),
    'h10': tf.Variable(tf.zeros([n_hidden_9, n_hidden_10])),
    'h11': tf.Variable(tf.zeros([n_hidden_10, n_hidden_11])),
    'h12': tf.Variable(tf.zeros([n_hidden_11, n_hidden_12])),
    'h13': tf.Variable(tf.zeros([n_hidden_12, n_hidden_13])),
    'h14': tf.Variable(tf.zeros([n_hidden_13, n_hidden_14])),
    'h15': tf.Variable(tf.zeros([n_hidden_14, n_hidden_15])),
    'h16': tf.Variable(tf.zeros([n_hidden_15, n_hidden_16])),
    'h17': tf.Variable(tf.zeros([n_hidden_16, n_hidden_17])),
    'h18': tf.Variable(tf.zeros([n_hidden_17, n_hidden_18])),
    'h19': tf.Variable(tf.zeros([n_hidden_18, n_hidden_19])),
    'h20': tf.Variable(tf.zeros([n_hidden_19, n_hidden_20])),
    'out': tf.Variable(tf.zeros([n_hidden_20, n_classes]))
}



def combine_even_odd(even, odd):
    res = tf.reshape(tf.concat([even[..., tf.newaxis], odd[..., tf.newaxis]], axis=-1), [tf.shape(even)[0], -1])
    return res

# needed to register custom activation derivative. 
# https://stackoverflow.com/questions/43256517/how-to-register-a-custom-gradient-for-a-operation-composed-of-tf-operations
@tf.RegisterGradient('my_derivative')
def my_derivative(op, grad): 
    
    x = op.inputs[0] # values of the layer: n_batches*layer_size
    
    
    even = x[:,::2] #slicing into odd and even parts on the batch
    odd = x[:,1::2]
    even_grad = grad[:,::2] # slicing gradients
    odd_grad = grad[:,1::2]
    compare = tf.cast(even<odd,dtype=tf.float32)
    compare_not = tf.cast(even>=odd, dtype=tf.float32)
    
    
    
    # OPLU
    grad_even_new = odd_grad * compare + even_grad * compare_not
    grad_odd_new = odd_grad * compare_not + even_grad * compare
    # inteweave gradients back
    grad_new = combine_even_odd(grad_even_new,grad_odd_new)
    
    return grad_new


def oplu_derivative(x, grad, name="oplu grad"): 
    
    #x = op.inputs[0] # values of the layer: n_batches*layer_size
    
    #grad is a vector with the same size as x (a matrix n_batches*layer_size - more precisely)
    #so it seem to be deltas
    
    
    #grad = apply_my_custom_activation_derivative(x, grad, ... )
    
    # example: manual relu
    #compare = tf.cast((x>0),dtype=tf.float32)
    #compare_not = tf.cast((x<=0),dtype=tf.float32)
    #relu_deriv = 1.0*compare
    #grad_new = grad*relu_deriv
    
    #starting in the same way forward oplu works
    even = x[:,::2] #slicing into odd and even parts on the batch
    odd = x[:,1::2]
    even_grad = grad[:,::2] # slicing gradients
    odd_grad = grad[:,1::2]
    compare = tf.cast(even<odd,dtype=tf.float32)
    compare_not = tf.cast(even>=odd, dtype=tf.float32)
    
    
    
    # OPLU
    grad_even_new = odd_grad * compare + even_grad * compare_not
    grad_odd_new = odd_grad * compare_not + even_grad * compare
    # inteweave gradients back
    grad_new = combine_even_odd(grad_even_new,grad_odd_new)
    
    
    #grad_new = 0.0001*tf.ones(tf.shape(grad), dtype=tf.float32)
    
    # we can check the shape this way
    #grad_new = tf.Print(grad_new, [name, tf.norm(grad_new)], message = 'op grad norm')
    
    
    return grad_new



def OPLU(x, name="OPLU"):
    # x is n_batches*layer_size
    
    #y = apply_my_custom_activation(x,...)
    
    # example: manual relu
    #compare = tf.cast((x>0),dtype=tf.float32)
    #compare_not = tf.cast((x<=0),dtype=tf.float32)    
    #y = compare*x
    
    even = x[:,::2] #slicing into odd and even parts on the batch
    odd = x[:,1::2]
    # OPLU
    
    compare = tf.cast((even<odd),dtype=tf.float32)  #if compare is 1, elements are permuted
    compare_not = tf.cast((even>=odd),dtype=tf.float32) #if compare_not is 1 instead, elements are not permuted
    
    even_new = odd * compare + even * compare_not
    odd_new = odd * compare_not + even * compare
    
    
    # combine into one
    y = combine_even_odd(even_new,odd_new)
        
    return y



# We want to mask OPLU from gradient computation on the backward pass, this is done by making it appear as Identity.
# OPLU gradient is then applied using overrided Identity op
# See there how to mask some commands from gradient computation: https://stackoverflow.com/questions/36456436/how-can-i-define-only-the-gradient-for-a-tensorflow-subgraph/36480182#36480182

'''
Suppose you want group of ops that behave as f(x) in forward mode, but as g(x) in the backward mode. You implement it as

t = g(x)
y = t + tf.stop_gradient(f(x) - t)

So in your case your g(x) could be an identity op, with a custom op gradient using gradient_override_map
'''


def my_activation(x, name='my_activation'): 
    
    
    
    # Applying custom gradient
    
    # https://stackoverflow.com/questions/43256517/how-to-register-a-custom-gradient-for-a-operation-composed-of-tf-operations
    
    # hack to apply custom activation derivative 
    
    g = tf.get_default_graph()

    with g.gradient_override_map({'Identity': 'my_derivative'}):
        y = tf.identity(x, name='my_activ')  # !!! we need to pass correct inputs (x) into gradient override
    #    return y
    
    
    # masking OPLU from gradient computation - this code should go after 
    t = tf.identity(y)
    y = t + tf.stop_gradient(OPLU(y) - t)  
    
    
    
    return y



# my custom function to modify gradient
def my_weight_gradient_modification(grad_var_tuple):
    
    
    grad = grad_var_tuple[0]
    variable = grad_var_tuple[1]
    
    #example:
    #grad_new = 0.8*grad
    
    # projection to the tangent supspace
    # costy! slows down computation
    grad_new = tf.matmul(tf.matmul(grad,variable,transpose_b=True) - tf.matmul(variable,grad,transpose_b=True), variable)
    
    #grad_new = tf.Print(grad, [tf.norm(grad)])
    
    '''
    if variable.get_shape()[0].value > variable.get_shape()[1].value:
        variable = tf.transpose(tf_gram_schmidt(tf.transpose(variable)))
    else:
        variable = tf_gram_schmidt(variable)
    '''
    
    return grad_new, variable



# Create model
#def multilayer_perceptron(x):



# bias replacements
x = X

ones = tf.ones([tf.shape(x)[0],1],tf.float32)

################### Adding biases 

# Let's change first pixel instead. We do not lose anything, as actual digit is smaller (28x28) in the middle of the 40x40 square
x = tf.concat([x[:,1:],ones], 1)

################### Compute Forward prop



def forwardpropagate_layer(layer_input, weights):
    layer_z = tf.matmul(layer_input, weights)
    layer_a = OPLU(layer_z)
    return layer_z, layer_a

def forwardpropagate_building_block(layer_input, later_pred_input, weights):    
    layer_z, layer_a = forwardpropagate_layer(0.5*(layer_input + later_pred_input),weights)    
    return layer_z, layer_a
    
    


layer_1_z, layer_1_a = forwardpropagate_layer(x, weights['h1'])
layer_2_z, layer_2_a = forwardpropagate_layer(layer_1_a, weights['h2'])


layer_3_z, layer_3_a = forwardpropagate_building_block(layer_2_a,layer_1_a,weights['h3'])
layer_4_z, layer_4_a = forwardpropagate_building_block(layer_3_a,layer_2_a,weights['h4'])
layer_5_z, layer_5_a = forwardpropagate_building_block(layer_4_a,layer_3_a,weights['h5'])
layer_6_z, layer_6_a = forwardpropagate_building_block(layer_5_a,layer_4_a,weights['h6'])
layer_7_z, layer_7_a = forwardpropagate_building_block(layer_6_a,layer_5_a,weights['h7'])
layer_8_z, layer_8_a = forwardpropagate_building_block(layer_7_a,layer_6_a,weights['h8'])
layer_9_z, layer_9_a = forwardpropagate_building_block(layer_8_a,layer_7_a,weights['h9'])
layer_10_z, layer_10_a = forwardpropagate_building_block(layer_9_a,layer_8_a,weights['h10'])
layer_11_z, layer_11_a = forwardpropagate_building_block(layer_10_a,layer_9_a,weights['h11'])
layer_12_z, layer_12_a = forwardpropagate_building_block(layer_11_a,layer_10_a,weights['h12'])
layer_13_z, layer_13_a = forwardpropagate_building_block(layer_12_a,layer_11_a,weights['h13'])
layer_14_z, layer_14_a = forwardpropagate_building_block(layer_13_a,layer_12_a,weights['h14'])
layer_15_z, layer_15_a = forwardpropagate_building_block(layer_14_a,layer_13_a,weights['h15'])
layer_16_z, layer_16_a = forwardpropagate_building_block(layer_15_a,layer_14_a,weights['h16'])
layer_17_z, layer_17_a = forwardpropagate_building_block(layer_16_a,layer_15_a,weights['h17'])
layer_18_z, layer_18_a = forwardpropagate_building_block(layer_17_a,layer_16_a,weights['h18'])
layer_19_z, layer_19_a = forwardpropagate_building_block(layer_18_a,layer_17_a,weights['h19'])
layer_20_z, layer_20_a = forwardpropagate_building_block(layer_19_a,layer_18_a,weights['h20'])




# Output fully connected layer with a neuron for each class
out_layer = tf.matmul(layer_20_a, weights['out'])
#out_layer = my_activation(out_layer)  # To not have any additional linear layer which evolves in non-orthogonal way. We have also softmax at the end
# training is very slow with oplu on fully connected layer

#return out_layer





# Construct model
logits = out_layer #multilayer_perceptron(X)

# Define loss and optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=Y))



#*****************************************************************************************

# Backpropagation


opt = tf.train.GradientDescentOptimizer(learning_rate=tf_learning_rate)


grads_and_vars = opt.compute_gradients(cost, [weights['h1'],weights['h2'],weights['h3'],weights['h4'],weights['h5'],weights['h6'],weights['h7'],weights['h8'],weights['h9'],weights['h10'],weights['h11'],weights['h12'],weights['h13'],weights['h14'],weights['h15'],weights['h16'],weights['h17'],weights['h18'],weights['h19'],weights['h20'],weights['out']])


# project weights
grads_and_vars[0]=my_weight_gradient_modification(grads_and_vars[0])
grads_and_vars[1]=my_weight_gradient_modification(grads_and_vars[1])
grads_and_vars[2]=my_weight_gradient_modification(grads_and_vars[2])
grads_and_vars[3]=my_weight_gradient_modification(grads_and_vars[3])
grads_and_vars[4]=my_weight_gradient_modification(grads_and_vars[4])
grads_and_vars[5]=my_weight_gradient_modification(grads_and_vars[5])
grads_and_vars[6]=my_weight_gradient_modification(grads_and_vars[6])
grads_and_vars[7]=my_weight_gradient_modification(grads_and_vars[7])
grads_and_vars[8]=my_weight_gradient_modification(grads_and_vars[8])
grads_and_vars[9]=my_weight_gradient_modification(grads_and_vars[9])
grads_and_vars[10]=my_weight_gradient_modification(grads_and_vars[10])
grads_and_vars[11]=my_weight_gradient_modification(grads_and_vars[11])
grads_and_vars[12]=my_weight_gradient_modification(grads_and_vars[12])
grads_and_vars[13]=my_weight_gradient_modification(grads_and_vars[13])
grads_and_vars[14]=my_weight_gradient_modification(grads_and_vars[14])
grads_and_vars[15]=my_weight_gradient_modification(grads_and_vars[15])
grads_and_vars[16]=my_weight_gradient_modification(grads_and_vars[16])
grads_and_vars[17]=my_weight_gradient_modification(grads_and_vars[17])
grads_and_vars[18]=my_weight_gradient_modification(grads_and_vars[18])
grads_and_vars[19]=my_weight_gradient_modification(grads_and_vars[19])
grads_and_vars[20]=my_weight_gradient_modification(grads_and_vars[20])




# Ask the optimizer to apply the modified gradients.
optimizer = opt.apply_gradients(grads_and_vars)



projecting = [
    
    tf.assign(weights['h1'], tf.transpose(tf_gram_schmidt(tf.transpose(weights['h1'])))),
    tf.assign(weights['h2'], tf_gram_schmidt(weights['h2'])),
    tf.assign(weights['h3'], tf_gram_schmidt(weights['h3'])),
    tf.assign(weights['h4'], tf_gram_schmidt(weights['h4'])),
    tf.assign(weights['h5'], tf_gram_schmidt(weights['h5'])),
    tf.assign(weights['h6'], tf_gram_schmidt(weights['h6'])),
    tf.assign(weights['h7'], tf_gram_schmidt(weights['h7'])),
    tf.assign(weights['h8'], tf_gram_schmidt(weights['h8'])),
    tf.assign(weights['h9'], tf_gram_schmidt(weights['h9'])),
    tf.assign(weights['h10'], tf_gram_schmidt(weights['h10'])),
    tf.assign(weights['h11'], tf_gram_schmidt(weights['h11'])),
    tf.assign(weights['h12'], tf_gram_schmidt(weights['h12'])),
    tf.assign(weights['h13'], tf_gram_schmidt(weights['h13'])),
    tf.assign(weights['h14'], tf_gram_schmidt(weights['h14'])),
    tf.assign(weights['h15'], tf_gram_schmidt(weights['h15'])),
    tf.assign(weights['h16'], tf_gram_schmidt(weights['h16'])),
    tf.assign(weights['h17'], tf_gram_schmidt(weights['h17'])),
    tf.assign(weights['h18'], tf_gram_schmidt(weights['h18'])),
    tf.assign(weights['h19'], tf_gram_schmidt(weights['h19'])),
    tf.assign(weights['h20'], tf_gram_schmidt(weights['h20']))
    
    ]


#*****************************************************************************************
# compute manual gradients

d_out = tf.gradients(cost, [out_layer])  # take dE/d_out_layer_a from tensorflow for softmax with logits
d_out=d_out[0]

# there is no activation for out_layer, so we immediately compute weights
#d_out_layer = tf.multiply(d_out, activation_derivative(out_layer))
d_w_out = tf.matmul(layer_20_a, d_out, transpose_a=True)


def backpropagate_layer(layer_prev_a, layer_z, weights_next, weights_current, d_layer_z_next, name="layer"):
    # layer
    d_layer_a = tf.matmul(d_layer_z_next, weights_next, transpose_b=True)
    
    #d_layer_7_z = tf.multiply(d_layer_7_a, activation_derivative(d_layer_7_z)) # activation_derivative is a permutation matrix in case of OPLU
    d_layer_z = oplu_derivative(layer_z, d_layer_a)  # oplu_derivative function is a permutation which acts on d_layer_7_a
    
    
    d_w = tf.matmul(layer_prev_a, d_layer_z, transpose_a=True)    
    
    
    # project gradients
    d_w, _ = my_weight_gradient_modification([d_w, weights_current])
    
    
    return d_w, d_layer_z


def backpropagate_building_block(layer_prev_a, layer_z, weights_next, weights_current, d_layer_z_next, d_layer_z_next_next, name="layer"):
    
    d_w, d_layer_z = backpropagate_layer(layer_prev_a, layer_z, weights_next, weights_current, 0.5*(d_layer_z_next+d_layer_z_next_next), name=name)
    return d_w, d_layer_z
    
    
    

d_w_20, d_layer_20_z = backpropagate_layer(layer_19_a, layer_20_z, weights['out'], weights['h20'], d_out, name="layer_20")
d_w_19, d_layer_19_z = backpropagate_layer(layer_18_a, layer_19_z, weights['h20'], weights['h19'], d_layer_20_z, name="layer_19")

d_w_18, d_layer_18_z = backpropagate_building_block(layer_17_a, layer_18_z, weights['h19'], weights['h18'], d_layer_19_z, d_layer_20_z,name="layer_18")
d_w_17, d_layer_17_z = backpropagate_building_block(layer_16_a, layer_17_z, weights['h18'], weights['h17'], d_layer_18_z, d_layer_19_z,name="layer_17")
d_w_16, d_layer_16_z = backpropagate_building_block(layer_15_a, layer_16_z, weights['h17'], weights['h16'], d_layer_17_z, d_layer_18_z,name="layer_16")
d_w_15, d_layer_15_z = backpropagate_building_block(layer_14_a, layer_15_z, weights['h16'], weights['h15'], d_layer_16_z, d_layer_17_z,name="layer_15")
d_w_14, d_layer_14_z = backpropagate_building_block(layer_13_a, layer_14_z, weights['h15'], weights['h14'], d_layer_15_z, d_layer_16_z,name="layer_14")
d_w_13, d_layer_13_z = backpropagate_building_block(layer_12_a, layer_13_z, weights['h14'], weights['h13'], d_layer_14_z, d_layer_15_z,name="layer_13")
d_w_12, d_layer_12_z = backpropagate_building_block(layer_11_a, layer_12_z, weights['h13'], weights['h12'], d_layer_13_z, d_layer_14_z,name="layer_12")
d_w_11, d_layer_11_z = backpropagate_building_block(layer_10_a, layer_11_z, weights['h12'], weights['h11'], d_layer_12_z, d_layer_13_z,name="layer_11")
d_w_10, d_layer_10_z = backpropagate_building_block(layer_9_a, layer_10_z, weights['h11'], weights['h10'], d_layer_11_z, d_layer_12_z,name="layer_10")
d_w_9, d_layer_9_z = backpropagate_building_block(layer_8_a, layer_9_z, weights['h10'], weights['h9'], d_layer_10_z, d_layer_11_z,name="layer_9")
d_w_8, d_layer_8_z = backpropagate_building_block(layer_7_a, layer_8_z, weights['h9'], weights['h8'], d_layer_9_z, d_layer_10_z,name="layer_8")
d_w_7, d_layer_7_z = backpropagate_building_block(layer_6_a, layer_7_z, weights['h8'], weights['h7'], d_layer_8_z, d_layer_9_z,name="layer_7")
d_w_6, d_layer_6_z = backpropagate_building_block(layer_5_a, layer_6_z, weights['h7'], weights['h6'], d_layer_7_z, d_layer_8_z,name="layer_6")
d_w_5, d_layer_5_z = backpropagate_building_block(layer_4_a, layer_5_z, weights['h6'], weights['h5'], d_layer_6_z, d_layer_7_z,name="layer_5")
d_w_4, d_layer_4_z = backpropagate_building_block(layer_3_a, layer_4_z, weights['h5'], weights['h4'], d_layer_5_z, d_layer_6_z,name="layer_4")
d_w_3, d_layer_3_z = backpropagate_building_block(layer_2_a, layer_3_z, weights['h4'], weights['h3'], d_layer_4_z, d_layer_5_z,name="layer_3")
d_w_2, d_layer_2_z = backpropagate_building_block(layer_1_a, layer_2_z, weights['h3'], weights['h2'], d_layer_3_z, d_layer_4_z,name="layer_2")
d_w_1, d_layer_1_z = backpropagate_building_block(x, layer_1_z, weights['h2'], weights['h1'], d_layer_2_z, d_layer_3_z,name="layer_1")



optimizer2 = [
    
    # save gradients
    tf.assign(grads['h1'],
            tf.multiply(tf_learning_rate, d_w_1)),
    
    tf.assign(grads['h2'],
            tf.multiply(tf_learning_rate, d_w_2)),
    
    tf.assign(grads['h3'],
            tf.multiply(tf_learning_rate, d_w_3)),
    
    tf.assign(grads['h4'],
            tf.multiply(tf_learning_rate, d_w_4)),
    
    tf.assign(grads['h5'],
            tf.multiply(tf_learning_rate, d_w_5)),
    
    tf.assign(grads['h6'],
            tf.multiply(tf_learning_rate, d_w_6)),
    
    tf.assign(grads['h7'],
            tf.multiply(tf_learning_rate, d_w_7)),
    
    tf.assign(grads['h8'],
            tf.multiply(tf_learning_rate, d_w_8)),
    
    tf.assign(grads['h9'],
            tf.multiply(tf_learning_rate, d_w_9)),
    
    tf.assign(grads['h10'],
            tf.multiply(tf_learning_rate, d_w_10)),
    
    tf.assign(grads['h11'],
            tf.multiply(tf_learning_rate, d_w_11)),
    
    tf.assign(grads['h12'],
            tf.multiply(tf_learning_rate, d_w_12)),
    
    tf.assign(grads['h13'],
            tf.multiply(tf_learning_rate, d_w_13)),
    
    tf.assign(grads['h14'],
            tf.multiply(tf_learning_rate, d_w_14)),
    
    tf.assign(grads['h15'],
            tf.multiply(tf_learning_rate, d_w_15)),
    
    tf.assign(grads['h16'],
            tf.multiply(tf_learning_rate, d_w_16)),
    
    tf.assign(grads['h17'],
            tf.multiply(tf_learning_rate, d_w_17)),
    
    tf.assign(grads['h18'],
            tf.multiply(tf_learning_rate, d_w_18)),
    
    tf.assign(grads['h19'],
            tf.multiply(tf_learning_rate, d_w_19)),
    
    tf.assign(grads['h20'],
            tf.multiply(tf_learning_rate, d_w_20)),
    
    
    tf.assign(grads['out'],
            tf.multiply(tf_learning_rate, d_w_out)),
    
    # update weights
    
    
    tf.assign(weights['h1'],
            tf.transpose( tf_gram_schmidt(tf.transpose(tf.subtract(weights['h1'], grads['h1']))))),
    
    tf.assign(weights['h2'],
            tf_gram_schmidt(tf.subtract(weights['h2'], grads['h2']))),
    
    tf.assign(weights['h3'],
            tf_gram_schmidt(tf.subtract(weights['h3'], grads['h3']))),
    
    tf.assign(weights['h4'],
            tf_gram_schmidt(tf.subtract(weights['h4'], grads['h4']))),
    
    tf.assign(weights['h5'],
            tf_gram_schmidt(tf.subtract(weights['h5'], grads['h5']))),
    
    tf.assign(weights['h6'],
            tf_gram_schmidt(tf.subtract(weights['h6'], grads['h6']))),
    
    tf.assign(weights['h7'],
            tf_gram_schmidt(tf.subtract(weights['h7'], grads['h7']))),
    
    tf.assign(weights['h8'],
            tf_gram_schmidt(tf.subtract(weights['h8'], grads['h8']))),
    
    tf.assign(weights['h9'],
            tf_gram_schmidt(tf.subtract(weights['h9'], grads['h9']))),
    
    tf.assign(weights['h10'],
            tf_gram_schmidt(tf.subtract(weights['h10'], grads['h10']))),
    
    tf.assign(weights['h11'],
            tf_gram_schmidt(tf.subtract(weights['h11'], grads['h11']))),
    
    tf.assign(weights['h12'],
            tf_gram_schmidt(tf.subtract(weights['h12'], grads['h12']))),
    
    tf.assign(weights['h13'],
            tf_gram_schmidt(tf.subtract(weights['h13'], grads['h13']))),
    
    tf.assign(weights['h14'],
            tf_gram_schmidt(tf.subtract(weights['h14'], grads['h14']))),
    
    tf.assign(weights['h15'],
            tf_gram_schmidt(tf.subtract(weights['h15'], grads['h15']))),
    
    tf.assign(weights['h16'],
            tf_gram_schmidt(tf.subtract(weights['h16'], grads['h16']))),
    
    tf.assign(weights['h17'],
            tf_gram_schmidt(tf.subtract(weights['h17'], grads['h17']))),
    
    tf.assign(weights['h18'],
            tf_gram_schmidt(tf.subtract(weights['h18'], grads['h18']))),
    
    tf.assign(weights['h19'],
            tf_gram_schmidt(tf.subtract(weights['h19'], grads['h19']))),
    
    tf.assign(weights['h20'],
            tf_gram_schmidt(tf.subtract(weights['h20'], grads['h20']))),
    
    
    tf.assign(weights['out'],
            tf.subtract(weights['out'], grads['out']))
    
    
    ]






# layer_6


##############################################




# Test model
pred = tf.nn.softmax(logits)  # Apply softmax to logits
#pred = logits


#pred = tf.Print(pred, [tf.shape(grads_and_vars[0]),tf.shape(grads_and_vars[1]),tf.shape(grads_and_vars[4])])

correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(Y, 1))
# Calculate accuracy
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))




# Initializing the variables
init = tf.global_variables_initializer()


passed_98 = False
passed_985 = False
passed_987 = False
passed_9884 = False
passed_99 = False
passed_992 = False


# Config to turn on JIT compilation https://www.tensorflow.org/performance/xla/jit
config = tf.ConfigProto()
config.graph_options.optimizer_options.global_jit_level = tf.OptimizerOptions.ON_1
run_metadata = tf.RunMetadata()

with tf.Session(config=config) as sess:
    sess.run(init)
    
        
    #### 
    #print('check orthogonality of inner layer h2 at start')
    #p = tf.matmul(weights['h2'], weights['h2'],transpose_a=True)
    
    #print(p.eval()) # just take last batch, do not generate anything
    #print(p.eval({x: mnist.test.images, y: mnist.test.labels}))
    ####
   
    
    

    # Training cycle
    for epoch in range(training_epochs):
        
        load_epoch(epoch)
        
        avg_cost = 0.
        
        ntrain = X1.shape[0]
        total_batch = int(ntrain / batch_size)
        arrayindices = range(ntrain)
        random.shuffle(arrayindices)
        
        logger.debug("epoch=%d"%(epoch+1))
        logger.debug("mnist element,[grad1,grad2,grad3,grad4,grad5,grad6,grad7,grad8,grad9,grad10,grad11,grad12,grad13,grad14,grad15,grad16,grad17,grad18,grad19,grad20,grad_out]")
        
        
        recorded = False
        
        # Loop over all batches
        for i in range(total_batch):
            
            idx = arrayindices[i*batch_size:(i+1)*batch_size]
            batch_x, batch_y = get_batch_train(idx)
            
            #print(cost.eval({X: batch_x, Y: batch_y, tf_learning_rate: learning_rate}))
            
            # Run optimization op (backprop) and cost op (to get loss value)
            
            _, c = sess.run([optimizer, cost], feed_dict={X: batch_x, Y: batch_y, tf_learning_rate: learning_rate})
            
            sess.run([projecting])
            
            
            '''
            _, c = sess.run([optimizer, cost], feed_dict={X: batch_x, Y: batch_y, tf_learning_rate: learning_rate},options=tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE), run_metadata=run_metadata)
            
            # Create a timeline for the last loop and export to json to view with
            # chrome://tracing/.
            trace = timeline.Timeline(step_stats=run_metadata.step_stats)
            
            if not recorded:
                with open('timeline.ctf.json', 'w') as trace_file:
                    trace_file.write(trace.generate_chrome_trace_format())
                recorded = True
            '''
            
            
            # dump gradients
            '''
            np_grads = sess.run([grads['h1'],grads['h2'],grads['h3'],grads['h4'],grads['h5'],grads['h6'],grads['h7'],grads['h8'],grads['h9'],grads['h10'],grads['h11'],grads['h12'],grads['h13'],grads['h14'],grads['h15'],grads['h16'],grads['h17'],grads['h18'],grads['h19'],grads['h20'],grads['out']])
            
            np_grads = [np.linalg.norm(x) for x in np_grads]
            
            logger.debug("i=%05d, %s"%(i,np_grads))
            '''
            
            if(i%1000)==0:
                accuracy_train_val = (accuracy.eval({X: X1, Y: T1}))*100
                accuracy_test_val = (accuracy.eval({X: X2, Y: T2}))*100
                print("i: %04d, accuracy_train = %1.2f%%, accuracy_test = %1.2f%%" % (i,accuracy_train_val,accuracy_test_val))
            
            # Compute average loss
            avg_cost += c / total_batch
            
        
        # Display logs per epoch step
        if epoch % display_step == 0:
            accuracy_train_val = (accuracy.eval({X: X1, Y: T1})) * 100                   
            accuracy_test_val = (accuracy.eval({X: X2, Y: T2}))*100

            print("Epoch:", '%04d' % (epoch+1), "learning rate: %1.9f"%learning_rate," cost={:.9f}".format(avg_cost),"||W2*W2^t - I||L2 = %1.4f " % ort_discrepancy(weights['h3']).eval(),  "accuracy_train = %1.2f%%" % accuracy_train_val, "accuracy_test = %1.2f%%" % accuracy_test_val)
            
            
            #logger.debug("Epoch: %04d, accuracy_train = %1.2f%%, accuracy_test = %1.2f%%" % (epoch+1,accuracy_train_val,accuracy_test_val))
            #logger.debug("Epoch:", '%04d' % (epoch+1), "learning rate: %1.9f"%learning_rate," cost={:.9f}".format(avg_cost),"||W2*W2^t - I||L2 = %1.4f " % ort_discrepancy(weights['h3']).eval(),  "accuracy_train = %1.2f%%" % accuracy_train_val, "accuracy_test = %1.2f%%" % accuracy_test_val)
           
           
        learning_rate = learning_rate*learning_rate_decay  #manually decay learning rate
        
        # empirical slowdown after reaching 98%
        if accuracy_test_val>98. and (not passed_98):
            learning_rate = learning_rate/1.5
            passed_98 = True
            momentum += 0.1
            
        if accuracy_test_val>98.5 and (not passed_985):
            learning_rate = learning_rate/1.5
            passed_985 = True
            momentum += 0.1
            
        if accuracy_test_val>98.7 and (not passed_987):
            learning_rate = learning_rate/1.5
            passed_987 = True
            momentum += 0.1
            
        if accuracy_test_val>98.84 and (not passed_9884):
            learning_rate = learning_rate/2.5
            passed_9884 = True
            #momentum += 0.1
            
        if accuracy_test_val>99. and (not passed_99):
            learning_rate = learning_rate/1.5
            passed_99 = True
            
        if accuracy_test_val>99.2 and (not passed_992):
            learning_rate = learning_rate/1.5
            passed_992 = True




    print("Optimization Finished!")

    #### 
    print('check orthogonality of inner layer h2 at the end')
    p = tf.matmul(weights['h2'], weights['h2'],transpose_a = True)
    
    print(p.eval()) # just take last batch, do not generate anything
    #print(p.eval({x: mnist.test.images, y: mnist.test.labels}))
    ####
    
 
