
from __future__ import print_function
import numpy as np
import random
import h5py #matlab v7.3 files


np.random.seed(1000)

# Import MNIST data


augmented_datasets_available = 100

    
mat_contents =  h5py.File('matlab_mnist/affNIST.mat')

X2 = np.array(mat_contents['all_images_val'],dtype=np.float32) # (10000, 1600) FLOAT32
T2 = np.array(mat_contents['all_labels_val'],dtype=np.float32) # (10000, 10) 

# clear space
mat_contents = None


X1,T1 = None,None


def load_epoch(epoch):
    
    if epoch>=augmented_datasets_available:
        epoch=epoch%augmented_datasets_available
  
    mat_contents = h5py.File('augdata/augmented'+str(epoch)+'.h5')
    
    #load new data
    
    global X1,T1
    
    X1 = np.array(mat_contents['all_images_train'],dtype=np.float32) # (1600000,1600) #already in Float32 from data augmenter!
    
    
    #X1 = np.random.normal(0,1,X2.shape)
    
    
    T1 = np.array(mat_contents['all_labels_train'],dtype=np.float32) # (1600000, 10)
    
    
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

#from adamax import *

# Parameters
batch_size = 1000

learning_rate = 0.00005#1.0/batch_size**2
learning_rate_decay = 0.99

momentum = 0.3

training_epochs = 1000
display_step = 1

# Network Parameters
n_hidden_1 = 2048 # 1st layer, adjusts data to the new size
n_hidden_2 = 1024 # Must be exactly 1st_layer/2 and so on!    Two matrices of second layer are initialized
n_hidden_3 = 512  # Four matrices of this layer are initialized
n_hidden_4 = 256  # Eight matrices of this layer are initialized

n_input = X1.shape[1]
n_classes = T1.shape[1]







# tf Graph input
X = tf.placeholder("float", [None, n_input])
Y = tf.placeholder("float", [None, n_classes])
tf_learning_rate = tf.placeholder(tf.float32, shape=[])  # need to be a placeholder if we want to apply new learning rate on each step

b = np.array([])

# function  that initializes orthogonal matrix
def ort_initializer(shape, dtype=tf.float32):
    
      
      scale = 1.0
      flat_shape = (shape[0], np.prod(shape[1:]))
      
      global b
      if b.shape == flat_shape:
          a = b
      else:
        a = np.random.normal(0, 1, flat_shape)
        b = a
      u, _, v = np.linalg.svd(a, full_matrices=False)
      # pick the one with the correct shape
      q = u if u.shape == flat_shape else v
      q = q.reshape(shape) #this needs to be corrected to float32
      #print('you have initialized one orthogonal matrix.')
      mat=tf.constant(scale * q[:shape[0], :shape[1]], dtype=tf.float32)
      
      #i = tf.matmul(mat,mat,transpose_b=True)
      
      #scale = 1.0/tf.sqrt(i[0,0])
      
      return scale*mat


# compute ||W*W^T - I||L2 for a matrix
def ort_discrepancy(matrix):
    
    wwt = tf.matmul(matrix, matrix, transpose_a=True)
    
    rows = tf.shape(wwt)[0]
    cols = tf.shape(wwt)[1]
    
    return tf.norm((wwt - tf.eye(rows,cols)),ord='euclidean') #/tf.to_float(rows*cols)


# this function receives tf.Variable which is not iterable, hence complications
def gram_schmidt(vectors):
    basis = []
    for v in vectors:
        w = v - np.sum( np.dot(v,b)*b  for b in basis )
        if (w > 1e-10).any():  
            basis.append(w/np.linalg.norm(w))
        else:
            basis.append(np.zeros(w.shape))
    return np.array(basis)

# Store layers weight & bias

weights = {
    #'h1': tf.Variable(ort_initializer([n_input+1, n_hidden_1])),  # use this to add biases
    'h1': tf.Variable(ort_initializer([n_input, n_hidden_1])),
    
    'h21': tf.Variable(ort_initializer([n_hidden_2, n_hidden_2])),
    'h22': tf.Variable(ort_initializer([n_hidden_2, n_hidden_2])),
    
    
    'h31': tf.Variable(ort_initializer([n_hidden_3, n_hidden_3])),
    'h32': tf.Variable(ort_initializer([n_hidden_3, n_hidden_3])),
    'h33': tf.Variable(ort_initializer([n_hidden_3, n_hidden_3])),
    'h34': tf.Variable(ort_initializer([n_hidden_3, n_hidden_3])),
    
    
    
    
    'h41': tf.Variable(ort_initializer([n_hidden_4, n_hidden_4])),
    'h42': tf.Variable(ort_initializer([n_hidden_4, n_hidden_4])),
    'h43': tf.Variable(ort_initializer([n_hidden_4, n_hidden_4])),
    'h44': tf.Variable(ort_initializer([n_hidden_4, n_hidden_4])),
    'h45': tf.Variable(ort_initializer([n_hidden_4, n_hidden_4])),
    'h46': tf.Variable(ort_initializer([n_hidden_4, n_hidden_4])),
    'h47': tf.Variable(ort_initializer([n_hidden_4, n_hidden_4])),
    'h48': tf.Variable(ort_initializer([n_hidden_4, n_hidden_4])),
    
    
    'summary': tf.Variable(ort_initializer([n_hidden_4*8, n_hidden_4])),
    
    'out1': tf.Variable(ort_initializer([n_hidden_4, n_classes]))
      
}





def combine_even_odd(even, odd):
    res = tf.reshape(tf.concat([even[..., tf.newaxis], odd[..., tf.newaxis]], axis=-1), [tf.shape(even)[0], -1])
    return res




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
    
    return grad_new, variable


# separate layer into two layers
def separate_oplu(x):
    even = x[:,::2]
    odd = x[:,1::2]
    return even, odd

def combine_summary(layer_41_a,layer_42_a,layer_43_a,layer_44_a,layer_45_a,layer_46_a,layer_47_a,layer_48_a):
    return tf.concat([layer_41_a,layer_42_a,layer_43_a,layer_44_a,layer_45_a,layer_46_a,layer_47_a,layer_48_a],1)


def split_summary(layer_summary_z):
    layer_41_a = layer_summary_z[:,0:n_hidden_4]
    layer_42_a = layer_summary_z[:,n_hidden_4:2*n_hidden_4]
    layer_43_a = layer_summary_z[:,2*n_hidden_4:3*n_hidden_4]
    layer_44_a = layer_summary_z[:,3*n_hidden_4:4*n_hidden_4]
    layer_45_a = layer_summary_z[:,4*n_hidden_4:5*n_hidden_4]
    layer_46_a = layer_summary_z[:,5*n_hidden_4:6*n_hidden_4]
    layer_47_a = layer_summary_z[:,6*n_hidden_4:7*n_hidden_4]
    layer_48_a = layer_summary_z[:,7*n_hidden_4:8*n_hidden_4]
    return layer_41_a,layer_42_a,layer_43_a,layer_44_a,layer_45_a,layer_46_a,layer_47_a,layer_48_a



def pearson_correlation_x_axis(x,y):
    length = tf.cast(tf.shape(x),dtype=tf.float32)#512
    
    original_loss =  length * tf.reduce_sum(tf.multiply(x, y)) - (tf.reduce_sum(x) * tf.reduce_sum(y))
    divisor = tf.sqrt(
        (length * tf.reduce_sum(tf.square(x)) - tf.square(tf.reduce_sum(x))) *
        (length * tf.reduce_sum(tf.square(y)) - tf.square(tf.reduce_sum(y)))
    )
    original_loss = tf.truediv(original_loss, divisor)
    return original_loss
    

# Create model
#def multilayer_perceptron(x):



# bias replacements
x = X

ones = tf.ones([tf.shape(x)[0],1],tf.float32)

################### Adding biases 

# Let's change first pixel instead. We do not lose anything, as actual digit is smaller (28x28) in the middle of the 40x40 square
x = tf.concat([x[:,1:],ones], 1)

################### Compute Forward prop

############## Layer 1
# First layer - increase data dimension to the power of two
layer_1_z = tf.matmul(x, weights['h1'])
layer_1_a = OPLU(layer_1_z)
# split layer 1 
even1, odd1 = separate_oplu(layer_1_a)



############## Layer 2
# separately apply layer 2
layer_21_z = tf.matmul(even1, weights['h21'])
layer_21_a = OPLU(layer_21_z)

layer_22_z = tf.matmul(odd1, weights['h22'])
layer_22_a = OPLU(layer_22_z)

# split the data
even21, odd21 = separate_oplu(layer_21_a)
even22, odd22 = separate_oplu(layer_22_a)



############## Layer 3
#separately apply layer 3
layer_31_z = tf.matmul(even21, weights['h31'])
layer_31_a = OPLU(layer_31_z)
layer_32_z = tf.matmul(odd21, weights['h32'])
layer_32_a = OPLU(layer_32_z)
layer_33_z = tf.matmul(even22, weights['h33'])
layer_33_a = OPLU(layer_33_z)
layer_34_z = tf.matmul(odd22, weights['h34'])
layer_34_a = OPLU(layer_34_z)

# split the data
even31, odd31 = separate_oplu(layer_31_a)
even32, odd32 = separate_oplu(layer_32_a)
even33, odd33 = separate_oplu(layer_33_a)
even34, odd34 = separate_oplu(layer_34_a)


############## Layer 4
layer_41_z = tf.matmul(even31, weights['h41'])
layer_41_a = OPLU(layer_41_z)
layer_42_z = tf.matmul(odd31, weights['h42'])
layer_42_a = OPLU(layer_42_z)
layer_43_z = tf.matmul(even32, weights['h43'])
layer_43_a = OPLU(layer_43_z)
layer_44_z = tf.matmul(odd32, weights['h44'])
layer_44_a = OPLU(layer_44_z)
layer_45_z = tf.matmul(even33, weights['h45'])
layer_45_a = OPLU(layer_45_z)
layer_46_z = tf.matmul(odd33, weights['h46'])
layer_46_a = OPLU(layer_46_z)
layer_47_z = tf.matmul(even34, weights['h47'])
layer_47_a = OPLU(layer_47_z)
layer_48_z = tf.matmul(odd34, weights['h48'])
layer_48_a = OPLU(layer_48_z)


############## Out layers

layer_summary_in = combine_summary(layer_41_a,layer_42_a,layer_43_a,layer_44_a,layer_45_a,layer_46_a,layer_47_a,layer_48_a)
                                   
layer_summary_z = tf.matmul(layer_summary_in, weights['summary'])
layer_summary_a = layer_summary_z #tf.nn.softplus(layer_summary_z) #tf.nn.relu(layer_summary_z)


out1 = tf.matmul(layer_summary_a, weights['out1'])


'''
out1 = tf.matmul(layer_41_a, weights['out1'])
out2 = tf.matmul(layer_42_a, weights['out2'])
out3 = tf.matmul(layer_43_a, weights['out3'])
out4 = tf.matmul(layer_44_a, weights['out4'])
out5 = tf.matmul(layer_45_a, weights['out5'])
out6 = tf.matmul(layer_46_a, weights['out6'])
out7 = tf.matmul(layer_47_a, weights['out7'])
out8 = tf.matmul(layer_48_a, weights['out8'])
'''

# softmax
'''
out1s = tf.nn.softmax(out1)
out2s = tf.nn.softmax(out2)
out3s = tf.nn.softmax(out3)
out4s = tf.nn.softmax(out4)
out5s = tf.nn.softmax(out5)
out6s = tf.nn.softmax(out6)
out7s = tf.nn.softmax(out7)
out8s = tf.nn.softmax(out8)


# averaging output
mean_out = tf.reduce_mean(tf.stack([out1s,out2s,out3s,out4s,out5s,out6s,out7s,out8s],1),1)
'''

out1s = tf.nn.softmax(out1)
# Construct model
logits = out1s #mean_out #multilayer_perceptron(X)

# Define loss and optimizer
#cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=Y))
cost = tf.nn.l2_loss((Y-logits))

#opt = tf.train.GradientDescentOptimizer(learning_rate=tf_learning_rate)
#opt = tf.train.MomentumOptimizer(learning_rate=tf_learning_rate, momentum=momentum, use_nesterov=True)
#opt = AdamaxOptimizer(learning_rate=tf_learning_rate) # default values are learning_rate=0.001, beta1=0.9, beta2=0.999, use_locking=False

# !!! Comment this out and uncomment code below for gradient projection
#optimizer = opt.minimize(cost) 

##############################################
## Here I want to do manual gradient modification
##############################################


# Compute the gradients for a list of variables I am interested in.
# Those are true weight gradients:

#grads_and_vars = opt.compute_gradients(cost, [weights['h1'],weights['h2'],weights['h3'],weights['h4'],weights['h5'],weights['h6'],weights['h7'],weights['out']])
# grads_and_vars is a list of tuples (gradient, variable).  Do whatever you
# need to the 'gradient' part, for example cap them, etc.



'''


# project weights


grads_and_vars[6]=my_weight_gradient_modification(grads_and_vars[6])
grads_and_vars[5]=my_weight_gradient_modification(grads_and_vars[5])
grads_and_vars[4]=my_weight_gradient_modification(grads_and_vars[4])
grads_and_vars[3]=my_weight_gradient_modification(grads_and_vars[3])
grads_and_vars[2]=my_weight_gradient_modification(grads_and_vars[2])
grads_and_vars[1]=my_weight_gradient_modification(grads_and_vars[1])
grads_and_vars[0]=my_weight_gradient_modification(grads_and_vars[0])

'''

# weights['out'] are not fixed !

# Ask the optimizer to apply the modified gradients.


#optimizer = opt.apply_gradients(grads_and_vars)


# compute manual gradients

d_out1 = tf.gradients(cost, [out1])[0]  # take dE/d_out_layer_a from tensorflow for softmax with logits

d_w_out_1 = tf.matmul(layer_summary_a, d_out1, transpose_a=True)

# summary layer

d_layer_summary_a = tf.matmul(d_out1, weights['out1'], transpose_b=True)

def tanh_derivative(x):
    return 1.0-tf.square((tf.tanh(x)))

def relu_derivative(x):
    return tf.cast(x>0, dtype=tf.float32)

def softplus_derivative(x):
    return tf.nn.sigmoid(x)

d_layer_summary_z = d_layer_summary_a#tf.multiply(softplus_derivative(layer_summary_z),d_layer_summary_a)#tf.multiply(relu_derivative(layer_summary_z),d_layer_summary_a) 

d_layer_summary_w = tf.matmul(layer_summary_in, d_layer_summary_z, transpose_a=True)    


# layer 4 (split layer)

d_layer_4_all_a = tf.matmul(d_layer_summary_z, weights['summary'], transpose_b=True)

d_layer_41_a,d_layer_42_a,d_layer_43_a,d_layer_44_a,d_layer_45_a,d_layer_46_a,d_layer_47_a,d_layer_48_a = split_summary(d_layer_4_all_a)


#d_out = tf.Print(d_out,[d_out], message = "out grad")
#d_out = tf.Print(d_out,[tf.shape(d_out)], message = "out grad shape")   # shape is [1, batch_size, n_classes] = [1,100,10]
#d_out=d_out[0]

# compute weights
#d_out_layer = tf.multiply(d_out, activation_derivative(out_layer))
#d_w_out = tf.matmul(layer_7_a, d_out, transpose_a=True)


def backpropagate_layer_pre_split(layer_pred_a, layer_z, d_layer_a, weights_current, name="layer_41"):
    # layer
    #d_layer_a = tf.matmul(d_layer_z_next, weights_next, transpose_b=True)
    #d_layer_a = tf.Print(d_layer_a, [tf.norm(d_layer_a)], message="d_"+name+"_a")

    #d_layer_7_z = tf.multiply(d_layer_7_a, activation_derivative(d_layer_7_z)) # activation_derivative is a permutation matrix in case of OPLU
    d_layer_z = oplu_derivative(layer_z, d_layer_a)  # oplu_derivative function is a permutation which acts on d_layer_7_a
    #d_layer_z = tf.Print(d_layer_z, [tf.norm(d_layer_z)], message="d_"+name+"_z")

    d_w = tf.matmul(layer_pred_a, d_layer_z, transpose_a=True)    
    
    # project gradients
    d_w, _ = my_weight_gradient_modification([d_w, weights_current])
    
    
    #d_w = tf.Print(d_w,[tf.norm(d_w)], message="***** d_w_"+name)
    
    
    return d_w, d_layer_z



def backpropagate_layer(layer_pred_a, layer_z, weights_next, weights_current, d_layer_z_next, name="layer_41"):
    # layer
    d_layer_a = tf.matmul(d_layer_z_next, weights_next, transpose_b=True)
    #d_layer_a = tf.Print(d_layer_a, [tf.norm(d_layer_a)], message="d_"+name+"_a")

    #d_layer_7_z = tf.multiply(d_layer_7_a, activation_derivative(d_layer_7_z)) # activation_derivative is a permutation matrix in case of OPLU
    d_layer_z = oplu_derivative(layer_z, d_layer_a)  # oplu_derivative function is a permutation which acts on d_layer_7_a
    #d_layer_z = tf.Print(d_layer_z, [tf.norm(d_layer_z)], message="d_"+name+"_z")

    d_w = tf.matmul(layer_pred_a, d_layer_z, transpose_a=True)    
    
    # project gradients
    d_w, _ = my_weight_gradient_modification([d_w, weights_current])
    
    
    #d_w = tf.Print(d_w,[tf.norm(d_w)], message="***** d_w_"+name)
    
    
    return d_w, d_layer_z



def backpropagate_split_layer(layer_pred_a, layer_z, weights_next_even, weights_next_odd, weights_current, d_layer_z_next_even, d_layer_z_next_odd, name="layer_41"):
    # layer
    d_layer_a_even = tf.matmul(d_layer_z_next_even, weights_next_even, transpose_b=True)
    d_layer_a_odd = tf.matmul(d_layer_z_next_odd, weights_next_odd, transpose_b=True)
    
    d_layer_a = combine_even_odd(d_layer_a_even, d_layer_a_odd)
    
    
    #d_layer_a = tf.Print(d_layer_a, [tf.norm(d_layer_a)], message="d_"+name+"_a")

    #d_layer_7_z = tf.multiply(d_layer_7_a, activation_derivative(d_layer_7_z)) # activation_derivative is a permutation matrix in case of OPLU
    d_layer_z = oplu_derivative(layer_z, d_layer_a)  # oplu_derivative function is a permutation which acts on d_layer_7_a
    #d_layer_z = tf.Print(d_layer_z, [tf.norm(d_layer_z)], message="d_"+name+"_z")

    d_w = tf.matmul(layer_pred_a, d_layer_z, transpose_a=True)    
    
    # project gradients
    d_w, _ = my_weight_gradient_modification([d_w, weights_current])
    
    
    #d_w = tf.Print(d_w,[tf.norm(d_w)], message="***** d_w_"+name)
    
    
    return d_w, d_layer_z




######## Layer 4

d_w_41, d_layer_41_z = backpropagate_layer_pre_split(even31, layer_41_z, d_layer_41_a, weights['h41'], name="layer_41")
d_w_42, d_layer_42_z = backpropagate_layer_pre_split(odd31, layer_42_z, d_layer_42_a, weights['h42'], name="layer_42")
# merging back
#d_layer_41_42_z = combine_even_odd(d_layer_41_z, d_layer_42_z)


d_w_43, d_layer_43_z = backpropagate_layer_pre_split(even32, layer_43_z, d_layer_43_a, weights['h43'], name="layer_43")
d_w_44, d_layer_44_z = backpropagate_layer_pre_split(odd32, layer_44_z, d_layer_44_a, weights['h44'], name="layer_44")
# merging back
#d_layer_43_44_z = combine_even_odd(d_layer_43_z, d_layer_44_z)

d_w_45, d_layer_45_z = backpropagate_layer_pre_split(even33, layer_45_z, d_layer_45_a, weights['h45'], name="layer_45")
d_w_46, d_layer_46_z = backpropagate_layer_pre_split(odd33, layer_46_z, d_layer_46_a, weights['h46'],name="layer_46")
# merging back
#d_layer_45_46_z = combine_even_odd(d_layer_45_z, d_layer_46_z)

d_w_47, d_layer_47_z = backpropagate_layer_pre_split(even34, layer_47_z, d_layer_47_a, weights['h47'],name="layer_47")
d_w_48, d_layer_48_z = backpropagate_layer_pre_split(odd34, layer_48_z, d_layer_48_a, weights['h48'], name="layer_48")
# merging back
#d_layer_47_48_z = combine_even_odd(d_layer_47_z, d_layer_48_z)



######## Layer 3

d_w_31, d_layer_31_z = backpropagate_split_layer(even21, layer_31_z, weights['h41'], weights['h42'], weights['h31'], d_layer_41_z, d_layer_42_z, name="layer_31")
d_w_32, d_layer_32_z = backpropagate_split_layer(odd21, layer_32_z, weights['h43'], weights['h44'], weights['h32'], d_layer_43_z, d_layer_44_z,  name="layer_32")

d_w_33, d_layer_33_z = backpropagate_split_layer(even22, layer_33_z, weights['h45'], weights['h46'], weights['h33'], d_layer_45_z, d_layer_46_z, name="layer_33")
d_w_34, d_layer_34_z = backpropagate_split_layer(odd22, layer_34_z, weights['h47'], weights['h48'], weights['h34'], d_layer_47_z, d_layer_48_z,  name="layer_34")


######## Layer 2

d_w_21, d_layer_21_z = backpropagate_split_layer(even1, layer_21_z, weights['h31'], weights['h32'], weights['h21'], d_layer_31_z, d_layer_32_z, name="layer_21")
d_w_22, d_layer_22_z = backpropagate_split_layer(odd1, layer_22_z, weights['h33'], weights['h34'], weights['h22'], d_layer_33_z, d_layer_34_z,  name="layer_22")


######## Layer 1

d_w_1, d_layer_1_z = backpropagate_split_layer(x, layer_1_z, weights['h21'], weights['h22'], weights['h1'], d_layer_21_z, d_layer_22_z, name="layer_1")




optimizer = [
    
    tf.assign(weights['out1'],
            tf.subtract(weights['out1'], tf.multiply(tf_learning_rate*(8+np.sqrt(8))/2, d_w_out_1))),
    
    tf.assign(weights['summary'],
            tf.subtract(weights['summary'], tf.multiply(tf_learning_rate, d_layer_summary_w))),
    
    
    tf.assign(weights['h41'],
            tf.subtract(weights['h41'], tf.multiply(tf_learning_rate*(8+np.sqrt(8))/2, d_w_41))),
    tf.assign(weights['h42'],
            tf.subtract(weights['h42'], tf.multiply(tf_learning_rate*(8+np.sqrt(8))/2, d_w_42))),
    tf.assign(weights['h43'],
            tf.subtract(weights['h43'], tf.multiply(tf_learning_rate*(8+np.sqrt(8))/2, d_w_43))),
    tf.assign(weights['h44'],
            tf.subtract(weights['h44'], tf.multiply(tf_learning_rate*(8+np.sqrt(8))/2, d_w_44))),
    tf.assign(weights['h45'],
            tf.subtract(weights['h45'], tf.multiply(tf_learning_rate*(8+np.sqrt(8))/2, d_w_45))),
    tf.assign(weights['h46'],
            tf.subtract(weights['h46'], tf.multiply(tf_learning_rate*(8+np.sqrt(8))/2, d_w_46))),
    tf.assign(weights['h47'],
            tf.subtract(weights['h47'], tf.multiply(tf_learning_rate*(8+np.sqrt(8))/2, d_w_47))),
    tf.assign(weights['h48'],
            tf.subtract(weights['h48'], tf.multiply(tf_learning_rate*(8+np.sqrt(8))/2, d_w_48))),
    
    
    
    
    
    tf.assign(weights['h31'],
            tf.subtract(weights['h31'], tf.multiply(tf_learning_rate*(4+np.sqrt(4))/2, d_w_31))),
    tf.assign(weights['h32'],
            tf.subtract(weights['h32'], tf.multiply(tf_learning_rate*(4+np.sqrt(4))/2, d_w_32))),
    tf.assign(weights['h33'],
            tf.subtract(weights['h33'], tf.multiply(tf_learning_rate*(4+np.sqrt(4))/2, d_w_33))),
    tf.assign(weights['h34'],
            tf.subtract(weights['h34'], tf.multiply(tf_learning_rate*(4+np.sqrt(4))/2, d_w_34))),
    
    
    tf.assign(weights['h21'],
            tf.subtract(weights['h21'], tf.multiply(tf_learning_rate*(2+np.sqrt(2))/2, d_w_21))),
    tf.assign(weights['h22'],
            tf.subtract(weights['h22'], tf.multiply(tf_learning_rate*(2+np.sqrt(2))/2, d_w_22))),
    
    
    
    
    tf.assign(weights['h1'],
            tf.subtract(weights['h1'], tf.multiply(tf_learning_rate, d_w_1)))
    
    
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


def pearson_correlation_21():
    return pearson_correlation_x_axis(even21[0,:],even21[1,:])
def pearson_correlation_31():
    return pearson_correlation_x_axis(even31[0,:],even31[1,:])
def pearson_correlation_41():
    even41, odd41 = separate_oplu(layer_41_a)
    return pearson_correlation_x_axis(even41[0,:],even41[1,:])


with tf.Session() as sess:
    sess.run(init)
    
    
    #### 
    print('check orthogonality of inner layer h2 at start')
    p = tf.matmul(weights['h21'], weights['h21'],transpose_a=True)
    
    print(p.eval()) # just take last batch, do not generate anything
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
        
        # Loop over all batches
        for i in range(total_batch):
            
            idx = arrayindices[i*batch_size:(i+1)*batch_size]
            batch_x, batch_y = get_batch_train(idx)
            
            
            # Run optimization op (backprop) and cost op (to get loss value)
            _, c = sess.run([optimizer, cost], feed_dict={X: batch_x, Y: batch_y, tf_learning_rate: learning_rate})
            
            #print("---")
            # Compute average loss
            avg_cost += c / total_batch
        # Display logs per epoch step
        if epoch % display_step == 0:
            accuracy_train_val = (accuracy.eval({X: X1, Y: T1})) * 100                   
            accuracy_test_val = (accuracy.eval({X: X2, Y: T2}))*100
            
            ort_discrepancy21 = ort_discrepancy(weights['h21']).eval()
            
            print("Epoch:", '%04d' % (epoch+1), "lr: %1.9f"%learning_rate," cost={:.9f}".format(avg_cost),"||W21*W21^t - I||L2 = %1.4f " % ort_discrepancy21, "||W48*W48^t - I||L2 = %1.4f " % ort_discrepancy(weights['h48']).eval(),  "Pearson layer 2 = %1.4f "%pearson_correlation_21().eval({X: batch_x, Y: batch_y, tf_learning_rate: learning_rate}), "Pearson layer 3 = %1.4f "%pearson_correlation_31().eval({X: batch_x, Y: batch_y, tf_learning_rate: learning_rate}), "Pearson layer 4 = %1.4f "%pearson_correlation_41().eval({X: batch_x, Y: batch_y, tf_learning_rate: learning_rate}), "accuracy_train = %1.2f%%" % accuracy_train_val, "accuracy_test = %1.2f%%" % accuracy_test_val)
           
        learning_rate = learning_rate*learning_rate_decay  #manually decay learning rate
        
        
        # Apply gram_schmidt to fix weights
        if ort_discrepancy21 > 0.001:
                print("Fixing orthogonality...")
                
                weights = {
                # non-square matrix, watch rows/columns!
                'h1': tf.Variable(gram_schmidt(weights['h1'].eval()),dtype=tf.float32),
                
                'h21':tf.Variable(gram_schmidt(weights['h21'].eval()),dtype=tf.float32),
                'h22':tf.Variable(gram_schmidt(weights['h22'].eval()),dtype=tf.float32),
                                                         
                'h31':tf.Variable(gram_schmidt(weights['h31'].eval()),dtype=tf.float32),
                'h32':tf.Variable(gram_schmidt(weights['h32'].eval()),dtype=tf.float32),
                'h33':tf.Variable(gram_schmidt(weights['h33'].eval()),dtype=tf.float32),
                'h34':tf.Variable(gram_schmidt(weights['h34'].eval()),dtype=tf.float32),
                                                        
                'h41':tf.Variable(gram_schmidt(weights['h41'].eval()),dtype=tf.float32),
                'h42':tf.Variable(gram_schmidt(weights['h42'].eval()),dtype=tf.float32),
                'h43':tf.Variable(gram_schmidt(weights['h43'].eval()),dtype=tf.float32),
                'h44':tf.Variable(gram_schmidt(weights['h44'].eval()),dtype=tf.float32),
                'h45':tf.Variable(gram_schmidt(weights['h45'].eval()),dtype=tf.float32),
                'h46':tf.Variable(gram_schmidt(weights['h46'].eval()),dtype=tf.float32),
                'h47':tf.Variable(gram_schmidt(weights['h47'].eval()),dtype=tf.float32),
                'h48':tf.Variable(gram_schmidt(weights['h48'].eval()),dtype=tf.float32),
                                                            
                'summary':weights['summary'],
                'out1':weights['out1']
                }
                sess.run(tf.variables_initializer([weights['h1'],weights['h21'],weights['h22'],weights['h31'],weights['h32'],weights['h33'],weights['h34'],weights['h41'],weights['h42'],weights['h43'],weights['h44'],weights['h45'],weights['h46'],weights['h47'],weights['h48']])) # re-initialize variables
        
        
        
        
        '''
        # empirical slowdown after reaching 98%
        if accuracy_test_val>98. and (not passed_98):
            learning_rate = learning_rate/1.5
            passed_98 = True
            momentum += 0.1
        '''
            
        if accuracy_test_val>98.5 and (not passed_985):
            learning_rate = learning_rate/2
            passed_985 = True
            momentum += 0.1
        '''
            
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

        '''


    print("Optimization Finished!")

    #### 
    print('check orthogonality of inner layer h2 at the end')
    p = tf.matmul(weights['h21'], weights['h21'],transpose_a = True)
    
    print(p.eval()) # just take last batch, do not generate anything
    #print(p.eval({x: mnist.test.images, y: mnist.test.labels}))
    ####
    
 
