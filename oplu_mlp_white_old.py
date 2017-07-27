'''
A Multilayer Perceptron implementation example using TensorFlow library.
This example is using the MNIST database of handwritten digits
(http://yann.lecun.com/exdb/mnist/)
Author: Aymeric Damien
Project: https://github.com/aymericdamien/TensorFlow-Examples/
'''



from __future__ import print_function

import numpy as np


# Import MNIST data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

import tensorflow as tf


# activation --------------------------


import scipy.io as sio

# input whitened mnist
mat_contents =  sio.loadmat('matlab_mnist/mnist.mat') 
mnist_white = np.array(mat_contents['xTilde']) # 300, 60000
#labels = mnist.labels; # not original data!  The data needs to be loaded as numpy.float32 (and normalized by dividing it through 255).
#https://stackoverflow.com/questions/40592422/why-doesnt-the-tf-mnist-example-work-with-original-data
labels = np.transpose(np.array(mat_contents['labelsmat'])) # 10, 60000


import random  


def get_batch(batchsize):
    
    arrayindices = range(labels.shape[0]-batchsize) #last one will be for testing
    
    random.shuffle(arrayindices)
    
    return np.transpose(mnist_white[:,arrayindices[0:batchsize]]), labels[arrayindices[0:batchsize],:]

def get_last(batchsize):
    
    return np.transpose(mnist_white[:,-batchsize:]), labels[-batchsize:,:]





    




from tensorflow.python.framework import ops


def combine_even_odd(even,odd):
    # slow
    #z = tf.transpose(tf.stack([tf.transpose(even), tf.transpose(odd)], axis=0))
    #res = tf.reshape(z,[tf.shape(even)[0],-1])
    
    # fast https://stackoverflow.com/questions/44952886/tensorflow-merge-two-2-d-tensors-according-to-even-and-odd-indices/
    
    res = tf.reshape( tf.concat([even[...,tf.newaxis], odd[...,tf.newaxis]], axis=-1), [tf.shape(even)[0],-1])
    
    return res

 

@tf.RegisterGradient("OPLUGrad")
def oplugrad(op, grad): 
    
    
    
    x = op.inputs[0]
    
    #starting in the same way forward oplu works
    
    even = x[:,::2] #slicing into odd and even parts on the batch
    odd = x[:,1::2]
    
       
    compare = tf.cast(even<odd,dtype=tf.float64)
    compare_not = tf.cast(even>=odd, dtype=tf.float64)
    
    
    #----------
    
    even_grad = grad[:,::2] # slicing gradients
    odd_grad = grad[:,1::2]
    
    
    # OPLU
    
    grad_even_new = odd_grad * compare + even_grad * compare_not
    grad_odd_new = odd_grad * compare_not + even_grad * compare
    
    # inteweave gradients back
    grad_new = combine_even_odd(grad_even_new,grad_odd_new)
           
    
    return grad_new





def tf_oplu(x, name="OPLU"):   
            
    
           
            even = x[:,::2] #slicing into odd and even parts on the batch
            odd = x[:,1::2]            
            
            # OPLU
                      
            compare = tf.cast(even<odd,dtype=tf.float64)
            compare_not = tf.cast(even>=odd,dtype=tf.float64)
            
            #def oplu(x,y): # trivial function  
            #    if x<y : # (x<y)==1
            #       return y, x 
            #    else:
            #       return x, y # (x<y)==0
            
                       
            
            even_new = odd * compare + even * compare_not
            odd_new = odd * compare_not + even * compare
            
            
            # combine into one 
            y = combine_even_odd(even_new,odd_new)            
                      
            
            # https://stackoverflow.com/questions/43256517/how-to-register-a-custom-gradient-for-a-operation-composed-of-tf-operations
            g = tf.get_default_graph()
        
            with g.gradient_override_map({"Identity": "OPLUGrad"}):
            
                y = tf.identity(y, name="oplu")
            
                #y = tf.Print(y, [tf.shape(y)], message = 'debug: ')

            
                # just return what forward layer computes
                return y 




# -------------------------------------





# Parameters

# non -whitening learning rate
#learning_rate = 0.18 # cost function quickly goes to infinity if rate is bigger
learning_rate = 0.05
training_epochs = 25
batch_size = 784
display_step = 1

# Network Parameters (all square matrices)
n_hidden_1 = 300 # 1st layer number of features
n_hidden_2 = 300 # 2nd layer number of features
n_hidden_3 = 300 # 3rd layer number of features
n_hidden_4 = 300 # 4th layer number of features
n_input = 300 # MNIST data input (img shape: 28*28)
n_classes = 10 # MNIST total classes (0-9 digits)

# tf Graph input
x = tf.placeholder("float64", [None, n_input])
y = tf.placeholder("float64", [None, n_classes])


# Create model
def multilayer_perceptron(x, weights, biases):
    
        
    # Hidden layer with RELU activation
    layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
    layer_1 = tf_oplu(layer_1)
    # Hidden layer with RELU activation
    layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
    layer_2 = tf_oplu(layer_2)
    layer_3 = tf.add(tf.matmul(layer_2, weights['h3']), biases['b3'])
    layer_3 = tf_oplu(layer_3)
    layer_4 = tf.add(tf.matmul(layer_3, weights['h4']), biases['b4'])
    layer_4 = tf_oplu(layer_4)
    # Output layer with linear activation
    out_layer = tf.matmul(layer_4, weights['out']) + biases['out']
    
    return out_layer

# Store layers weight & bias
#weights = {
#    'h1': tf.Variable(tf.random_normal([n_input, n_hidden_1])),
#    'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
#    'out': tf.Variable(tf.random_normal([n_hidden_2, n_classes]))
#}
#biases = {
#    'b1': tf.Variable(tf.random_normal([n_hidden_1])),
#    'b2': tf.Variable(tf.random_normal([n_hidden_2])),
#    'out': tf.Variable(tf.random_normal([n_classes]))
#}
#
# orthogonal initialization https://stats.stackexchange.com/questions/228704/how-does-one-initialize-neural-networks-as-suggested-by-saxe-et-al-using-orthogo
def ort_initializer(shape, dtype=tf.float32):
      scale = 1.1
      flat_shape = (shape[0], np.prod(shape[1:]))
      a = np.random.normal(0.1, 0.9, flat_shape)
      u, _, v = np.linalg.svd(a, full_matrices=False)
      # pick the one with the correct shape
      q = u if u.shape == flat_shape else v
      q = q.reshape(shape) #this needs to be corrected to float32
      #print('you have initialized one orthogonal matrix.')
      return tf.constant(scale * q[:shape[0], :shape[1]], dtype=tf.float64)
  
weights = {
    'h1': tf.Variable(ort_initializer([n_input, n_hidden_1]), dtype=tf.float64),
    'h2': tf.Variable(ort_initializer([n_hidden_1, n_hidden_2]), dtype=tf.float64),
    'h3': tf.Variable(ort_initializer([n_hidden_2, n_hidden_3]), dtype=tf.float64),
    'h4': tf.Variable(ort_initializer([n_hidden_3, n_hidden_4]), dtype=tf.float64),
    'out': tf.Variable(ort_initializer([n_hidden_4, n_classes]), dtype=tf.float64)
}
biases = {
    'b1': tf.Variable(tf.zeros([n_hidden_1], dtype=tf.float64), dtype=tf.float64),
    'b2': tf.Variable(tf.zeros([n_hidden_2], dtype=tf.float64), dtype=tf.float64),
    'b3': tf.Variable(tf.zeros([n_hidden_3], dtype=tf.float64), dtype=tf.float64),
    'b4': tf.Variable(tf.zeros([n_hidden_4], dtype=tf.float64), dtype=tf.float64),
    'out': tf.Variable(tf.zeros([n_classes], dtype=tf.float64), dtype=tf.float64)
}


# Construct model
pred = multilayer_perceptron(x, weights, biases)

# Define loss and optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
#optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cost)

# Initializing the variables
init = tf.global_variables_initializer()

# Launch the graph
with tf.Session(config=tf.ConfigProto(
  intra_op_parallelism_threads=8)) as sess:
    sess.run(init)
    
    #### check orthogonality of h2 --------
    p = tf.matmul(weights['h2'], tf.transpose(weights['h2']))
    
    x_b, y_b = get_batch(batch_size)
    
    
    print(p.eval({x: x_b, y: y_b}))
    #print(p.eval({x: mnist.test.images, y: mnist.test.labels}))
    ####------------------------------------

    # Training cycle
    for epoch in range(training_epochs):
        avg_cost = 0.
        total_batch = int(mnist.train.num_examples/batch_size)
        # Loop over all batches
        for i in range(total_batch):
            
            #batch_x, batch_y = mnist.train.next_batch(batch_size)
            batch_x, batch_y = get_batch(batch_size)
            
            #batch_x = zca_whiten(batch_x) # python whitening
            
            # Run optimization op (backprop) and cost op (to get loss value)
            _, c = sess.run([optimizer, cost], feed_dict={x: batch_x,
                                                          y: batch_y})
            # Compute average loss
            avg_cost += c / total_batch
        # Display logs per epoch step
        if epoch % display_step == 0:
            
            print("Epoch:", '%04d' % (epoch+1), "cost=", \
                "{:.9f}".format(avg_cost))
    print("Optimization Finished!")
    
    #### check orthogonality of h2 at the end --------
    p = tf.matmul(weights['h2'], tf.transpose(weights['h2']))
    
    x_b, y_b = get_batch(batch_size)
    print(p.eval({x: x_b, y: y_b}))
    #print(p.eval({x: mnist.test.images, y: mnist.test.labels}))
    #### ---------------------------------------------
    
    # Test model
    correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
    # Calculate accuracy
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    
    x_b, y_b = get_last(batch_size)
    print("Accuracy:", accuracy.eval({x: x_b, y: y_b})) 
    #print("Accuracy:", accuracy.eval({x: mnist.test.images, y: mnist.test.labels})) 
