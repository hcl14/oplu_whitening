
from __future__ import print_function
import numpy as np
import random
import h5py #matlab v7.3 files

import sys


np.random.seed(1001)

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
    
    
    X2 = np.array(mat_contents['all_images_train'],dtype=np.float32) # (1600000,1600) #already in Float32 from data augmenter!
    
    global X1,T1
    
    X1 = np.random.normal(0,1,X2.shape)
    
    
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
learning_rate = 0.01
learning_rate_decay = 0.98

momentum = 0.3

training_epochs = 20
batch_size = 100
display_step = 1

# Network Parameters
n_hidden_1 = 512 # 1st layer number of neurons
n_hidden_2 = 512 # 2nd layer number of neurons
n_hidden_3 = 512 # 3rd layer
n_hidden_4 = 512
n_hidden_5 = 512
n_hidden_6 = 512
n_hidden_7 = 512

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




# Store layers weight & bias

weights = {
    #'h1': tf.Variable(ort_initializer([n_input+1, n_hidden_1])),  # use this to add biases
    'h1': tf.Variable(ort_initializer([n_input, n_hidden_1])),
    'h2': tf.Variable(ort_initializer([n_hidden_1, n_hidden_2])),
    'h3': tf.Variable(ort_initializer([n_hidden_2, n_hidden_3])),
    'h4': tf.Variable(ort_initializer([n_hidden_3, n_hidden_4])),
    'h5': tf.Variable(ort_initializer([n_hidden_4, n_hidden_5])),
    'h6': tf.Variable(ort_initializer([n_hidden_5, n_hidden_6])),
    'h7': tf.Variable(ort_initializer([n_hidden_6, n_hidden_7])),
    'out': tf.Variable(ort_initializer([n_hidden_7, n_classes]))
}



'''
weights = {
    #'h1': tf.Variable(ort_initializer([n_input+1, n_hidden_1])),  # use this to add biases
    'h1': tf.Variable(tf.eye(n_input, n_hidden_1)),
    'h2': tf.Variable(tf.eye(n_hidden_1, n_hidden_2)),
    'h3': tf.Variable(tf.eye(n_hidden_2, n_hidden_3)),
    'h4': tf.Variable(tf.eye(n_hidden_3, n_hidden_4)),
    'h5': tf.Variable(tf.eye(n_hidden_4, n_hidden_5)),
    'h6': tf.Variable(tf.eye(n_hidden_5, n_hidden_6)),
    'h7': tf.Variable(tf.eye(n_hidden_6, n_hidden_7)),
    'out': tf.Variable(tf.eye(n_hidden_7, n_classes))
}
'''



def combine_even_odd(even, odd):
    res = tf.reshape(tf.concat([even[..., tf.newaxis], odd[..., tf.newaxis]], axis=-1), [tf.shape(even)[0], -1])
    return res

# needed to register custom activation derivative. 
# https://stackoverflow.com/questions/43256517/how-to-register-a-custom-gradient-for-a-operation-composed-of-tf-operations
@tf.RegisterGradient('my_derivative')
def my_derivative(op, grad): 
    
    x = op.inputs[0] # values of the layer: n_batches*layer_size
    
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
    grad_new = tf.Print(grad_new, [op.name, tf.norm(grad_new)], message = 'op grad norm')
    
    
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
    
    
    grad_new = 0.0001*tf.ones(tf.shape(grad), dtype=tf.float32)
    
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



layer_1_z = tf.matmul(x, weights['h1'])

layer_1_z = tf.Print(layer_1_z, [tf.norm(x)], message="** input norm")

layer_1_a = OPLU(layer_1_z)
layer_1_a = tf.Print(layer_1_a, [tf.norm(layer_1_a)], message="** layer_1_a norm ")

layer_1_a = tf.Print(layer_1_a, [tf.norm(tf.matmul(layer_1_z, 0.0001*tf.ones(tf.shape(layer_1_a),dtype=tf.float32), transpose_a=True))])  
layer_1_a = tf.Print(layer_1_a, [tf.norm(tf.matmul(layer_1_a, 0.0001*tf.ones(tf.shape(layer_1_a),dtype=tf.float32), transpose_a=True))])  

#eigenvalues,_ = tf.self_adjoint_eig(weights['h1'])
#layer_1_a = tf.Print(layer_1_a,[tf.reduce_max(eigenvalues)], message="w1 eigenvalues")


layer_2_z = tf.matmul(layer_1_a, weights['h2'])
layer_2_a = OPLU(layer_2_z)
layer_2_a = tf.Print(layer_2_a, [tf.norm(layer_2_a)], message="** layer_2_a norm ")

layer_2_a = tf.Print(layer_2_a, [tf.norm(tf.matmul(layer_2_z, 0.0001*tf.ones(tf.shape(layer_2_a),dtype=tf.float32), transpose_a=True))]) 
layer_2_a = tf.Print(layer_2_a, [tf.norm(tf.matmul(layer_2_a, 0.0001*tf.ones(tf.shape(layer_2_a),dtype=tf.float32), transpose_a=True))]) 


layer_3_z = tf.matmul(layer_2_a, weights['h3'])
layer_3_a = OPLU(layer_3_z)
layer_3_a = tf.Print(layer_3_a, [tf.norm(layer_3_a)], message="** layer_3_a norm ")

layer_3_a = tf.Print(layer_3_a,[ort_discrepancy(weights['h3'])], message="w3 discr")

layer_3_a = tf.Print(layer_3_a, [tf.norm(tf.matmul(layer_3_z, 0.0001*tf.ones(tf.shape(layer_3_a),dtype=tf.float32), transpose_a=True))])  
layer_3_a = tf.Print(layer_3_a, [tf.norm(tf.matmul(layer_3_a, 0.0001*tf.ones(tf.shape(layer_3_a),dtype=tf.float32), transpose_a=True))])  



layer_4_z = tf.matmul(layer_3_a, weights['h4'])
layer_4_a = OPLU(layer_4_z)
layer_4_a = tf.Print(layer_4_a, [tf.norm(layer_4_a)], message="** layer_4_a norm ")

layer_4_a = tf.Print(layer_4_a,[ort_discrepancy(weights['h4'])], message="w4 discr")


layer_4_a = tf.Print(layer_4_a, [tf.norm(tf.matmul(layer_4_z, 0.0001*tf.ones(tf.shape(layer_4_a),dtype=tf.float32), transpose_a=True))])  




layer_5_z = tf.matmul(layer_4_a, weights['h5'])
layer_5_a = OPLU(layer_5_z)
layer_5_a = tf.Print(layer_5_a, [tf.norm(layer_5_a)], message="** layer_5_a norm ")


layer_5_a = tf.Print(layer_5_a,[ort_discrepancy(weights['h5'])], message="w5 discr")


layer_5_a = tf.Print(layer_5_a, [tf.norm(tf.matmul(layer_5_z, 0.0001*tf.ones(tf.shape(layer_5_a),dtype=tf.float32), transpose_a=True))]) 




layer_6_z = tf.matmul(layer_5_a, weights['h6'])
layer_6_a = OPLU(layer_6_z)
layer_6_a = tf.Print(layer_6_a, [tf.norm(layer_6_a)], message="** layer_6_a norm ")


layer_6_a = tf.Print(layer_6_a,[ort_discrepancy(weights['h6'])], message="w6 discr")


layer_6_a = tf.Print(layer_6_a, [tf.norm(tf.matmul(layer_6_z, 0.0001*tf.ones(tf.shape(layer_6_a),dtype=tf.float32), transpose_a=True))])  




layer_7_z = tf.matmul(layer_6_a, weights['h7'])
layer_7_a = OPLU(layer_7_z)
layer_7_a = tf.Print(layer_7_a, [tf.norm(layer_7_a)], message="** layer_7_a norm ")

layer_7_a = tf.Print(layer_7_a,[ort_discrepancy(weights['h7'])], message="w7 discr")


layer_7_a = tf.Print(layer_7_a, [tf.norm(tf.matmul(layer_7_z, 0.0001*tf.ones(tf.shape(layer_7_a),dtype=tf.float32), transpose_a=True))]) 




# Output fully connected layer with a neuron for each class
out_layer = tf.matmul(layer_7_a, weights['out'])
#out_layer = my_activation(out_layer)  # To not have any additional linear layer which evolves in non-orthogonal way. We have also softmax at the end
# training is very slow with oplu on fully connected layer

#return out_layer





# Construct model
logits = out_layer #multilayer_perceptron(X)

# Define loss and optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=Y))

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

d_out = tf.gradients(cost, [out_layer])  # take dE/d_out_layer_a from tensorflow for softmax with logits
#d_out = tf.Print(d_out,[d_out], message = "out grad")
#d_out = tf.Print(d_out,[tf.shape(d_out)], message = "out grad shape")   # shape is [1, batch_size, n_classes] = [1,100,10]
d_out=d_out[0]

# there is no activation for out_layer, so we immediately compute weights
#d_out_layer = tf.multiply(d_out, activation_derivative(out_layer))
d_w_out = tf.matmul(layer_7_a, d_out, transpose_a=True)


def backpropagate_layer(layer_prev_a, layer_z, weights_next, weights_current, d_layer_z_next, name="layer"):
    # layer
    d_layer_a = tf.matmul(d_layer_z_next, weights_next, transpose_b=True)
    d_layer_a = tf.Print(d_layer_a, [tf.norm(d_layer_a)], message="d_"+name+"_a")

    #d_layer_7_z = tf.multiply(d_layer_7_a, activation_derivative(d_layer_7_z)) # activation_derivative is a permutation matrix in case of OPLU
    d_layer_z = oplu_derivative(layer_z, d_layer_a)  # oplu_derivative function is a permutation which acts on d_layer_7_a
    d_layer_z = tf.Print(d_layer_z, [tf.norm(d_layer_z)], message="d_"+name+"_z")
    
    
    d_w = tf.matmul(layer_prev_a, d_layer_z, transpose_a=True)    
    
    d_w = tf.Print(d_w, [tf.shape(d_layer_z),tf.shape(layer_prev_a)])    
    d_w = tf.Print(d_w,[tf.norm(layer_prev_a)], message= "-- prev_layer_a_norm "+name)
    d_w = tf.Print(d_w,[tf.norm(d_layer_z)], message= "-- d_layer_z_norm "+ name)
    
    # project gradients
    #d_w, _ = my_weight_gradient_modification([d_w, weights_current])
    
    
    d_w = tf.Print(d_w,[tf.norm(d_w)], message="***** d_w_"+name)
    
    
    return d_w, d_layer_z


# layer 7
d_w_7, d_layer_7_z = backpropagate_layer(layer_6_a, layer_7_z, weights['out'], weights['h7'], d_out, name="layer_7")
# layer 6
d_w_6, d_layer_6_z = backpropagate_layer(layer_5_a, layer_6_z, weights['h7'], weights['h6'], d_layer_7_z, name="layer_6")
# layer 5
d_w_5, d_layer_5_z = backpropagate_layer(layer_4_a, layer_5_z, weights['h6'], weights['h5'], d_layer_6_z, name="layer_5")
# layer 4
d_w_4, d_layer_4_z = backpropagate_layer(layer_3_a, layer_4_z, weights['h5'], weights['h4'], d_layer_5_z, name="layer_4")
# layer 3
d_w_3, d_layer_3_z = backpropagate_layer(layer_2_a, layer_3_z, weights['h4'], weights['h3'], d_layer_4_z, name="layer_3")
# layer 2
d_w_2, d_layer_2_z = backpropagate_layer(layer_1_a, layer_2_z, weights['h3'], weights['h2'], d_layer_3_z, name="layer_2")
# layer 4
d_w_1, d_layer_1_z = backpropagate_layer(x, layer_1_z, weights['h2'], weights['h1'], d_layer_2_z, name="layer_1")


# force evaluate all the weights
#d_w_1 = tf.Print(d_w_1, [d_w_1,d_w_2,d_w_3,d_w_4,d_w_5,d_w_6,d_w_7,d_w_out], message="force evaluate")
    


optimizer = [
    
    
    tf.assign(weights['h1'],
            tf.subtract(weights['h1'], tf.multiply(tf_learning_rate, d_w_1))),
    
    tf.assign(weights['h2'],
            tf.subtract(weights['h2'], tf.multiply(tf_learning_rate, d_w_2))),
    
    tf.assign(weights['h3'],
            tf.subtract(weights['h3'], tf.multiply(tf_learning_rate, d_w_3))),
    
    tf.assign(weights['h4'],
            tf.subtract(weights['h4'], tf.multiply(tf_learning_rate, d_w_4))),
    
    tf.assign(weights['h5'],
            tf.subtract(weights['h5'], tf.multiply(tf_learning_rate, d_w_5))),
    
    tf.assign(weights['h6'],
            tf.subtract(weights['h6'], tf.multiply(tf_learning_rate, d_w_6))),
    
    tf.assign(weights['h7'],
            tf.subtract(weights['h7'], tf.multiply(tf_learning_rate, d_w_7))),
    
    tf.assign(weights['out'],
            tf.subtract(weights['out'], tf.multiply(tf_learning_rate, d_w_out)))
    
    
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


with tf.Session() as sess:
    sess.run(init)
    
        
    #### 
    print('check orthogonality of inner layer h2 at start')
    p = tf.matmul(weights['h2'], weights['h2'],transpose_a=True)
    
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
        for i in range(5): #range(total_batch):
            
            idx = range(100) #range(1000,1100) #arrayindices[i*batch_size:(i+1)*batch_size]
            batch_x, batch_y = get_batch_train(idx)
            
            #print(cost.eval({X: batch_x, Y: batch_y, tf_learning_rate: learning_rate}))
            
            # Run optimization op (backprop) and cost op (to get loss value)
            _, c = sess.run([optimizer, cost], feed_dict={X: batch_x, Y: batch_y, tf_learning_rate: learning_rate})
            
            print("---")
            # Compute average loss
            avg_cost += c / total_batch
            
        sys.exit()
        # Display logs per epoch step
        if epoch % display_step == 0:
            accuracy_train_val = (accuracy.eval({X: X1, Y: T1})) * 100                   
            accuracy_test_val = (accuracy.eval({X: X2, Y: T2}))*100

            print("Epoch:", '%04d' % (epoch+1), "learning rate: %1.9f"%learning_rate," cost={:.9f}".format(avg_cost),"||W2*W2^t - I||L2 = %1.4f " % ort_discrepancy(weights['h3']).eval(),  "accuracy_train = %1.2f%%" % accuracy_train_val, "accuracy_test = %1.2f%%" % accuracy_test_val)
           
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
    
 
