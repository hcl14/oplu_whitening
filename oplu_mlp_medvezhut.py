
from __future__ import print_function
import numpy as np
import random
import h5py #matlab v7.3 files


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

# Parameters
learning_rate = 0.0005
learning_rate_decay = 0.98

training_epochs = 100
batch_size = 100
display_step = 1

# Network Parameters
n_hidden_1 = 2048 # 1st layer number of neurons
n_hidden_2 = 1024 # 2nd layer number of neurons
n_hidden_3 = 512 # 3rd layer
n_input = X1.shape[1] #784 # MNIST data input (img shape: 28*28)
n_classes = T1.shape[1] #10 # MNIST total classes (0-9 digits)







# tf Graph input
X = tf.placeholder("float", [None, n_input])
Y = tf.placeholder("float", [None, n_classes])
tf_learning_rate = tf.placeholder(tf.float32, shape=[])



# function  that initializes orthogonal matrix
def ort_initializer(shape, dtype=tf.float32):
    
      
      scale = 1.0
      flat_shape = (shape[0], np.prod(shape[1:]))
      a = np.random.normal(0, 1, flat_shape)
      u, _, v = np.linalg.svd(a, full_matrices=False)
      # pick the one with the correct shape
      q = u if u.shape == flat_shape else v
      q = q.reshape(shape) #this needs to be corrected to float32
      #print('you have initialized one orthogonal matrix.')
      mat=tf.constant(scale * q[:shape[0], :shape[1]], dtype=tf.float32)
      
      i = tf.matmul(mat,mat,transpose_b=True)
      
      scale = 1.0/tf.sqrt(i[0,0])
      
      return scale*mat




# Store layers weight & bias
weights = {
    'h1': tf.Variable(ort_initializer([n_input+1, n_hidden_1])),
    'h2': tf.Variable(ort_initializer([n_hidden_1, n_hidden_2])),
    'h3': tf.Variable(ort_initializer([n_hidden_2, n_hidden_3])),
    'out': tf.Variable(ort_initializer([n_hidden_3, n_classes]))
}


def combine_even_odd(even, odd):
    res = tf.reshape(tf.concat([even[..., tf.newaxis], odd[..., tf.newaxis]], axis=-1), [tf.shape(even)[0], -1])
    return res

# needed to register custom activation derivative. Does not seem to be actual gradient
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
    
    # we can check the shape this way
    # grad_new = tf.Print(grad_new, [tf.shape(grad_new)], message = 'debug: ')
    
    
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

So in your case your g(x) could be an identity op, with a custom gradient using gradient_override_map
'''


def my_activation(x, name='my_activation'): 
    
    # masking OPLU from gradient computation
    t = tf.identity(x)
    y = t + tf.stop_gradient(OPLU(x) - t)
    
    
    # Applying custom gradient
    
    # https://stackoverflow.com/questions/43256517/how-to-register-a-custom-gradient-for-a-operation-composed-of-tf-operations
    
    # hack to apply custom activation derivative - it is not the same size as weight matrix, but a layer-size vector (incl. batches), so name 'gradient' does not seem correct
    
    g = tf.get_default_graph()

    with g.gradient_override_map({'Identity': 'my_derivative'}):
        y = tf.identity(y, name='my_activ')
        return y
    #return y




def my_weight_gradient_modification(grad_var_tuple):
    
    
    grad = grad_var_tuple[0]
    variable = grad_var_tuple[1]
    
    #example:
    #grad_new = 0.8*grad
    
    # projection to the tangent supspace
    # costy! slows down computation
    grad_new = tf.matmul(tf.matmul(grad,variable,transpose_b=True) - tf.matmul(variable,grad,transpose_b=True), variable)
    
    return grad_new, variable



# Create model
def multilayer_perceptron(x):
    
    
    # bias replacements
    ones = tf.ones([tf.shape(x)[0],1],tf.float32)

    
    
    #adding biases
    #input shape is batch_size*input, so we just add batch_size column of ones
    x = tf.concat([x,ones], 1)
    
    
    # Hidden fully connected layer
    layer_1 = tf.matmul(x, weights['h1'])
    layer_1 = my_activation(layer_1)
    
    #adding biases
    #layer shape is batch_size*layer, so we just add batch_size column of ones
    #layer_1 = tf.concat([layer_1,ones], 1)
    
    # Hidden fully connected layer
    layer_2 = tf.matmul(layer_1, weights['h2'])
    layer_2 = my_activation(layer_2)
    #layer_2 = tf.concat([layer_2,ones], 1)
    
    
    
    # Hidden fully connected layer
    layer_3 = tf.matmul(layer_2, weights['h3'])
    layer_3 = my_activation(layer_3)
    #layer_3 = tf.concat([layer_3,ones], 1)
    
    
    
    # Output fully connected layer with a neuron for each class
    out_layer = tf.matmul(layer_3, weights['out'])
    return out_layer





# Construct model
logits = multilayer_perceptron(X)

# Define loss and optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=Y))
opt = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)

#optimizer = opt.minimize(cost)

##############################################
## Here I want to do manual gradient modification
##############################################


# Compute the gradients for a list of variables I am interested in.
# Those are true weight gradients:
grads_and_vars = opt.compute_gradients(cost, [weights['h1'],weights['h2'],weights['h3'],weights['out']])
# grads_and_vars is a list of tuples (gradient, variable).  Do whatever you
# need to the 'gradient' part, for example cap them, etc.

# project weights
grads_and_vars[0]=my_weight_gradient_modification(grads_and_vars[0])
grads_and_vars[1]=my_weight_gradient_modification(grads_and_vars[1])
grads_and_vars[2]=my_weight_gradient_modification(grads_and_vars[2])
grads_and_vars[3]=my_weight_gradient_modification(grads_and_vars[3])



# Ask the optimizer to apply the modified gradients.
optimizer = opt.apply_gradients(grads_and_vars)

##############################################




# Test model
pred = tf.nn.softmax(logits)  # Apply softmax to logits


#pred = tf.Print(pred, [tf.shape(grads_and_vars[0]),tf.shape(grads_and_vars[1]),tf.shape(grads_and_vars[4])])

correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(Y, 1))
# Calculate accuracy
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))




# Initializing the variables
init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    
    
    #### 
    print('check orthogonality of inner layer h2 at start')
    p = tf.matmul(weights['h2'], tf.transpose(weights['h2']))
    
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
            # Compute average loss
            avg_cost += c / total_batch
        # Display logs per epoch step
        if epoch % display_step == 0:
            accuracy_train_val = (accuracy.eval({X: X1, Y: T1})) * 100                   
            accuracy_test_val = (accuracy.eval({X: X2, Y: T2}))*100

            print("Epoch:", '%04d' % (epoch+1), "learning rate: %1.9f"%learning_rate," cost={:.9f}".format(avg_cost),"accuracy_train = %1.2f%%" % accuracy_train_val, "accuracy_test = %1.2f%%" % accuracy_test_val)
            
        learning_rate = learning_rate*learning_rate_decay  #manually decay learning rate



    print("Optimization Finished!")

    #### 
    print('check orthogonality of inner layer h2 at the end')
    p = tf.matmul(weights['h2'], tf.transpose(weights['h2']))
    
    print(p.eval()) # just take last batch, do not generate anything
    #print(p.eval({x: mnist.test.images, y: mnist.test.labels}))
    ####
    
