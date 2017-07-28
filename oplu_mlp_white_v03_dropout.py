from __future__ import print_function

import sys
import random
import numpy as np
import scipy.io as sio
import tensorflow as tf
from tensorflow.python.framework import ops

# Parameters
learning_rate = 0.06 # will be divided by 2 each 10 epochs:
learning_rate_decrease_step = 10 #(epochs)

training_epochs = 200
batch_size = 1000#32
display_step = 1

bsize = tf.constant(batch_size,dtype=tf.int32)

# input whitened mnist
#mat_contents =  sio.loadmat('matlab_mnist/mnist_blank.mat')
mat_contents =  sio.loadmat('matlab_mnist/mnist_wh.mat')

X1 = np.array(mat_contents['X1']) # (784, 60000)
T1 = np.transpose(np.array(mat_contents['T1'])) # (1, 60000) # NOT ONE-HOT!
X2 = np.array(mat_contents['X2']) # (784, 10000)
T2 = np.transpose(np.array(mat_contents['T2'])) # (1, 10000) # NOT ONE-HOT!

print('Shapes:')
print(X1.shape)
print(T1.shape)
print('---------')

ntrain = X1.shape[0]
nbatches = int(ntrain / batch_size)
arrayindices = range(ntrain)

# Network Parameters
n_input = X1.shape[1]
n_classes = T1.shape[1]

N_HIDDEN = 512#256

n_hidden_1 = N_HIDDEN
n_hidden_2 = N_HIDDEN
n_hidden_3 = N_HIDDEN
n_hidden_4 = N_HIDDEN
n_hidden_5 = N_HIDDEN
n_hidden_6 = N_HIDDEN
n_hidden_7 = N_HIDDEN
n_hidden_8 = N_HIDDEN
n_hidden_9 = N_HIDDEN
n_hidden_10 = N_HIDDEN


@tf.RegisterGradient('OPLUGrad')
def oplugrad(op, grad): 
    x = op.inputs[0]
    #starting in the same way forward oplu works
    even = x[:,::2] #slicing into odd and even parts on the batch
    odd = x[:,1::2]
    even_grad = grad[:,::2] # slicing gradients
    odd_grad = grad[:,1::2]
    #compare = tf.cast(even<odd,dtype=tf.float64)
    #compare_not = tf.cast(even>=odd, dtype=tf.float64)
    
    # DROPOUT-------------
    
    # we apply dropout only for training where first dimension of the data equals batch_size
    mask1 = tf.cond(tf.equal(tf.shape(x)[0],bsize), lambda:mask[:,::2], lambda:tf.constant(0,dtype=tf.float64)) # here we have ones
    mask2 = tf.cond(tf.equal(tf.shape(x)[0],bsize), lambda:mask[:,1::2], lambda:tf.constant(1,dtype=tf.float64)) # here we have zeros
            
    #even = tf.Print(even, [tf.shape(mask)], message = 'debug: ')
    compare = tf.cast((even<odd),dtype=tf.float64)  #if compare is 1, elements are permuted
    compare_not = tf.cast((even>=odd),dtype=tf.float64) #if compare_not is 1 instead, elements are not permuted
    
    compare_not = compare_not*mask2 + mask1  #compare_not is set to always 1 here on the places defined by mask
    compare = compare*mask2  # compare is set to always 0 there on the places defined by mask1, which is 0 on even places
    #---------------------
            
    
    
    #---------------------
    
    
    
    
    # OPLU
    grad_even_new = odd_grad * compare + even_grad * compare_not
    grad_odd_new = odd_grad * compare_not + even_grad * compare
    # inteweave gradients back
    grad_new = combine_even_odd(grad_even_new,grad_odd_new)
    
    #grad_new = tf.Print(grad_new, [tf.shape(grad_new)], message = 'debug: ')
    
    return grad_new



def tf_oplu(x, name='OPLU'): 
    
            even = x[:,::2] #slicing into odd and even parts on the batch
            odd = x[:,1::2]
            # OPLU
            
            # DROPOUT-------------
            
            # we apply dropout only for training where first dimension of the data equals batch_size
            mask1 = tf.cond(tf.equal(tf.shape(x)[0],bsize), lambda:mask[:,::2], lambda:tf.constant(0,dtype=tf.float64)) # all zeroes except affected places (=1)
            mask2 = tf.cond(tf.equal(tf.shape(x)[0],bsize), lambda:mask[:,1::2], lambda:tf.constant(1,dtype=tf.float64)) # is all ones except affected places (=0)
                    
            #even = tf.Print(even, [tf.shape(mask)], message = 'debug: ')
            compare = tf.cast((even<odd),dtype=tf.float64)  #if compare is 1, elements are permuted
            compare_not = tf.cast((even>=odd),dtype=tf.float64) #if compare_not is 1 instead, elements are not permuted
            
            compare_not = compare_not*mask2 + mask1  #compare_not is set to always 1 here on the places defined by mask1
            compare = compare*mask2  # compare is set to always 0 there on the places defined by mask2
            #---------------------
            
            
            
            #---------------------
            
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

            with g.gradient_override_map({'Identity': 'OPLUGrad'}):
                y = tf.identity(y, name='oplu')
                #y = tf.Print(y, [tf.shape(y)], message = 'debug: ')
                # just return what forward layer computes                
                return y

def combine_even_odd(even, odd):
    # slow
    # z = tf.transpose(tf.stack([tf.transpose(even), tf.transpose(odd)], axis=0))
    # res = tf.reshape(z,[tf.shape(even)[0],-1])
    # fast https://stackoverflow.com/questions/44952886/tensorflow-merge-two-2-d-tensors-according-to-even-and-odd-indices/
    res = tf.reshape(tf.concat([even[..., tf.newaxis], odd[..., tf.newaxis]], axis=-1), [tf.shape(even)[0], -1])
    return res

def get_batch_train(idx):
    return X1[idx, :], T1[idx, :]

# tf Graph input
x = tf.placeholder('float64', [None, n_input])
y = tf.placeholder('float64', [None, n_classes])




# DROPOUT: -------------------



def compute_np_mask():
    #print('Entered')

    #create binary mask of the corresponding shape with zeros in specific places
    indices = np.array(range(N_HIDDEN)) 

    # select 20 indices to be affected ...
    indices_to_shuffle = 15

    to_be_affected = np.random.permutation(indices)[0:indices_to_shuffle]

    to_be_affected = (to_be_affected/2) # select corresponding pairs

    #initialize binary mask
    row_mask = np.ones(N_HIDDEN, dtype=bool)  # ones mean unaffected rows

    row_mask[::2] = 0 # zero mask on even places everywhere. Row_mask[1::2] = 1 still
    
    #set those pairs to never permute.

    # This is kinda artificial. Actually, these two are two separate masks. See how it is working in the code of ofrward pass
    row_mask[::2][to_be_affected] = 1 # first mask is zet to 1
    row_mask[1::2][to_be_affected] = 0 #second mask is 0
    
    
    
    # populate over all batches
    res = np.tile(row_mask,(batch_size,1))
    
    return res.astype(np.float64)




# So we have mask with all ones and zeros in the places where there should be no permutation ever

# We can use this mask to combine two matrices: permuted and unpermuted

# modified = permuted*(mask) + unpermuted*(!mask)

# (but things a little bit more complicated because of even and odd parts)



np_mask = tf.placeholder('float64',[batch_size,N_HIDDEN]) # DROPOUT: compute new mask each batch-------------------------
mask = tf.cast(np_mask,dtype=tf.float64) #compatibility. You can set identity operation here, just to have placeholder called, but tensorflow complains about format sometimes
#------------------------------




# Create model
def mlp_OPLU(x, weights, biases):
    
    
    
    
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

'''
def mlp_OPLU2(x, weights, biases):
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


def mlp_OPLU_10(x, weights, biases):
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
    layer_5 = tf.add(tf.matmul(layer_4, weights['h5']), biases['b5'])
    layer_5 = tf_oplu(layer_5)
    layer_6 = tf.add(tf.matmul(layer_5, weights['h6']), biases['b6'])
    layer_6 = tf_oplu(layer_6)
    layer_7 = tf.add(tf.matmul(layer_6, weights['h7']), biases['b7'])
    layer_7 = tf_oplu(layer_7)
    layer_8 = tf.add(tf.matmul(layer_7, weights['h8']), biases['b8'])
    layer_8 = tf_oplu(layer_8)
    layer_9 = tf.add(tf.matmul(layer_8, weights['h9']), biases['b9'])
    layer_9 = tf_oplu(layer_9)
    layer_10 = tf.add(tf.matmul(layer_9, weights['h10']), biases['b10'])
    layer_10 = tf_oplu(layer_10)

    # Output layer with linear activation
    out_layer = tf.matmul(layer_10, weights['out']) + biases['out']
    return out_layer

'''

# orthogonal initialization https://stats.stackexchange.com/questions/228704/how-does-one-initialize-neural-networks-as-suggested-by-saxe-et-al-using-orthogo
def ort_initializer(shape, dtype=tf.float64):
      scale = 1.1
      flat_shape = (shape[0], np.prod(shape[1:]))
      a = np.random.normal(0.1, 0.9, flat_shape)
      u, _, v = np.linalg.svd(a, full_matrices=False)
      # pick the one with the correct shape
      q = u if u.shape == flat_shape else v
      q = q.reshape(shape) #this needs to be corrected to float64
      #print('you have initialized one orthogonal matrix.')
      return tf.constant(scale * q[:shape[0], :shape[1]], dtype=tf.float64)

weights = {
    'h1': tf.Variable(ort_initializer([n_input, n_hidden_1]), dtype=tf.float64),
    'h2': tf.Variable(ort_initializer([n_hidden_1, n_hidden_2]), dtype=tf.float64),
    'h3': tf.Variable(ort_initializer([n_hidden_2, n_hidden_3]), dtype=tf.float64),
    'h4': tf.Variable(ort_initializer([n_hidden_3, n_hidden_4]), dtype=tf.float64),
    'h5': tf.Variable(ort_initializer([n_hidden_4, n_hidden_5]), dtype=tf.float64),
    'h6': tf.Variable(ort_initializer([n_hidden_5, n_hidden_6]), dtype=tf.float64),
    'h7': tf.Variable(ort_initializer([n_hidden_6, n_hidden_7]), dtype=tf.float64),
    'h8': tf.Variable(ort_initializer([n_hidden_7, n_hidden_8]), dtype=tf.float64),
    'h9': tf.Variable(ort_initializer([n_hidden_8, n_hidden_9]), dtype=tf.float64),
    'h10': tf.Variable(ort_initializer([n_hidden_9, n_hidden_10]), dtype=tf.float64),
    'out': tf.Variable(ort_initializer([n_hidden_10, n_classes]), dtype=tf.float64)
}
biases = {
    'b1': tf.Variable(tf.zeros([n_hidden_1], dtype=tf.float64), dtype=tf.float64),
    'b2': tf.Variable(tf.zeros([n_hidden_2], dtype=tf.float64), dtype=tf.float64),
    'b3': tf.Variable(tf.zeros([n_hidden_3], dtype=tf.float64), dtype=tf.float64),
    'b4': tf.Variable(tf.zeros([n_hidden_4], dtype=tf.float64), dtype=tf.float64),
    'b5': tf.Variable(tf.zeros([n_hidden_5], dtype=tf.float64), dtype=tf.float64),
    'b6': tf.Variable(tf.zeros([n_hidden_6], dtype=tf.float64), dtype=tf.float64),
    'b7': tf.Variable(tf.zeros([n_hidden_7], dtype=tf.float64), dtype=tf.float64),
    'b8': tf.Variable(tf.zeros([n_hidden_8], dtype=tf.float64), dtype=tf.float64),
    'b9': tf.Variable(tf.zeros([n_hidden_9], dtype=tf.float64), dtype=tf.float64),
    'b10': tf.Variable(tf.zeros([n_hidden_10], dtype=tf.float64), dtype=tf.float64),
    'out': tf.Variable(tf.zeros([n_classes], dtype=tf.float64), dtype=tf.float64)
}

pred = mlp_OPLU(x, weights, biases)
#pred = mlp_OPLU_10(x, weights, biases)
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cost)

init = tf.global_variables_initializer()

correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, 'float'))

with tf.Session(config=tf.ConfigProto(
  intra_op_parallelism_threads=8)) as sess:
    sess.run(init)
        
        
    # Training cycle
    for epoch in range(training_epochs):
        avg_cost_train = 0.
        random.shuffle(arrayindices)

        # Loop over all batches
        for nb in range(nbatches):
            

            
            idx = arrayindices[nb*batch_size:(nb+1)*batch_size]
            batch_x, batch_y = get_batch_train(idx)
            _, c = sess.run([optimizer, cost], feed_dict={x: batch_x, y: batch_y, np_mask: compute_np_mask()}) # DROPOUT ADDED!
            avg_cost_train += c / nbatches
            
        if epoch % learning_rate_decrease_step == 0:
            learning_rate = float(learning_rate)/2  #decrease learning rate each 10 epochs

        # Display logs per epoch step
        if epoch % display_step == 0:
            accuracy_train_val = (accuracy.eval({x: X1, y: T1, np_mask: compute_np_mask()})) * 100
            accuracy_test_val = (accuracy.eval({x: X2, y: T2, np_mask: compute_np_mask()})) * 100
            
            print('Epoch: %d, cost_train = %1.9f, accuracy_train = %1.2f%%, accuracy_test = %1.2f%%' % (epoch + 1, avg_cost_train, accuracy_train_val, accuracy_test_val))

    print('Optimization Finished!')

    #### 
    print('check orthogonality of inner layer h2 at the end')
    p = tf.matmul(weights['h2'], tf.transpose(weights['h2']))
    
    print(p.eval({x: batch_x, y: batch_y})) # just take last batch, do not generate anything
    #print(p.eval({x: mnist.test.images, y: mnist.test.labels}))
    ####
    
    
    sys.exit()

    p = tf.matmul(weights['h2'], tf.transpose(weights['h2']))

    x_b, y_b = get_batch_train(batch_size) 
    print(p.eval({x: x_b, y: y_b, np_mask: compute_np_mask()})) # # DROPOUT ADDED! (BUT NOT USED HERE)
    correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, 'float'))
    
    #x_b, y_b = get_last(batch_size)
    #print('Accuracy:', accuracy.eval({x: x_b, y: y_b}))
    ##print('Accuracy:', accuracy.eval({x: mnist.test.images, y: mnist.test.labels}))
