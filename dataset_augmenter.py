from __future__ import print_function

import sys
import random
import numpy as np
import scipy.io as sio


import h5py #matlab v7.3 files


import datumio.datagen as dtd



epochs = 50


# load matlab

mat_contents =  h5py.File('matlab_mnist/affNIST.mat')


X1 = np.array(mat_contents['all_images_train'],dtype=np.uint8) # (1600000,1600) UINT8
T1 = np.array(mat_contents['all_labels_train'],dtype=np.float32) # (1600000, 10)
#X2 = np.array(mat_contents['all_images_val'],dtype=np.float32) # (10000, 1600) FLOAT32
#T2 = np.array(mat_contents['all_labels_val'],dtype=np.float32) # (10000, 10) 

batch_size=100#X1.shape[0]

#load augmenter
rng_aug_params = {'rotation_range': (-20, 20),
                  'translation_range': (-4, 4),
                  'do_flip_lr': False}



'''
X: iterable, ndarray
        Dataset to generate batch from.
        X.shape must be (dataset_length, height, width, channels)
    y: iterable, ndarray, default=None
        Corresponding labels to dataset. If label is None, get_batch will
        only return minibatches of X. y.shape = (data, ) or
        (data, one-hot-encoded)
'''



batch_generator = dtd.BatchGenerator(
                     X1.reshape(-1, 40, 40, 1),
                     y=T1, rng_aug_params=rng_aug_params)





#augment: create one big batch with random transformations and shuffling

for epoch in range(epochs):
    
    
    new_batch_x = None
    new_Batch_y = None
    
    for i in range(1): # we need more: 50.000*20 = 1.000.000     
        
        
        
        for batch in batch_generator.get_batch(batch_size=batch_size, shuffle=True):
    
            batch_x = np.array(batch[0].reshape(batch_size,1600),dtype=np.float32)/255.0  #float32    
            batch_y = batch[1]
            #batch_x = X1
            #batch_y = T1
            
            if new_batch_x is None:
                new_batch_x = batch_x
                new_batch_y = batch_y
            else:
                new_batch_x = np.vstack((new_batch_x,batch_x.copy()))
                new_batch_y = np.vstack((new_batch_y,batch_y.copy()))
            
    
    print('computed' + str(epoch) + ' ' + str(new_batch_x.shape))
    
        #save

    h5f = h5py.File('augdata/augmented'+str(epoch)+'.h5', 'w') 
    h5f.create_dataset('all_images_train', data=new_batch_x)
    h5f.create_dataset('all_labels_train', data=new_batch_y)
    h5f.close()