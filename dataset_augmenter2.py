from __future__ import print_function

import sys
import random
import numpy as np
import scipy.io as sio


import h5py #matlab v7.3 files


import datumio.datagen as dtd


# elastic distortions
from mnist_helpers import *


from multiprocessing import Pool, Process



    


def distort(image):
    
     image1 = image.reshape(40,40)
    
     #distorted_image = elastic_transform(image, kernel_dim=15,
     #                        alpha=5.5,
     #                        sigma=35)
     # just call the function elastic_transform function 
     # with a suitable kernel size, alpha and sigma
     # as a rule of thumb, if use sigma as a value near 6,
     # alpha 36-40, kernel size 13-15
     #
     # NOTE: the input image SHOULD be of square dimension,
     # ie no.of rows should be equal to number of cols.

     # get the transformed image
     distorted_image = elastic_transform(image1, kernel_dim=15,
                                    alpha=36,
                                    sigma=6)
     
     return distorted_image.ravel()



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




def distort_func(batch):
    
    
    # add distortion
    for i in range(batch.shape[0]):                
        batch[i,:] = distort(batch[i,:])
        
    return batch
        

#https://stackoverflow.com/questions/29857498/how-to-apply-a-function-to-a-2d-numpy-array-with-multiprocessing
pool = Pool(4)

def parallel(batch):
    M = batch.shape[0]
    N = batch.shape[1]
    results = pool.map(distort_func, (batch[0:batch.shape[0]/4,:],batch[batch.shape[0]/4 : batch.shape[0]/2,:],batch[batch.shape[0]/2 : 3*batch.shape[0]/4,:],batch[3*batch.shape[0]/4 : batch.shape[0],:]))
    return np.array(results).reshape(M, N)



#augment: create one big batch with random transformations and shuffling

for epoch in range(epochs):
    
    
    new_batch_x = None
    new_Batch_y = None
    
    count=0
    
    for i in range(1): #if we need more: 50.000*20 = 1.000.000     
        
        
        
        for batch in batch_generator.get_batch(batch_size=batch_size, shuffle=True):
            
            count+=1
    
            batch_x = np.array(batch[0].reshape(batch_size,1600),dtype=np.float32)/255.0  #float32    
            batch_y = batch[1]
            #batch_x = X1
            #batch_y = T1
            
            
            
            #add distortion in parallel
            
            batch_x = parallel(batch_x)
            
            print('batch '+str(count)+' processed')
            
            
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