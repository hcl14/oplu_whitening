#!/usr/bin/env bash 

#mkdir training_batches

mkdir centered


#wget http://www.cs.toronto.edu/~tijmen/affNIST/32x/transformed/training_batches.zip   #AUGMENTED DATA
wget http://www.cs.toronto.edu/~tijmen/affNIST/32x/just_centered/validation.mat.zip      #ORIGINALS
wget http://www.cs.toronto.edu/~tijmen/affNIST/32x/just_centered/training.mat.zip      #ORIGINALS


#unzip training_batches.zip
unzip training.mat.zip -d centered
unzip validation.mat.zip -d centered

matlab -nodisplay -nodesktop -r "run load_distorted.m; quit();"
