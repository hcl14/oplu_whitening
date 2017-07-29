#!/usr/bin/env bash 

#mkdir training_batches

mkdir centered


wget http://www.cs.toronto.edu/~tijmen/affNIST/32x/transformed/training_batches.zip   #AUGMENTED DATA
wget http://www.cs.toronto.edu/~tijmen/affNIST/32x/originals/validation.mat.zip       #ORIGINALS

unzip training_batches.zip
unzip validation.mat.zip -d centered

matlab -nodisplay -nodesktop -r "run load_distorted.m; quit();"
