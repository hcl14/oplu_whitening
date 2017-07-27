clc;
clear;

disp('Hi!');

images_train = loadMNISTImages('train-images-idx3-ubyte');
labels_train = loadMNISTLabels('train-labels-idx1-ubyte');

images_test = loadMNISTImages('t10k-images-idx3-ubyte');
labels_test = loadMNISTLabels('t10k-labels-idx1-ubyte');

 %x = images_train(:,1);
 %imshow(reshape(x, 28, 28));
 
 X1 = single(images_train);
 X2 = single(images_test);
 
 T1 = zeros(size(labels_train, 1), 10);
 for k = 1:size(labels_train, 1)
     l = labels_train(k);
     T1(k, l+1) = 1;
 end
 
 T2 = zeros(size(labels_test, 1), 10);
 for k = 1:size(labels_test, 1)
     l = labels_test(k);
     T2(k, l+1) = 1;
 end
 
 % for correct loading in python
 X1 = X1';
 X2 = X2';
 T1 = T1';
 T2 = T2';
 
 whos X1
 whos T1
 
 save('mnist_blank.mat','X1', 'T1','X2', 'T2');
 
 disp('Done!');
