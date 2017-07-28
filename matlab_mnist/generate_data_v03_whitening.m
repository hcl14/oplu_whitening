clc;
clear;

%K = 512;
K = 784;

disp('Hi!');
images_train = loadMNISTImages('train-images-idx3-ubyte');
labels_train = loadMNISTLabels('train-labels-idx1-ubyte');

images_test = loadMNISTImages('t10k-images-idx3-ubyte');
labels_test = loadMNISTLabels('t10k-labels-idx1-ubyte');

%X1 = single(images_train)';
X1 = images_train';
X1 = X1 + normrnd(0,0.01,[size(X1,1),size(X1,2)]); %add noise

%X2 = single(images_test)';
X2 = images_test';
X2 = X2 + normrnd(0,0.01,[size(X2,1),size(X2,2)]); %add noise


[Xwh, whitemat, avg, sigma, xTilde, D] = stanford_white(X1',0.0001,K);

%X1_wh = single(X1*whitemat);
%X2_wh = single(X2*whitemat);

%X1_wh = single(X1*D');
%X2_wh = single(X2*D');

X1_wh = X1*D';  % ZCA implemented!
X2_wh = X2*D';

if (2 == 1)
    close all;
    figure
    imshow(reshape(X1(1,:), 28, 28));
    figure
    imshow(reshape(X1_wh(1,:), 28, 28));
    error A
end

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
 X1 = X1_wh;
 X2 = X2_wh;
 T1 = T1';
 T2 = T2';
 
 whos X1
 whos T1
 
 cormat = corr(X1);
 save('mnist_wh.mat','X1', 'T1','X2', 'T2');
 
 disp('Done!');
