clc;
clear;

K = 1600;

epochs=2;%249;

whitemat = zeros(K,K);

for i = 0:epochs
    
    filename = strcat(strcat('../augdata/augmented',num2str(i)),'.h5');
    disp(filename)

    images_train = h5read(filename,'/all_images_train');

    %X1 = single(images_train)';
    X1 = images_train';
    X1 = X1 + normrnd(0,0.01,[size(X1,1),size(X1,2)]); %add noise

    %X2 = single(images_test)';
    %X2 = images_test';
    %X2 = X2 + normrnd(0,0.01,[size(X2,1),size(X2,2)]); %add noise


    [Xwh, whitemat, avg, sigma, xTilde, D] = stanford_white(X1',0.0001,K);

    %X1_wh = single(X1*whitemat);
    
    %X1_wh = single(X1*D');
    
    %X1_wh = X1*D';  % ZCA implemented!
    
    
    whitemat = whitemat + (1.0/(epochs+1)*D');
end    

 % for correct loading in python
 
 X1_wh = X1*whitemat;  % ZCA implemented!
 
 cormat = corr(X1_wh);
 save('whitemat.mat','whitemat');
 
 disp('Done!');
