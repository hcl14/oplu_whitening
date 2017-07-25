% Change the filenames if you've saved the files under different names
% On some platforms, the files might be saved as 
% train-images.idx3-ubyte / train-labels.idx1-ubyte


images = loadMNISTImages('train-images-idx3-ubyte');
labels = loadMNISTLabels('train-labels-idx1-ubyte');



x = images; %(:,1:1000);

[Xwh, whitemat, avg, sigma,xTilde] = stanford_white(x,0.0001);

cm = corr(xTilde');

cormat2 = corr(Xwh');

%cormat = corr(Xwh);




% save variables to .mat file

batches = size(xTilde,2);

labelsmat = zeros(10,batches);

for i =1:batches 
    labelsmat(labels(i,1)+1,i) = 1;
end

save('mnist.mat','xTilde', 'labelsmat')
