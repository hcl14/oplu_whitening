% Data:
% http://www.cs.toronto.edu/~tijmen/affNIST/


% Change the current folder to the folder of this m-file.
if(~isdeployed)
  cd(fileparts(which(mfilename)));
end

clear; % in case we are running the script second time

%%%% Load training data

%create variables
%load('training_batches/1.mat') % load the affNIST data variable
load('centered/training.mat') % load the MNIST data variable

all_images_train = affNISTdata.image;
all_labels_train = affNISTdata.label_one_of_n;

%loop all files
%for i = 2:32 
    
%    load(strcat('training_batches/',int2str(i),'.mat')) % load new affNIST data variable
    
%    all_images_train = [all_images_train,affNISTdata.image];
%    all_labels_train = [all_labels_train,affNISTdata.label_one_of_n];
    
    
%end


%%%%% Load validation data

load('centered/validation.mat')

all_images_val = affNISTdata.image;
all_labels_val = affNISTdata.label_one_of_n;


% can't load batches: python will have out-of-memory when computing
% accuracy

%create variables
%load('validation_batches/1.mat') % load the affNIST data variable

%all_images_val = affNISTdata.image;
%all_labels_val = affNISTdata.label_one_of_n;

%loop all files
%for i = 2:32
    
%    load(strcat('validation_batches/',int2str(i),'.mat')) % load new affNIST data variable
    
%    all_images_val = [all_images_val,affNISTdata.image];
%    all_labels_val = [all_labels_val,affNISTdata.label_one_of_n];
    
    
%end

%switch to 0-1 No, we can't do that! We will transfer it (9GB) to python, then np.array() would attempt to copy it, so 9+9=18>16 ==> we'll be out of memory. 
% We transfer as Uint8, convert each batch to float32 on demand, also we will pass whitening matrix. 
all_images_val = single(all_images_val)/255.0;

% save data, '-v7.3' is because it is bigger than 2GB in memory
save('affNIST.mat','all_images_train','all_labels_train','all_images_val','all_labels_val','-v7.3')


%%----------------- whitening matrix
%calulate whitening matrix (1600x1600) to be transferred into python

%convert trainig data to sigle (double would be 19GB, we can't afford it here)
x = single(all_images_train)/255.0;

%clear memory
clear('all_images_train');
clear('all_images_val');
clear('all_labels_train');
clear('all_labels_val');

%http://ufldl.stanford.edu/wiki/index.php/Implementing_PCA/Whitening

avg = mean(x, 2);     % Compute the mean pixel intensity value separately for each pixel across all patches
%x = x - repmat(avg, 1, size(x, 2)); %out of memory
%x = bsxfun(@minus,x,avg); % out of memory

% straight ant brute
for i = 1:size(x,2)
    x(:,i) = x(:,i) - avg;    
end


sigma = x * x' / size(x, 2);


[U,S,V] = svd(sigma);



svect = diag(S);

% use only k principal components
k = size(svect,1);
epsilon = 0.0001;

%PCA whitening matrix
whitemat = diag(1./sqrt(svect(1:k) + epsilon)) * U(:,1:k)';

% correlation matrix:
%corrmat = corr(whitemat*x); %cannot compute coorelation matrix: out of memory
%%----------------------------------

% save whitening matrix
%save('whitemat.mat','whitemat');
