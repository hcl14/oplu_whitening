% Data:
% http://www.cs.toronto.edu/~tijmen/affNIST/


% Change the current folder to the folder of this m-file.
if(~isdeployed)
  cd(fileparts(which(mfilename)));
end



%%%% Load training data

%create variables
load('training_batches/1.mat') % load the affNIST data variable

all_images_train = affNISTdata.image;
all_labels_train = affNISTdata.label_one_of_n;

%loop all files
for i = 2:32 
    
    load(strcat('training_batches/',int2str(i),'.mat')) % load new affNIST data variable
    
    all_images_train = [all_images_train,affNISTdata.image];
    all_labels_train = [all_labels_train,affNISTdata.label_one_of_n];
    
    
end


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

%switch to 0-1 No! Transfer as Uint8
%all_images_train = single(all_images_train)/255.0;
all_images_val = single(all_images_val)/255.0;


% save data, '-v7.3' is because it is bigger than 2GB in memory
save('affNIST.mat','all_images_train','all_labels_train','all_images_val','all_labels_val','-v7.3')
