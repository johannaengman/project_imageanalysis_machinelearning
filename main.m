clear;
clc;

% State chosen image size
sizex = 100;
sizey = 100;

% Read and crop images from each folder
I1 = readim(dir('benign'),sizex,sizey);
I2 = readim(dir('G3'),sizex,sizey);
I3 = readim(dir('G4'),sizex,sizey);
I4 = readim(dir('G5'),sizex,sizey);

%%
% 25% of the data should be testdata
t1 = round(size(I1,4)/4);
t2 = round(size(I2,4)/4);
t3 = round(size(I3,4)/4);
t4 = round(size(I4,4)/4);

test_im = zeros(sizey,sizex,1,t1+t2+t3+t4);
test_im(:,:,1,1:t1) = I1(:,:,1,1:t1);
test_im(:,:,1,t1+1:t1+t2) = I2(:,:,1,1:t2);
test_im(:,:,1,t1+t2+1:t1+t2+t3) = I3(:,:,1,1:t3);
test_im(:,:,1,t1+t2+t3+1:t1+t2+t3+t4) = I4(:,:,1,1:t4);
test_classes = categorical([1*ones(t1,1); 2*ones(t2,1); 3*ones(t3,1); 4*ones(t4,1)]);

train_im = zeros(sizey,sizex,1,size(I1,4)+size(I2,4)+size(I3,4)+size(I4,4)-(t1+t2+t3+t4));
train_im(:,:,1,1:size(I1,4)-t1) = I1(:,:,1,t1+1:end);
train_im(:,:,1,size(I1,4)-t1+1:size(I1,4)-t1+size(I2,4)-t2) = I2(:,:,1,t2+1:end);
train_im(:,:,1,size(I1,4)-t1+size(I2,4)-t2+1:size(I1,4)-t1+size(I2,4)-t2+size(I3,4)-t3) = I3(:,:,1,t3+1:end);
train_im(:,:,1,size(I1,4)-t1+size(I2,4)-t2+size(I3,4)-t3+1:size(I1,4)-t1+size(I2,4)-t2+size(I3,4)-t3+size(I4,4)-t4) = I4(:,:,1,t4+1:end);
train_classes = categorical([1*ones(length(I1)-t1,1); 2*ones(length(I2)-t2,1); 3*ones(length(I3)-t3,1); 4*ones(length(I4)-t4,1)]);

layers = [
    imageInputLayer([sizey sizex 1])
    convolution2dLayer(3,64,'Padding',1)
    batchNormalizationLayer
    reluLayer
    
    maxPooling2dLayer(2,'Stride',2)
    convolution2dLayer(3,32,'Padding',1)
    batchNormalizationLayer
    reluLayer
    
    maxPooling2dLayer(2,'Stride',2)
    convolution2dLayer(3,16,'Padding',1)
    batchNormalizationLayer
    reluLayer
    
    maxPooling2dLayer(2,'Stride',2)
    convolution2dLayer(3,32,'Padding',1)
    batchNormalizationLayer
    reluLayer
    
    maxPooling2dLayer(2,'Stride',2)
    convolution2dLayer(3,64,'Padding',1)
    batchNormalizationLayer
    reluLayer
    
    fullyConnectedLayer(4)
    softmaxLayer
    classificationLayer];

%% Train deep learning network

miniBatchSize = 512;
max_epochs = 60;
learning_rate = 0.001;
options = trainingOptions( 'sgdm',...
    'MaxEpochs',max_epochs,...
    'MiniBatchSize', miniBatchSize,...
    'InitialLearnRate',learning_rate, ...
    'Plots', 'training-progress');

net = trainNetwork(train_im, train_classes, layers, options);

%% Test the classifier on the test set

[Y_result2,scores2] = classify(net,test_im);
accuracy2 = sum(Y_result2 == test_classes)/numel(Y_result2);
disp(['The accuracy on the test set: ' num2str(accuracy2)]);

%% Test the classifier on the training set

[Y_result1,scores1] = classify(net,train_im);
accuracy1 = sum(Y_result1 == train_classes)/numel(Y_result1);
disp(['The accuracy on the training set: ' num2str(accuracy1)]);

