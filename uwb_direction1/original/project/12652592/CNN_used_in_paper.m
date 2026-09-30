
% Used network for classification 
numGestures=12; %Adjust it based on your number of gestures
layers = [imageInputLayer([245 389 3]);
    convolution2dLayer(3,8,'Padding',1)
    batchNormalizationLayer
    reluLayer
    maxPooling2dLayer(2,'Stride',2)
    
    convolution2dLayer(3,16,'Padding',1)
    batchNormalizationLayer
    reluLayer  
    maxPooling2dLayer(2,'Stride',1)
     
    convolution2dLayer(3,32,'Padding',1)
    batchNormalizationLayer
    reluLayer    
    maxPooling2dLayer(2,'Stride',2)
%      
    convolution2dLayer(3,64,'Padding',1)
    batchNormalizationLayer
    reluLayer
    maxPooling2dLayer(4,'Stride',4)
%     
    convolution2dLayer(3,128,'Padding',1)
    batchNormalizationLayer
    reluLayer
    maxPooling2dLayer(2,'Stride',2)
%     
    convolution2dLayer(3,256,'Padding',1)
    batchNormalizationLayer
    reluLayer
%     
    convolution2dLayer(3,2*256,'Padding',1)
    batchNormalizationLayer
    reluLayer    
    maxPooling2dLayer(2,'Stride',2)

    
    fullyConnectedLayer(numGestures)
    softmaxLayer
    classificationLayer];

miniBatchsize = 30;

options = trainingOptions('sgdm', ...
    'MaxEpochs',30, ...
    'LearnRateDropFactor',0.2, ...
    'LearnRateDropPeriod',5, ...   
    'ValidationData',imdsValidation, ...
    'ValidationFrequency',2, ...
    'Verbose',false, ...
    'Plots','training-progress');

% set up the image path according to your directory first and create image
% store with training data in TrGestureData and uncomment below section
% [NeT,tr] = trainNetwork(TrGestureData,layers,options);
% YPred = classify(NeT,TrGestureData);
% YValidation = TrGestureData.Labels;
