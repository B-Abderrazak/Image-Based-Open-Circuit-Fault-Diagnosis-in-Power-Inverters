clc; clear; close all;

%% Parameters
dataDir= 'C:\Users\HP\Documents\tam\TAM_2026_close';

k = 5;
inputSize = [224 224 3];

%% Load Dataset
imds = imageDatastore(dataDir, ...
    'IncludeSubfolders', true, ...
    'LabelSource', 'foldernames');
%% K-Fold Cross Validation
cv = cvpartition(imds.Labels, 'KFold', k);
numClasses = numel(categories(imds.Labels));

accuracy = zeros(k,1);

for i = 1:1
    fprintf('Fold %d/%d\n', i, k);

    trainIdx = training(cv, i);
    valIdx   = test(cv, i);

    imdsTrain = subset(imds, trainIdx);
    imdsVal   = subset(imds, valIdx);

    %% ONLY Resize (No augmentation)
    augTrain = augmentedImageDatastore(inputSize, imdsTrain, ...
        'ColorPreprocessing','gray2rgb');

    augVal = augmentedImageDatastore(inputSize, imdsVal, ...
        'ColorPreprocessing','gray2rgb');

    %% Load ResNet-50
    net = resnet50;
    lgraph = layerGraph(net);

    %% Replace Final Layers
    newLayers = [
        fullyConnectedLayer(numClasses, 'Name', 'fc_new', ...
            'WeightLearnRateFactor', 10, ...
            'BiasLearnRateFactor', 10)
        softmaxLayer('Name','softmax')
        classificationLayer('Name','classoutput')
    ];

    lgraph = replaceLayer(lgraph, 'fc1000', newLayers(1));
    lgraph = replaceLayer(lgraph, 'fc1000_softmax', newLayers(2));
    lgraph = replaceLayer(lgraph, 'ClassificationLayer_fc1000', newLayers(3));

    %% Training Options
    options = trainingOptions('adam', ...
        'MiniBatchSize', 16, ...
        'MaxEpochs', 5, ...
        'InitialLearnRate', 1e-4, ...
        'Shuffle', 'every-epoch', ...
        'ValidationData', augVal, ...
        'Verbose', false,...
     'Plots','training-progress');

    %% Train
    trainedNet = trainNetwork(augTrain, lgraph, options);

    %% Evaluate
    preds = classify(trainedNet, augVal);
    YVal = imdsVal.Labels;

    accuracy(i) = mean(preds == YVal);
    fprintf('Accuracy Fold %d = %.4f\n', i, accuracy(i));
end

%% Final Results
fprintf('\nMean Accuracy: %.4f\n', mean(accuracy));
fprintf('Std Accuracy : %.4f\n', std(accuracy));