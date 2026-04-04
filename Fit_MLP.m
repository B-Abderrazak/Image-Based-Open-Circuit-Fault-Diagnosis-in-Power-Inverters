function [Acc, YPred, probs] = Fit_MLP(Xtrain, Ytrain, Xvalid, Yvalid)

% Reproducibility
% rng(1);

% Basic info
numClasses  = numel(categories(Ytrain));
featureSize = size(Xtrain, 3);

%% ===== Network Architecture =====
layers = [
    imageInputLayer([1 1 featureSize], 'Name','input')

    fullyConnectedLayer(256, 'Name','fc1')
    batchNormalizationLayer('Name','bn1')
    reluLayer('Name','relu1')
    dropoutLayer(0.2, 'Name','drop1')

    fullyConnectedLayer(128, 'Name','fc2')
    reluLayer('Name','relu2')
    dropoutLayer(0.2, 'Name','drop2')

    fullyConnectedLayer(numClasses, 'Name','fc_out')
    softmaxLayer('Name','softmax')
    classificationLayer('Name','output')
];

%% ===== Training Options (Improved) =====
options = trainingOptions('adam', ...
    'MaxEpochs', 10, ...
    'MiniBatchSize', 64, ...
    'InitialLearnRate', 1e-4, ...
    'Shuffle','every-epoch', ...
    'ValidationData',{Xvalid, Yvalid}, ...
    'ValidationFrequency', 20, ...
    'ValidationPatience', 10, ...   % EARLY STOPPING ✅
    'L2Regularization', 1e-4, ...
    'Verbose', false, ...
    'Plots','training-progress');

%% ===== Train Network =====
net = trainNetwork(Xtrain, Ytrain, layers, options);

%% ===== Prediction =====
[YPred, probs] = classify(net, Xvalid);

%% ===== Accuracy =====
Acc = mean(YPred == Yvalid);

end
