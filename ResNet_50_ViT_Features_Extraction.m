clc;close all;clear;
modelType = 1; % 1 = ResNet50, 2 = ViT
datasetPath = '100';
% datasetPath= 'C:\Users\HP\Documents\tam\TAM_2026_close';
imds = imageDatastore(datasetPath,'IncludeSubfolders',true,'LabelSource','foldernames'); 
switch modelType
    case 1
fprintf('Features extraction using ResNet50\n');
numClasses = numel(categories(imds.Labels));
net = resnet50;
featureLayer = 'avg_pool';
inputSize = net.Layers(1).InputSize;
aug_Res_50 = augmentedImageDatastore(inputSize(1:2),imds,'ColorPreprocessing','gray2rgb'); % NO augmentation
features_res_50 = activations(net, aug_Res_50, featureLayer);
labels=imds.Labels;
save features_res_close labels  features_res_50
    case 2
 fprintf('Features extraction using ViT\n');
net = visionTransformer("tiny-16-imagenet-384");
inputSize = net.Layers(1).InputSize;
featureExtractor = layerGraph(net);
featureExtractor = removeLayers(featureExtractor, { 'softmax'});
featureExtractorNet = dlnetwork(featureExtractor);
classNames = categories(imds.Labels);
numClasses = numel(categories(imds.Labels))
aug_ViT = augmentedImageDatastore(inputSize(1:2),imds,'ColorPreprocessing','gray2rgb'); % NO augmentation
labels = [];
features_ViT=zeros(1000,aug_ViT.NumObservations);
for idx=1:aug_ViT.NumObservations
 [testData, label] = readByIndex(aug_ViT,idx);
I = testData.input{1};
feature = minibatchpredict(featureExtractorNet,(I));
features_ViT(:,idx) = feature; 
C = categorical(label.Label);
labels=[labels;C]; 
end 
save features_ViT_close labels  features_ViT
    otherwise
  error('Invalid modelType. Use 1 for ResNet50 or 2 for ViT.');
end

