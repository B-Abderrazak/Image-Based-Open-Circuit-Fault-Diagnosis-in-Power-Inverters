%% EDA.m - Exploratory Data Analysis for Inverter Fault Dataset
clc; clear; close all;

% Dataset Path (EDIT THIS)
 datasetPath = 'Data\Full_Current_Cycle'; % <-- change if needed

%Load Classes
classes = dir(datasetPath);
classes = classes([classes.isdir]);
classes = classes(3:end); % remove . and ..

numClasses = length(classes);
fprintf('Number of classes: %d\n', numClasses);

%Count Images per Class
counts = zeros(numClasses,1);
for i = 1:numClasses
    files = dir(fullfile(datasetPath, classes(i).name, '*.tif'));
    counts(i) = length(files);
end

%Plot Class Distribution
figure;
bar(counts);
title('Class Distribution');
xlabel('Class Index');
ylabel('Number of Images');
grid on;

%Display Sample Images (first image of each class)
figure;
numDisplay = min(numClasses, 9); % show max 9 classes

for i = 1:numDisplay
    files = dir(fullfile(datasetPath, classes(i).name, '*.tif'));
    
    if ~isempty(files)
        img = imread(fullfile(datasetPath, classes(i).name, files(1).name));
        
        subplot(3,3,i);
        imshow(img,[]);
        title(classes(i).name, 'Interpreter', 'none');
    end
end

sgtitle('Sample Images from Dataset');

%Check Image Size Consistency
sizes = zeros(numClasses,2);

for i = 1:numClasses
    files = dir(fullfile(datasetPath, classes(i).name, '*.tif'));
    
    if ~isempty(files)
        img = imread(fullfile(datasetPath, classes(i).name, files(1).name));
        sizes(i,:) = size(img(:,:,1)); % grayscale assumed
    end
end

disp('Image sizes per class (height, width):');
disp(sizes);

if all(all(sizes == sizes(1,:)))
    fprintf('All images have consistent size: %dx%d\n', sizes(1,1), sizes(1,2));
else
    warning('Inconsistent image sizes detected!');
end

%Compute Mean and Standard Deviation (global)
pixelSum = 0;
pixelSqSum = 0;
totalPixels = 0;

for i = 1:numClasses
    files = dir(fullfile(datasetPath, classes(i).name, '*.tif'));
    
    for j = 1:length(files)
        img = imread(fullfile(datasetPath, classes(i).name, files(j).name));
        img = double(img);
        
        pixelSum = pixelSum + sum(img(:));
        pixelSqSum = pixelSqSum + sum(img(:).^2);
        totalPixels = totalPixels + numel(img);
    end
end

meanVal = pixelSum / totalPixels;
stdVal = sqrt((pixelSqSum / totalPixels) - meanVal^2);

fprintf('Global Mean: %.4f\n', meanVal);
fprintf('Global Std Dev: %.4f\n', stdVal);

%Pixel Intensity Histogram (sample)
figure;

% Take one sample image
files = dir(fullfile(datasetPath, classes(1).name, '*.tif'));
img = imread(fullfile(datasetPath, classes(1).name, files(1).name));

histogram(img(:), 50);
title('Pixel Intensity Distribution');
xlabel('Pixel Value');
ylabel('Frequency');
grid on;

%Done
disp('EDA completed successfully.');
