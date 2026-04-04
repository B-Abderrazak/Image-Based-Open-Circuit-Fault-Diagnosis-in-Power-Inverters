clc; clear; close all;
%% ===== MODEL SELECTION =====

kfold = 5;
modelType = 2; % 1 = ResNet50, 2 = ViT

%% ===== MODEL SELECTION =====
switch modelType
    case 1
        fprintf('Using ResNet50\n');
         load features_res_close.mat

        % load("C:\Users\HP\Desktop\Nouveau dossier\features_res_close.mat")
        X = features_res_50;
        Y = categorical(labels);
    case 2
        fprintf('Using ViT\n');
        load features_ViT_close.mat

        % load("C:\Users\HP\Desktop\Nouveau dossier\features_ViT_close.mat")
        [numFeatures, numSamples] = size(features_ViT);
         X = zeros(1, 1, numFeatures, numSamples);
        for j = 1:numSamples
            X(1,1,:,j) = features_ViT(:,j);
        end
                Y = categorical(labels);
    otherwise
        error('Invalid modelType. Use 1 for ResNet50 or 2 for ViT.');
end
cv = cvpartition(Y, 'KFold', kfold);
AllResults = []; % store all folds
for ll = 1:cv.NumTestSets
    fprintf('\n===== Fold %d =====\n', ll);
    % Split data
    trainIdx = training(cv, ll);
    testIdx  = test(cv, ll);
    X_train = X(1,1,:,trainIdx);
    X_test  = X(1,1,:,testIdx);

    Y_train = Y(trainIdx);
    Y_test  = Y(testIdx);

    % Train model
    [Acc, YPred, probs] = Fit_MLP(X_train, Y_train, X_test, Y_test);

    % Confusion matrix
    confMat = confusionmat(Y_test, YPred);
    figure;
    confusionchart(Y_test, YPred);
    title(['Confusion Matrix - Fold ', num2str(ll)]);
    % Global accuracy
    Accuracy = 100*sum(diag(confMat)) / sum(confMat(:));

    nClasses = size(confMat,1);
    classes = categories(Y_test);

    precision = zeros(nClasses,1);
    recall    = zeros(nClasses,1);
    f1score   = zeros(nClasses,1);
    classACC  = zeros(nClasses,1);

    for k = 1:nClasses
        TP = confMat(k,k);
        FP = sum(confMat(:,k)) - TP;
        FN = sum(confMat(k,:)) - TP;
        TN = sum(confMat(:)) - (TP+FP+FN);

        precision(k) = 100*TP / (TP + FP + eps);
        recall(k)    =100* TP / (TP + FN + eps);
        f1score(k)   = 2 * precision(k) * recall(k) / (precision(k) + recall(k) + eps);
        classACC(k)  =100* (TP + TN) / sum(confMat(:));
    end

    % Create results table
    T = table(classes, classACC, precision, recall, f1score, ...
        'VariableNames', {'Class','Accuracy','Precision','Recall','F1Score'});

    disp(T)

    % Macro averages
    macroPrecision = mean(precision);
    macroRecall    = mean(recall);
    macroF1        =mean(f1score);
    macroACC       = mean(classACC);

    fprintf('Fold %d Results:\n', ll);
    fprintf('Accuracy     = %.4f\n', Accuracy);
    fprintf('Macro Prec   = %.4f\n', macroPrecision);
    fprintf('Macro Recall = %.4f\n', macroRecall);
    fprintf('Macro F1     = %.4f\n', macroF1);

    % Store results
    AllResults = [AllResults; [Accuracy, macroPrecision, macroRecall, macroF1]];

    %% ROC Curve
    figure; hold on;
    legends = {};

    for k = 1:numel(classes)
        yTrue = (Y_test == classes(k));
        yScore = probs(:,k);

        [Xc,Yc,~,AUC] = perfcurve(yTrue, yScore, 1);
        plot(Xc,Yc, 'LineWidth', 2);

        legends{k} = sprintf('%s (AUC=%.3f)', string(classes(k)), AUC);
    end

    xlabel('False Positive Rate');
    ylabel('True Positive Rate');
    title(['ROC Curve - Fold ', num2str(ll)]);
    legend(legends, 'Location', 'Best');
    grid on;

end

%% ===== FINAL CROSS-VALIDATION RESULTS =====
fprintf('\n===== FINAL RESULTS (Cross-Validation) =====\n');

meanResults = mean(AllResults,1);
stdResults  = std(AllResults,[],1);

FinalTable = table(meanResults', stdResults', ...
    'VariableNames', {'Mean','Std'}, ...
    'RowNames', {'Accuracy','Precision','Recall','F1Score'});

disp(FinalTable)