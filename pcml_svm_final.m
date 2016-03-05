%%
% Group - AGRA
% PCML_SVM function returns learns Linear SVM model with given Training Data
% Inputs:
% multi      = input 1 if multiclass classification is to be done else binary
%              classification is done
% dopca      = input 1 if PCA is to be done training data else all features are
%              taken
% hogfeature = input 1 if HOG features are to be used for training the SVM else CNN
%              features are taken
function pcml_svm_final(multi,dopca,hogfeature)

if ~exist('multi','var')
    multi=0;
end

if ~exist('dopca','var')
    dopca=0;
end

if ~exist('hogfeature','var')
    hogfeature=0;
end

load('train/train.mat');

%Normalize the Data
if hogfeature == 1
    disp('HOG Feature');
    X = zscore(train.X_hog);
else
    disp('CNN Feature');
    X = zscore(train.X_cnn);
end


if dopca
    %Add Path to include Piotr's toolbox(one mentioned in Project webpage)
    addpath(genpath('/Users/vidit/Desktop/Semester 1/PCML'));

     disp('Applying PCA');

    [coeff mu latent] =  pca(X');
    %taking 95% variance, but as mentioned in the report idx = 100 was
    %taken for best result in CNN feature
    idx = max(find(cumsum(latent/sum(latent))<0.95));
 %  X = pcaApply(X',coeff,mu,idx)';       %Uncomment to take 95% variance
    X = pcaApply(X',coeff,mu,100)';       %Comment if above one is used 

end

%set the value for the fold
Kfold = 10;

doclassification(Kfold,X,train,multi);

%Remove Path to Piotr's toolbox to avoid conflict with MATLAB's inbuilt
%function
if dopca
    rmpath(genpath('/Users/vidit/Desktop/Semester 1/PCML'));
end

%%
% Main function where training of the classifier is done
% Inputs:
% Kfold - Number of Folds to be taken for Cross Validation
% X     - Normalized (and dimension reduced) feature vector
% train - Training data with labels and feature vectors
% multi - if 1 for multiclass classification else binary classification
function doclassification(Kfold,X,train,multi)
%One Vs All strategy implemented
if multi 
    fprintf('Multiclass Classification \n');
    fprintf('ONE Vs ALL\n')
    

   % C = [0.001:0.01:0.1]; % Used for grid search
    C = 1;
    for hyper1 = 1:length(C)
        for hyper2 = 1:length(C)
           % hyperpara = [ 0.1 0.1 C(hyper1) C(hyper2)]; % Used for grid search
           hyperpara = [0.1 0.1 0.061 0.051];  % best BER obtained for this config
 
            idxCV = crossValidation(size(train.y,1),Kfold);
            
            for fold = 1:Kfold
                fprintf('\n***** Fold %d *****\n',fold);
                
                test_idx = idxCV(fold,:);
                train_idx = idxCV([1:fold-1 fold+1:end],:);
                train_idx = train_idx(:);
                testX = X(test_idx,:);
                ytr = train.y(train_idx);
                trainX = X(train_idx,:);
                yte = train.y(test_idx);
                
                for k = 1:4             
                    class1 = trainX(ytr == k,:);
                    class2 = trainX(ytr ~= k,:);
                    tX = [class1;class2];
                    y = [ones(size(class1,1),1)*k;ones(size(class2,1),1)*0];
                    classifier{k} = bin_svm(tX,y,hyperpara(k));             
                end
                
                for k = 1:4
                    group_test(:,k) = classify(classifier{k},testX,ytr);
                end

                [ypred I] = min(group_test,[],2);
            
                fprintf('Confusion Matrix for Test data\n');
                confmat_test = confusionmat(categorical(yte),categorical(I));
                disp(confmat_test);
                ber_test(hyper1,hyper2,fold) = mean((1-diag(confmat_test)./sum(confmat_test,2))*100);
                fprintf('BER Test: %f\n',ber_test(hyper1,hyper2,fold));
                
            end
        end
    end
    fprintf('Multiclass Classification: Final BER\n');
    disp(ber_test);
else
    
    fprintf('Binary Classification \n');
    
    %C =0.001:0.009:0.1;              % used for grid search
    C = 0.025;                        % best BER obtained for this C when 100 PCs taken

    for hyper = 1:length(C)
        idxCV = crossValidation(size(train.y,1),Kfold);
        for fold = 1:Kfold
            fprintf('\n***** Fold %d *****\n',fold);
            
            test_idx = idxCV(fold,:);
            train_idx = idxCV([1:fold-1 fold+1:end],:);
            train_idx = train_idx(:);
            testX = X(test_idx,:);
            ytr = train.y(train_idx);
            trainX = X(train_idx,:);
            yte = train.y(test_idx);
            
            svmstruct = bin_svm(trainX,(ytr==4)+1,C(hyper));
            
            group_test = svmclassify(svmstruct,testX);
            
            fprintf('Confusion Matrix for Test data\n');
            confmat_test = confusionmat(categorical((yte==4)+1),group_test);
            disp(confmat_test);
            ber_test(hyper,fold) = mean((1-diag(confmat_test)./sum(confmat_test,2))*100);
            fprintf('BER Test: %f\n',ber_test(hyper,fold));
        end
        fprintf('Binary Classification: Final BER\n');
        disp(ber_test(hyper,:));
        meanBer(hyper) = mean(ber_test(hyper,:));
    end
    disp(meanBer);
end

%%
% Binary Class SVM- function called to learn decision boundary
% Input:
% tX        - Training data to be learnt
% y         - Labels for the training data
% C         - Hyperparameter for the SVM
% Output:
% svmstruct - Structure containing information of the trained SVM classifier 
%
function [svmstruct]= bin_svm(tX,y,C)
if ~exist('C','var') || isempty(C)
    C=1;
end

option = statset('MaxIter',250000);
svmstruct = svmtrain(tX,categorical(y),'options',option,'tolkkt',1e-4,'boxconstraint',C);

group_train = svmclassify(svmstruct,tX);

class =  unique(y);
fprintf('Class:%d vs Class:%d\n',class(1),class(2));
confmat_train = confusionmat(categorical(y),group_train);
% disp('Confusion Matrix');
% disp(confmat_train);
ber_train = mean((1-diag(confmat_train)./sum(confmat_train,2))*100);
fprintf('BER Train: %f \n',ber_train);

%%
% Classify - returns labels for the test data depending on the classifier
% Input: 
% svmstruct - Structure containing information of the trained SVM classifier 
% test      - Test data whose labels are to be predicted
% ytr       - Labels of the training data
% Ouput:
% pred      - Predicted Distance of test data from the Decision boundary 
%
function pred = classify(svmstruct,test,ytr)
sv = svmstruct.SupportVectors;
idx = svmstruct.SupportVectorIndices;
alpha = svmstruct.Alpha;
bias = svmstruct.Bias;
meanval = svmstruct.ScaleData.shift;
stdval = svmstruct.ScaleData.scaleFactor;

test_norm = test - ones(size(test,1),1)*meanval;
test_norm = test_norm*diag(stdval);

ytr(ytr == 0) = -1;
ytr(ytr ~= 0) = 1;
ysv = ytr(idx);



kernel_pred = (test_norm*sv');
pred = kernel_pred * (alpha .* double(ysv)) + bias;

%%
% CrossValidation - returns permuted indices for Cross Validation
% Input:
% N       - Size of the Array 
% K       - No. of Folds
% Output:
% idxCV   - Permuted indices
function idxCV = crossValidation(N,K)
setSeed(1);
idx = randperm(N);
Nk = floor(N/K);
for ki = 1:K
    idxCV(ki,:) = idx(1+(ki-1)*Nk:ki*Nk);
end


%%
% SetSeed - for setting the seed of random function generator
% Input:
% seed - value of the seed
function setSeed(seed)
global RNDN_STATE  RND_STATE
RNDN_STATE = randn('state');
randn('state',seed);
RND_STATE = rand('state');
%rand('state',seed);
rand('twister',seed);

