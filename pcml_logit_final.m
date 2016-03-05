%%
% Group - AGRA
% PCML_Logit function  learns Logistic Regression model with given Training Data
% Inputs:
% multi      = input 1 if multiclass classification is to be done else binary
%              classification is done
% dopca      = input 1 if PCA is to be done training data else all features are
%              taken
% hogfeature = input 1 if HOG features are to be used for training the Logistic Model else CNN
%              features are taken
function pcml_logit_final(multi,dopca,hogfeature)

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
    
    [coeff mu_mean latent] =  pca(X');
    
    %taking 95% variance, but as mentioned in the report idx = 100 was
    %taken for best result in CNN feature and applied only for multiclass classification
    idx = max(find(cumsum(latent/sum(latent))<0.95));
    %    X = pcaApply(X',coeff,mu_mean,idx)';           %Uncomment to take 95% variance
    X = pcaApply(X',coeff,mu_mean,100)';                %Comment if above one is used 

end


%set the value for the fold
Kfold = 10;


doclassification(Kfold,X,train,multi);

%Remove Path to Piotr's toolbox to avoid conflict with MATLAB's inbuilt
%function
if dopca
    addpath(genpath('/Users/vidit/Desktop/Semester 1/PCML'));
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
    fprintf('ONE Vs ALL\n');
    
    alpha = 0.1;
    lambda = 0;         %as mentioned in the report reduced dimension are taken
    %hence not regularized
    
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
        
        trainX = [ ones(size(trainX,1),1) trainX];
        testX = [ ones(size(testX,1),1) testX];
        
        for k = 1:4
            class1 = trainX(ytr == k,:);
            class2 = trainX(ytr ~= k,:);
            tX = [class1;class2];
            y = [ones(size(class1,1),1);zeros(size(class2,1),1)];
            beta(:,k) = logisticRegression(tX,y,alpha,lambda);
        end
        
        group_test = sigmoid(testX*beta);
        
        [ypred I] = max(group_test,[],2); %assign label to max sigmoid output value
        
        fprintf('Confusion Matrix for Test data\n');
        confmat_test = confusionmat(categorical(yte),categorical(I));
        disp(confmat_test);
        ber_test(fold) = mean((1-diag(confmat_test)./sum(confmat_test,2))*100);
        fprintf('BER Test: %f\n',ber_test(fold));
        
    end
    
    fprintf('Multiclass Classification: Final BER\n');
    disp(ber_test);
else
    
    fprintf('Binary Classification \n');
    alpha = 0.1;
    %lambda = [0.001 0.01 0.1 1 10 100] ;      % for doing learning best lambda
    lambda = 1000;                             % best BER obtained for lambda
    
    for hyper = 1:length(lambda)
        fprintf('lambda %f', lambda(hyper));
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
            
            trainX = [ones(size(trainX,1),1) trainX];
            testX  = [ones(size(testX,1),1)  testX];
            
            beta = logisticRegression(trainX,(ytr==4),alpha,lambda(hyper));
            
            group_test = (sigmoid(testX*beta)>0.5);
            
            fprintf('Confusion Matrix for Test data\n');
            confmat_test = confusionmat(categorical((yte==4)),categorical(group_test));
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
% Logistic Regression function
% Input:
% tX     - Training Data
% y      - Label for the training data
% alpha  - Learning rate of the model
% lambda - Regularizer for the model
% Output:
% beta   - learnt model parameters
function beta = logisticRegression(tX,y,alpha,lambda)
beta = zeros(size(tX,2),1);
maxIter = 1000;
conv=0;
for it = 1: maxIter
    g = logisticRegloss( beta,y,tX,lambda);
    %Convergence Criteria
    if( max(abs(g-0))<0.0001)
        conv=conv+1;
    else
        conv=0;
    end
    %Extra check to ensure convergence
    if(conv>3)
        break;
    end
    % beta update
    beta = beta-alpha*g;
end


%%
% logisticRegloss - returns Gradient of negative Log Likelihood Cost
% Input:
% beta   - Learnt model parameters
% y      - Output Label of training data tX
% tX     - Training data
% lambda - Regularizer for the model
% Output:
% g      - Gradient of negative Log Likelihood Cost
function g = logisticRegloss( beta,y,tX,lambda)
[N,b]=size(y);
Lbeta = lambda*beta/N;
Lbeta(1,1) = 0;

g=tX'*[sigmoid(tX*beta)-y];
g=g/N + Lbeta;

%%
%Sigmoid - Calculates the sigmoid output
% Input:
% input - for which sigmoid value is to be calculated
% Output:
% val - sigmoid value
function val = sigmoid(input)
t = exp(-input);
val = 1./(1+t);
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

