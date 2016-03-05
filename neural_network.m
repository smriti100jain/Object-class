clearvars;
%This code is for neural network for binary class

     
    
        
%To import Reduced 6000x100 data matrix(after applying PCA and selecting 100 PC's)
load 'Z:/PCML/X_100.mat';
X=X_100
%To import multi class labels
load 'Z:/PCML/y_binary.mat';
y=y_binary;


K = 5; %5 fold cross validation

error=zeros(K,1); %contains Test error for k folds
    
setSeed(42); %set a fixed seed
N = size(y,1);
% split data in K fold by creating indices)
idx = randperm(N);
Nk = floor(N/K);
for k = 1:K
    idxCV(k,:) = idx(1+(k-1)*Nk:k*Nk);
end
    
addpath(genpath('Z:\PCML\DeepLearnToolbox-master'));
	
for k = 1:K
            % get k'th subgroup in test, others in train
            
            Tr = [];
            Te = [];
            idxTe = idxCV(k,:);
            idxTr = idxCV([1:k-1 k+1:end],:);
            idxTr = idxTr(:);
            Te.y = y(idxTe);
            Te.X = X(idxTe,:);
            Tr.y = y(idxTr);
            Tr.X = X(idxTr,:);

            
                %%
            fprintf('Training simple neural network..\n');

            addpath(genpath('Z:\PCML\DeepLearnToolbox-master'));
    

            rng(8339,'twister');  % fix seed, this    NN may be very sensitive to initialization

             % setup NN. The first layer needs to have number of features neurons,
             % and the last layer the number of classes (here 2 as binary classification).
            
			%NNSETUP creates a Feedforward Backpropagate Neural Network

			nn = nnsetup([size(Tr.X,2) 50 2]); % just 1 hidden layer of 50 units  
            opts.numepochs =  30;   %  Number of full sweeps through data
            opts.batchsize = 500;  %  Take a mean gradient step over this many samples 500

            % if == 1 => plots trainin error as the NN is trained
            opts.plot               = 0;

            nn.learningRate = 2;
        
        
            nn.activation_function              = 'tanh_opt';   %  Activation functions of hidden layers: 'sigm' (sigmoid) or 'tanh_opt' (optimal tanh).
            nn.learningRate                     = 1.5;            %  learning rate Note: typically needs to be lower when using 'sigm' activation function and non-normalized inputs.
            nn.momentum                         = 0.5;          %  Momentum
            nn.scaling_learningRate             = 1;            %  Scaling factor for the learning rate (each epoch)
            nn.weightPenaltyL2                  = 0;            %  L2 regularization
            nn.dropoutFraction                  = 0.5;            %  Dropout level 
            nn.testing                          = 0;            %  Internal variable. nntest sets this to one.
            nn.output                           = 'sigm';       %  output unit 'sigm' (=logistic), 'softmax' a


			% this neural network implementation requires number of samples to be a
			% multiple of batchsize, so we remove some for this to be true.
            numSampToUse = opts.batchsize * floor( size(Tr.X) / opts.batchsize);
            Tr.X = Tr.X(1:numSampToUse,:);
            Tr.y = Tr.y(1:numSampToUse);

			% normalize data
            [Tr.normX, mu, sigma] = zscore(Tr.X); % train, get mu and std

            %prepare labels for binary classification
			LL = [1*(Tr.y == 0), 1*(Tr.y == 1) ];  % first column, p(y=1)
                                % second column, p(y=2), etc

			[nn, L] = nntrain(nn, Tr.normX, LL, opts);
			Te.normX = normalize(Te.X, mu, sigma);  % normalize test data

			% to get the scores we need to do nnff (feed-forward)
			nn.testing = 1;
			nn = nnff(nn, Te.normX, zeros(size(Te.normX,1), nn.size(end)));
			nn.testing = 0;


			% predict on the test set
			nnPred = nn.a{end};

			% get the most likely class
			[~,classVote] = max(nnPred,[],2);
			classVote=classVote~=1;
			
			confusionmat(Te.y,classVote)
			predErr = sum( classVote ~= Te.y ) / length(Te.y);
			%Class_weight=[sum(Te.y==1),sum(Te.y==2),sum(Te.y==3),sum(Te.y==4)];
			Class_weight=[sum(Te.y==0),sum(Te.y==1)];
			Ber=0.0;
			% get overall Balanced Error Rate
			for err=1:size(Te.y,1)
				Ber=Ber+((1/Class_weight((Te.y(err)+1)))*(classVote(err)~=Te.y(err)));
			end
			Ber=Ber/2;  % as two classes
			error(k,1)=Ber*100
	
  

        end
        
    