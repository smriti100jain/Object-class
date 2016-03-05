#This Code find the optimised parameter for BINARY CLASSIFICATION  by 10 fold cross-validation 
#search by randomly sampling
#parameters from parameter space defined such that they minimize the Balanced Error Rate 

print(__doc__)

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import make_scorer
from time import time
from operator import itemgetter
from sklearn.grid_search import RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from collections import OrderedDict
from sklearn import tree
from sklearn import cross_validation
from sklearn.metrics import confusion_matrix
import scipy.io


#To import Reduced 6000x500 data matrix(after applying PCA and selecting 500 PC's)
X = scipy.io.loadmat('C:\Users\Smriti\Desktop\pcmproject\cnn_500.mat')
X=X['X']

#To import data labels converted as binary label(0 for others and 1 for rest)
y = scipy.io.loadmat('C:\Users\Smriti\Desktop\pcmproject\kbinary_y.mat')
y=y['binary_y']
y=y.ravel()

def my_custom_loss_func(ground_truth, predictions):
			#My Custom Loss Function to Calculate BER Loss Function so that Randomised search 
			#finds parameters which minimizes BER

            C=2
            Nc=np.array([np.sum(ground_truth==0),np.sum(ground_truth==1)])
            
            
            error=0.0
            for j in range(0,len(predictions)):
                arr=predictions[j]
                if(arr!=ground_truth[j]):
                    error=error+(1/float(Nc[(ground_truth[j])]))
            error=error/float(C)
            
            return error
         


size=X.shape
N=size[0]   #number of samples
D=size[1]   #number of feature


# build a Random Forest classifier
clf = RandomForestClassifier(n_estimators=20,bootstrap=True,warm_start=False)


# Utility function to report best scores
def report(grid_scores, n_top=3):
    top_scores = sorted(grid_scores, key=itemgetter(1), reverse=True)[:n_top]
    for i, score in enumerate(top_scores):
        print("Model with rank: {0}".format(i + 1))
        print("Mean validation score: {0:.3f} (std: {1:.3f})".format(
              score.mean_validation_score,
              np.std(score.cv_validation_scores)))
        print("Parameters: {0}".format(score.parameters))
        print("")

#My Custom Loss Function to Calculate BER Loss Function so that Randomised search 
#finds parameters which minimizes BER
#This fnction makes my BER as scorer
loss  = make_scorer(my_custom_loss_func, greater_is_better=False)

# Search Space for parameters where randomised search is done
param_grid = {"max_depth": [3,4,5,6,7,8,9,10],
              "min_samples_split": [4,5,6,7,8,9,10],
              "min_samples_leaf": [4,5,6,7,8,9,10],
              "criterion": ["gini","entropy"],"max_features" : ["log2","sqrt","auto"]
              }

n_iter_search = 50  #This variable store number of times parameter are randomly sampled.
# n_iter trades off runtime vs quality of the solution

#Rndomized Search over parameters with BER as the score function with 10 fold CV
random_search = RandomizedSearchCV(clf, param_distributions=param_grid,scoring=loss,
                                   n_iter=n_iter_search)

start = time()

#Run fit on the estimator with randomly drawn parameters.
random_search.fit(X, y)

print("RandomizedSearchCV took %.2f seconds for %d candidates"
      " parameter settings." % ((time() - start), n_iter_search))
#report function reports the best three parameter setting which least BER after 
#all iterations of randomised search is complete
report(random_search.grid_scores_)

