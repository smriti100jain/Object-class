# Random Forest for binary Classification using optimised parameters found by parameter_optimization_binary
#Libraries used  numpy1.9.2-2, scikit_learn 0.17-1, scipy 0.16.1-1


import numpy as np
import matplotlib.pyplot as plt
from collections import OrderedDict
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.cross_validation import StratifiedKFold
from sklearn import tree
from sklearn.metrics import confusion_matrix
import scipy.io

#To import Reduced 6000x500 data matrix(after applying PCA and selecting 500 PC's)
X = scipy.io.loadmat('C:\Users\Smriti\Desktop\pcmproject\cnn_500.mat')
X=X['X']

#To import data labels converted as binary label(0 for others and 1 for rest)
y = scipy.io.loadmat('C:\Users\Smriti\Desktop\pcmproject\kbinary_y.mat')
y=y['binary_y']
y=y.ravel()

size=X.shape
N=size[0]   #number of samples
D=size[1]   #number of feature

print(__doc__)

RANDOM_STATE = 123  # random state set

#No. of trees to explore varied from 2 to 100
min_estimators = 2
max_estimators = 100

k=10 #no. of folds: 10 fold CV

C=2 #number of classes
size=y.size
# We used StratifiedKFold function of sklearn library of python. 
#It divides all the samples in k-groups of test and
# train sets such that each set contains approximately 
# the same percentage of samples of each class. 
# This is useful as our dataset have unbalanced classes. 
# And this approach makes the tree balanced and prevent the tree from being biased towards dominant classes.
k_fold = StratifiedKFold(y, k)

#ber is an array with rows indicating folds and column indicating number of trees
ber=np.zeros((k,(max_estimators-min_estimators+1)))

count=0
for train_indices, test_indices in k_fold:

	#Warm_start is set True--> this implies too reuse the solution of previous call(in this case previous number of trees)--just to make it faster
	#max_features=None -> all the 500 features are randomly tried out at each split/decision point in order to find the best split
	clf = RandomForestClassifier(warm_start=True, max_features=None,max_depth=9,min_samples_split=4,min_samples_leaf=5,
                                oob_score=False,bootstrap=True,random_state=RANDOM_STATE)
	

            
    for i in range(min_estimators, max_estimators):
		#loop for varying num_trees    
                
            print "%d",i
            clf.set_params(n_estimators=i)  #set num_trees as i
            clf.fit(X[train_indices], y[train_indices]) #fit the classifier on train data
            out=clf.predict(X[test_indices]) #predicts on test indices
            y_split=y[test_indices] #contains actual test labels
    
            
            Nc=np.array([np.sum(y_split==0),np.sum(y_split==1)]) #conatins numbers of 1's and 0's
            confusion_matrix(y_split, out) #confusion matrix between actual and predicted labels
            error=0.0  # to find BER
            for j in range(0,len(test_indices)):
                arr=out[j]
                if(arr!=y_split[j]):
                    error=error+(1/float(Nc[(y_split[j])]))
            error=error/float(C)
            ber[count][(i-2)]=error*100  #stores BER corresponding to num_trees and corresponding fold
                       
            print "%d %d %f",count,i,error
            
               
    count=count+1 #increment the counter for num_trees        
net_error=np.mean(ber,axis=0) #average BER accross all folds for all num_trees
net_std=np.std(ber,axis=0) #standard deviation accross all folds for all num_trees

