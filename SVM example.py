#imports
import numpy as np
import seaborn as sns
sns.set_context('notebook')
sns.set_style('white')

from scipy.io import loadmat
from sklearn import svm

# load the train data and test data
spam_train = loadmat('spamTrain.mat')
spam_test = loadmat('spamTest.mat')

# show the data
print(spam_train)
print(spam_test)

# split the data to feature (X) and target (y)
X = spam_train['X']
Xtest = spam_test['Xtest']
y = spam_train['y'].ravel()
ytest = spam_test['ytest'].ravel()

# show the data after split
print(Xtest)
print(X.shape, y.shape, Xtest.shape, ytest.shape)

#train the data by using SCV algorithm 
clf2 = svc = svm.SVC()
clf2.fit(X, y)

# Testing and calculate the accuracy
print('Test accuracy = {0}%'.format(np.round(svc.score(Xtest, ytest) * 100, 2)))
