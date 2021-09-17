#import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.optimize as opt


# function used in the model
#-----------------------------------------------------------------------------

#the segmoied function
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# cost functhion J(θ) = (−1/m)[∑y(i)log(hθ(x(i)))+(1−y(i))log(1−hθ(x(i)))] 
def cost(theta, X, y):
    theta = np.matrix(theta)
    X = np.matrix(X)
    y = np.matrix(y)
    first = np.multiply(-y, np.log(sigmoid(X * theta.T))) #y(i)log(hθ(x(i))
    second = np.multiply((1 - y), np.log(1 - sigmoid(X * theta.T))) # (1−y(i))log(1−hθ(x(i))
    return np.sum(first - second) / (len(X))

#  this is a gradiant decent function, function to calculate the new theta
def gradient(theta, X, y):
    theta = np.matrix(theta)
    X = np.matrix(X)
    y = np.matrix(y)
    
    parameters = int(theta.ravel().shape[1])
    grad = np.zeros(parameters)
    
    error = sigmoid(X * theta.T) - y
    
    for i in range(parameters):
        term = np.multiply(error, X[:,i])
        grad[i] = np.sum(term) / len(X)
    
    return grad

#input data to the model and let it to predict if it is accepted or refuced (Admitted or not)
def predict(theta, X):
    probability = sigmoid(X * theta.T)
    return [1 if x >= 0.5 else 0 for x in probability]


#-----------------------------------------------------------------------------
#read data from the source
path = 'collage accepted.txt'
data = pd.read_csv(path, header=None, names=['Exam 1', 'Exam 2', 'Admitted'])

#show data 
print('data = ')
print(data.head(10))

#saperate the accepted person from refused to plot 
accepted = data[data['Admitted'].isin([1])]
refused = data[data['Admitted'].isin([0])]

# plot the data
fig, ax = plt.subplots(figsize=(5,5))
ax.scatter(accepted['Exam 1'], accepted['Exam 2'], s=50, c='b', marker='o', label='accepted')
ax.scatter(refused['Exam 1'], refused['Exam 2'], s=50, c='r', marker='x', label='refused')
ax.legend()
ax.set_xlabel('Exam_1')
ax.set_ylabel('Exam_2')


# add a ones column
data.insert(0, 'Ones', 1)


# separate X (training data) from y (target variable)
cols = data.shape[1]
X = data.iloc[:,0:cols-1]
y = data.iloc[:,cols-1:cols]

print('**************************************')
print('X2 data = \n' ,X.head(10) ) #print the 'X' values
print('y2 data = \n' ,y.head(10) ) #print the 'Y' values
print('**************************************')


# convert from data frames to numpy matricesX = np.array(X.values)
X = np.array(X.values)
y = np.array(y.values)
theta = np.zeros(3)


thiscost = cost(theta, X, y)
print()
print('cost = ' , thiscost)


#optimization function
result = opt.fmin_tnc(func=cost, x0=theta, fprime=gradient, args=(X, y))

print(result)

costafteroptimize = cost(result[0], X, y)
print()
print('cost after optimize = ' , costafteroptimize)
print()

#the real theta
theta_min = np.matrix(result[0])

# predict the output
predictions = predict(theta_min, X)
print(theta_min.shape)
print(X.shape)

#calculate the accuracy of the model
correct = [1 if ((a == 1 and b == 1) or (a == 0 and b == 0)) else 0 for (a, b) in zip(predictions, y)]
accuracy = (sum(map(int, correct)) % len(correct))
print ('accuracy = {0}%'.format(accuracy))
