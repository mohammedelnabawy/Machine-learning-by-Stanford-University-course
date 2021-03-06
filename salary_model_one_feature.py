#import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


#read data from the source
path = 'D:\\modle\\salary_data.csv'
data = pd.read_csv(path)

#show data details
print('data = \n' ,data.head()) #print the first 5 records


#draw data to see points
data.plot(kind='scatter', x='YearsExperience', y='Salary', figsize=(5,5))


# adding a new column called ones to the data have value (1) (x0)
data.insert(0, 'Ones', 1)
print('new data = \n' ,data.head()) #print the first 5 records after adding the 'ones' columns
print('**************************************')


# separate X (training data) from y (target variable)
cols = data.shape[1]
X = data.iloc[:,0:cols-1]
y = data.iloc[:,cols-1:cols]


print('**************************************')
print('X data = \n' ,X.head() ) #print the 'X' values
print('y data = \n' ,y.head() ) #print the 'Y' values
print('**************************************')



# convert from data frames to numpy matrices
X = np.matrix(X.values) # matrix for X values
y = np.matrix(y.values) # matrix for y values
theta = np.matrix(np.array([0,0])) # theta matrix



# cost functhion = (1/2m)∑(h(x) - y)^2
def computeCost(X, y, theta):
    z = np.power(((X * theta.T) - y), 2) #
    return np.sum(z) / (2 * len(X))


# GD function
# θj = θj -(1/m)∑(h(x) - y)
def gradientDescent(X, y, theta, alpha, iters):
    temp = np.matrix(np.zeros(theta.shape)) # matrix for theta
    parameters = int(theta.ravel().shape[1]) # number of parameter in the equation
    cost = np.zeros(iters) # matrix to pull the cost values
    
    for i in range(iters): #repeat number of times to decrease the cost
        error = (X * theta.T) - y
        
        for j in range(parameters): # repeat to determine all theta 
            term = np.multiply(error, X[:,j])
            temp[0,j] = theta[0,j] - ((alpha / len(X)) * np.sum(term))
            
        theta = temp
        cost[i] = computeCost(X, y, theta)
        
    return theta, cost


# initialize variables for learning rate and iterations
alpha = 0.01 # learning rate 
iters = 1000 # number of iteration

# # perform gradient descent to "fit" the model parameters
g, cost = gradientDescent(X, y, theta, alpha, iters)


print('g = ' , g)
print('cost  = ' , cost[0:50] )
print('computeCost = ' , computeCost(X, y, g))
print('**************************************')

# get best fit line
x = np.linspace(data.YearsExperience.min(), data.YearsExperience.max(), 100)
print('x \n',x)
print('g \n',g)

f = g[0, 0] + (g[0, 1] * x) # the new function for the point
print('f \n',f)



# draw the line
fig, ax = plt.subplots(figsize=(5,5))
ax.plot(x, f, 'r', label='Prediction')
ax.scatter(data.YearsExperience, data.Salary, label='Traning Data')
ax.legend(loc=2)
ax.set_xlabel('Population')
ax.set_ylabel('Profit')
ax.set_title('Predicted Profit vs. Population Size')


# draw error graph
fig, ax = plt.subplots(figsize=(5,5))
ax.plot(np.arange(iters), cost, 'r')
ax.set_xlabel('Iterations')
ax.set_ylabel('Cost')
ax.set_title('Error vs. Training Epoch')
