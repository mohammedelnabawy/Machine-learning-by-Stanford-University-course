import numpy as np 
hidden = 4

#the segmoied function
def sigmoid(z):
    return 1 / (1 + np.exp(-z)) 


def dir_sigmoid(z):
    return z * (1 - z)


class neural:
    # init the weights
    def __init__(self, x, y):
        self.input = x
        # print('the input records ',self.input)
        self.y = y
        # print('the y of neural \n', self.y)
        self.weight1 = np.random.rand(self.input.shape[1], hidden)
        # print('weight of layer 1 \n', self.weight1)
        self.weight2 = np.random.rand(hidden, 1)
        # print('weight of layer 2 \n', self.weight2)
        self.output = np.zeros(self.y.shape)
        # print('the output matrix \n', self.output)
        
    
    # the feed forward operation
    def feedforward(self):
        self.firstlayer = sigmoid(np.dot(self.input, self.weight1))
        # print('first layer \n ', self.firstlayer)
                
        self.output = sigmoid(np.dot(self.firstlayer, self.weight2))
        # print('output layer \n ', self.output)

    # the back propagation operation
    def backpro(self):
        # application of the chain rule to find derivative of the loss function with respect to weights2 and weights1
        d_weights2 = np.dot(self.firstlayer.T, (2*(self.y - self.output) * dir_sigmoid(self.output)))
        # print('d_weights2  \n',d_weights2  )
        # print()        
        d_weights1 = np.dot(self.input.T, (np.dot(2*(self.y - self.output) * dir_sigmoid(self.output),self.weight2.T) * dir_sigmoid(self.firstlayer)))
        # print('d_weights1 \n',d_weights1)
        # print()        

        # update the weights with the derivative (slope) of the loss function
        self.weight1 += d_weights1
        self.weight2 += d_weights2
        
        
x = np.array([[0,0,1],
              [0,1,1],
              [1,0,1],
              [1,1,1]])


y = np.array([[0],
              [1],
              [1],
              [0]])


z = neural(x, y)

# repeat number of times
for i in range(20000):
    z.feedforward()
    z.backpro()
    
    
print(z.output)