 ## Titanic: Machine Learning from Disaster
*Predict survival on the Titanic passengers

![GitHub](https://d1s0cxawdx09re.cloudfront.net/uploads/2015/04/09_titanic.jpg)


In this notebook, we are building a 3-layer neural network with numpy for the Kaggle Titanic Dataset, and comparing the performance difference between a standard Stochastic Gradient Descent and Adam.

## Introduction
The sinking of the RMS Titanic is one of the most infamous shipwrecks in history. On April 15, 1912, during her maiden voyage, the Titanic sank after colliding with an iceberg, killing 1502 out of 2224 passengers and crew. This sensational tragedy shocked the international community and led to better safety regulations for ships.

One of the reasons that the shipwreck led to such loss of life was that there were not enough lifeboats for the passengers and crew. Although there was some element of luck involved in surviving the sinking, some groups of people were more likely to survive than others, such as women, children, and the upper-class.


```python
# Imports
%matplotlib inline
%config InlineBackend.figure_format = 'retina'

import numpy as np # For Mathematical functions 
import pandas as pd # Widely used tool for data manipulation
import matplotlib.pyplot as plt # Visualization
```

### Data Preparation 


```python
train = pd.read_csv('train.csv')
# Converting Passenger Class and Embarked Location to binary variables
dummy_fields = ['Sex', 'Pclass', 'Embarked']
for i in dummy_fields:
    dummies = pd.get_dummies(train[i], prefix=i, drop_first=False)
    train = pd.concat([train, dummies], axis=1)

# Dropping unneeded columns 
fields_to_drop = ['PassengerId', 'Ticket', 'Name', 'Cabin', 'Fare', 'Pclass', 'Embarked', 'Sex']
data = train.drop(fields_to_drop, axis=1)
```

```python
# Normalising Age
mean, std = data['Age'].mean(), data['Age'].std()
data.loc[:, 'Age'] = (data['Age'] - mean)/std
data = data.fillna(0) # Replace NaN age with mean (0)
```

```python
# Shuffle Data 
data = data.sample(frac=1).reset_index(drop=True)

# Splitting Data into Train Val and Test Set
features, targets = data.drop('Survived', axis=1), data['Survived'] 
targets = targets.values.reshape(-1, 1) # reshaping to numpy n x 1 matrix

test_X, test_y = features[-120:], targets[-120:]
val_X, val_y = features[-200:-120], targets[-200:-120]
train_X, train_y = features[:-200], targets[:-200]
```

### Neural Network

```python
class NeuralNetwork(object):
    def __init__(self, input_nodes, hidden_nodes, output_nodes, learning_rate):
        self.input_nodes = input_nodes
        self.hidden_nodes = hidden_nodes
        self.output_nodes = output_nodes
        self.lr = learning_rate
        self.l2_m = 0
        self.l1_m = 0
        self.l2_v = 0
        self.l1_v = 0
        self.t = 0
        
        # Weights Initilization
        self.w0 = np.random.normal(0.0, 0.1, (self.input_nodes, self.hidden_nodes))
        self.w1 = np.random.normal(0.0, 0.1, (self.hidden_nodes, self.output_nodes))
        
        def sigmoid(x, deriv=False):
            
            if deriv:
                return x*(1-x)
            return 1/(1+np.exp(-x))
        
        self.activation_function = sigmoid
        
    def train(self, features, targets, optimizer, decay_rate_1 = None, 
              decay_rate_2 = None, epsilon = None):
        # Feed Forward
        l0 = features
        l1 = self.activation_function(np.dot(l0, self.w0))
        l2 = self.activation_function(np.dot(l1, self.w1))
        
        # Backpropagation
        l2_error = l2 - targets
        l2_delta = l2_error * self.activation_function(l2, deriv=True)
        l1_error = l2_delta.dot(self.w1.T)
        l1_delta = l1_error * self.activation_function(l1, deriv=True)
        
        if optimizer == 'sgd':
            # Update Weights
            self.w1 -= self.lr * l1.T.dot(l2_delta)
            self.w0 -= self.lr * l0.T.dot(l1_delta)
            
        if optimizer == 'adam':
            # Gradients for each layer
            g1 = l1.T.dot(l2_delta)
            g0 = l0.T.dot(l1_delta)
            
            self.t += 1 # Increment Time Step
            
            # Computing 1st and 2nd moment for each layer
            self.l2_m = self.l2_m * decay_rate_1 + (1- decay_rate_1) * g1
            self.l1_m = self.l1_m * decay_rate_1 + (1- decay_rate_1) * g0
            
            self.l2_v = self.l2_v * decay_rate_2 + (1- decay_rate_2) * (g1 ** 2)
            self.l1_v = self.l1_v * decay_rate_2 + (1- decay_rate_2) * (g0 ** 2)
            
            l2_m_corrected = self.l2_m / (1-(decay_rate_1 ** self.t))
            l2_v_corrected = self.l2_v / (1-(decay_rate_2 ** self.t))
            
            # Computing bias-corrected moment
            l1_m_corrected = self.l1_m / (1-(decay_rate_1 ** self.t))
            l1_v_corrected = self.l1_v / (1-(decay_rate_2 ** self.t))
            
            # Update Weights
            w1_update = l2_m_corrected / (np.sqrt(l2_v_corrected) + epsilon)
            w0_update = l1_m_corrected / (np.sqrt(l1_v_corrected) + epsilon)
            
            self.w1 -= (self.lr * w1_update)
            self.w0 -= (self.lr * w0_update)
            
    def run(self, features):
        l0 = features
        l1 = self.activation_function(np.dot(l0, self.w0))
        l2 = self.activation_function(np.dot(l1, self.w1))
        
        return l2

```

```python
def MSE(y, Y):
    return np.mean((y-Y)**2)
```    
