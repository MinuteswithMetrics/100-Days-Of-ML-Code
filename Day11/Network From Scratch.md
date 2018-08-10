# Network From Scratch

## Iris Species: Classify iris plants into three species in this classic dataset

The Iris dataset was used in R.A. Fisher's classic 1936 paper, The Use of [Multiple Measurements in Taxonomic Problems](http://rcs.chemometrics.ru/Tutorials/classification/Fisher.pdf), and can also be found on the [UCI Machine Learning Repository](http://archive.ics.uci.edu/ml/).

It includes three iris species with 50 samples each as well as some properties about each flower. One flower species is linearly separable from the other two, but the other two are not linearly separable from each other.

The columns in this dataset are:

* Id
* SepalLengthCm
* SepalWidthCm
* PetalLengthCm
* PetalWidthCm
* Species

![Iris](https://kasperfred.com/media/uploads/Figure_2_P6AJPcH.png)


## Data

The two libraries we will mainly use are `numpy` for the mathematical operations and `pandas` for reading the dataset (we will also use another library later on). Let's import them!

```python
import numpy as np
import pandas as pd
```
Pandas is used to shuffle the dataset into train/test/validation to avoid overfitting

```python
iris = pd.read_csv("../input/Iris.csv")
iris = iris.sample(frac=1).reset_index(drop=True) # Shuffle
```

```python
iris = pd.read_csv("../input/Iris.csv")
iris = iris.sample(frac=1).reset_index(drop=True) # Shuffle
```

```python
X = iris[['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']]
X = np.array(X)
X[:5]
```

```python
from sklearn.preprocessing import OneHotEncoder
one_hot_encoder = OneHotEncoder(sparse=False)

Y = iris.Species
Y = one_hot_encoder.fit_transform(np.array(Y).reshape(-1, 1))
Y[:5]
```
## Split Dataset 
```python
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.15)
X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size=0
```

## Implementation

In this tutorial, we are going to build a simple neural network that supports multiple layers and validation. The main function is `NeuralNetwork`, which will train the network for the specified number of epochs. At first, the weights of the network will get randomly initialized by `InitializeWeights`. Then, in each epoch, the weights will be updated by `Train` and finally, every 20 epochs accuracy both for the training and validation sets will be printed by the `Accuracy` function. As input the function receives the following:

* `X_train`, `Y_train`: The training data and target values.
* `X_val`, `Y_val`: The validation data and target values. These are optional parameters.
* `epochs`: Number of epochs. Defaults at 10.
* `nodes`: A list of integers. Each integer denotes the number of nodes in each layer. The length of this list denotes the number of layers. That is, each integer in this list corresponds to the number of nodes in each layer.
* `lr`: The learning rate of the back-propagation training algorithm. Defaults at 0.15.
credit: [Anthony Marakis](https://www.kaggle.com/antmarakis/another-neural-network-from-scratch "Another Neural Network From Scratch")

```python
def NeuralNetwork(X_train, Y_train, X_val=None, Y_val=None, epochs=10, nodes=[], lr=0.15):
    hidden_layers = len(nodes) - 1
    weights = InitializeWeights(nodes)

    for epoch in range(1, epochs+1):
        weights = Train(X_train, Y_train, lr, weights)

        if(epoch % 20 == 0):
            print("Epoch {}".format(epoch))
            print("Training Accuracy:{}".format(Accuracy(X_train, Y_train, weights)))
            if X_val.any():
                print("Validation Accuracy:{}".format(Accuracy(X_val, Y_val, weights)))
            
    return weights
```
The weights of the network are initialized randomly in the range [-1, 1] by `InitializeWeights`. This function takes as input `nodes` and returns a multi-dimensional array, `weights`. Each element in the `weights` list represents a hidden layer and holds the weights of connections from the previous layer (including the bias) to the current layer. So, element `i` in `weights` holds the weights of the connections from layer `i-1` to layer `i`. Note that the input layer has no incoming connections so it is not present in `weights`.

For example, let's say we have four features (as is the case with the Iris dataset) and the hidden layers have 5, 10 and 3 (for the output, one for each class) nodes. Thus, `nodes == [4, 5, 10, 3]`  Then, the connections between the input layer and the first hidden layer will be (4+1)\*5 = 25. After augmenting the input with the bias (in this case the bias has a constant value of 1), the input layer has 5 nodes. By fully connecting this layer to the next (each node in the input layer is connected will every node of the hidden layer), we get that the total number of connections is 25. Similarly, we get that the connections between the first hidden layer and the second one will be (5+1)\*10 = 60 and between the second hidden layer with the output we have (10+1)\*3 = 33 connections.

In the implementation, `numpy` is used to generate a random number in the `[-1, 1]` range for each connection.

```python
def InitializeWeights(nodes):
    """Initialize weights with random values in [-1, 1] (including bias)"""
    layers, weights = len(nodes), []
    
    for i in range(1, layers):
        w = [[np.random.uniform(-1, 1) for k in range(nodes[i-1] + 1)] for j in range(nodes[i])]
        weights.append(np.matrix(w))
    
    return weights
```
With the weights of the network at hand, we want to continuously adjust them across the epochs so that (hopefully) our network becomes more accurate. The training of the weights is accomplished via the popular (Forward) Back-Propagation algorithm. In this technique, the input first passes through the whole network and the output is calculated. Then, according to the error of this output, the weights of the network are updated from last to first. The error is propagated *backwards*, hence the name of the titular algorithm. Let's get into more detail about these two steps:

**Forward Propagation:**

* Each layer receives an input and computes an output. The output is computed by first calculating the dot product between the input and the weights of the layer and then passing this dot product through an activation function (in this case, the sigmoid function).
* The output of each layer is the input of the next.
* The input of the first layer is the feature vector.
* The output of the final layer is the prediction of the network.

```python
def ForwardPropagation(x, weights, layers):
    activations, layer_input = [x], x
    for j in range(layers):
        activation = Sigmoid(np.dot(layer_input, weights[j].T))
        activations.append(activation)
        layer_input = np.append(1, activation) # Augment with bias
    
    return activations
```

**Backward Propagation:**

* Calculate error at final output.
* Propagate error backwards through the layers and perform corrections.
    * Calculate Delta: Error of next layer *times* Sigmoid derivation of current layer activation
    * Update Weights between current layer and previous layer: Multiply delta with activation of previous layer and learning rate, and add this product to weights of previous layer
    * Calculate error for current layer. Remove the bias from the weights of the previous layer and multiply the result with delta to get error.
   
```python
def BackPropagation(y, activations, weights, layers):
    outputFinal = activations[-1]
    error = np.matrix(y - outputFinal) # Error at output
    
    for j in range(layers, 0, -1):
        currActivation = activations[j]
        
        if(j > 1):
            # Augment previous activation
            prevActivation = np.append(1, activations[j-1])
        else:
            # First hidden layer, prevActivation is input (without bias)
            prevActivation = activations[0]
        
        delta = np.multiply(error, SigmoidDerivative(currActivation))
        weights[j-1] += lr * np.multiply(delta.T, prevActivation)

        w = np.delete(weights[j-1], [0], axis=1) # Remove bias from weights
        error = np.dot(delta, w) # Calculate error for current layer
    
    return weights
```

```python
def Train(X, Y, lr, weights):
    layers = len(weights)
    for i in range(len(X)):
        x, y = X[i], Y[i]
        x = np.matrix(np.append(1, x)) # Augment feature vector
        
        activations = ForwardPropagation(x, weights, layers)
        weights = BackPropagation(y, activations, weights, layers)

    return weights
```

```python   
def Sigmoid(x):
    return 1 / (1 + np.exp(-x))

def SigmoidDerivative(x):
    return np.multiply(x, 1-x)  
```

```python
def Predict(item, weights):
    layers = len(weights)
    item = np.append(1, item) # Augment feature vector
    
    ##_Forward Propagation_##
    activations = ForwardPropagation(item, weights, layers)
    
    outputFinal = activations[-1].A1
    index = FindMaxActivation(outputFinal)

    # Initialize prediction vector to zeros
    y = [0 for i in range(len(outputFinal))]
    y[index] = 1  # Set guessed class to 1

    return y # Return prediction vector


def FindMaxActivation(output):
    """Find max activation in output"""
    m, index = output[0], 0
    for i in range(1, len(output)):
        if(output[i] > m):
            m, index = output[i], i
    
    return index
```
