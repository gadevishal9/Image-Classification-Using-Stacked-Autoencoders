'''
Python version - 3.6
'''
import numpy as np
from load_mnist import fashion_mnist
import matplotlib.pyplot as plt
import pdb
import random

import matplotlib.pyplot as plt
import pdb
import math
import sys, ast
epsilon = 1e-5

def relu(Z):
    '''
    computes relu activation of Z

    Inputs: 
        Z is a numpy.ndarray (n, m)

    Returns: 
        A is activation. numpy.ndarray (n, m)
        cache is a dictionary with {"Z", Z}
    '''
    A = np.maximum(0,Z)
    cache = {}
    cache["Z"] = Z
    return A, cache

def relu_der(dA, cache):
    '''
    computes derivative of relu activation

    Inputs: 
        dA is the derivative from subsequent layer. numpy.ndarray (n, m)
        cache is a dictionary with {"Z", Z}, where Z was the input 
        to the activation layer during forward propagation

    Returns: 
        dZ is the derivative. numpy.ndarray (n,m)
    '''
    dZ = np.array(dA, copy=True)
    Z = cache["Z"]
    dZ[Z<0] = 0
    return dZ

def linear(Z):
    '''
    computes linear activation of Z
    This function is implemented for completeness

    Inputs: 
        Z is a numpy.ndarray (n, m)

    Returns: 
        A is activation. numpy.ndarray (n, m)
        cache is a dictionary with {"Z", Z}
    '''
    A = Z
    cache = {}
    return A, cache

def linear_der(dA, cache):
    '''
    computes derivative of linear activation
    This function is implemented for completeness

    Inputs: 
        dA is the derivative from subsequent layer. numpy.ndarray (n, m)
        cache is a dictionary with {"Z", Z}, where Z was the input 
        to the activation layer during forward propagation

    Returns: 
        dZ is the derivative. numpy.ndarray (n,m)
    '''
   
    dZ = np.array(dA, copy=True)
    n,m = dZ.shape
    return dZ

def softmax_cross_entropy_loss(Z, Y=np.array([])):
    '''
    Computes the softmax activation of the inputs Z
    Estimates the cross entropy loss

    Inputs: 
        Z - numpy.ndarray (n, m)
        Y - numpy.ndarray (1, m) of labels
            when y=[] loss is set to []
    
    Returns:
        A - numpy.ndarray (n, m) of softmax activations
        cache -  a dictionary to store the activations later used to estimate derivatives
        loss - cost of prediction
    '''
    ### CODE HERE 
    #print(Z.shape)
    """iter1 = Z.shape[1]
    Z = Z.T
    e_z = Z
    Loss = 0
    cache = {}
    print("surya")
    for i in range(iter1):
        print(Z[i])
        e_z[i] = np.exp(Z[i] - np.max(Z[i]))
        e_z[i] = e_z[i]/np.sum(e_z[i])
        # print(e_z[i][int(Y[0][i])])
        if(e_z[i][int(Y[0][i])]>0):
            Loss = Loss + math.log(e_z[i][int(Y[0][i])])
        #print(np.sum(e_z[i]))

    A = e_z.T
    Loss = -Loss/iter1
    # print(Loss)
    cache['A'] = A
    #print(A.shape)
    return A, cache, Loss"""

    maximumZ = np.max(Z, axis = 0, keepdims = True)
    e = np.exp(Z - maximumZ)
    A = e / np.sum(e, axis = 0, keepdims = True)
    cache = {}
    cache["A"] = A
    
    loss = 0
    for i in range(Y.shape[1]):
        x = int(Y[0][i])
        loss += np.log(A[x, i])

    loss = -loss/Y.shape[1]
    return A,cache,loss
    """A = np.exp(Z - np.max(Z))
    sums = A.sum(axis=0)
    for index, x in np.ndenumerate(A):
        A[index[0], index[1]] = x/sums[index[1]]

    cache = {}
    cache['A'] = A

    if Y.shape[0]:
        cross_entropy_array = []
        for index, x in np.ndenumerate(Y):
            cross_entropy_array.append(-1. * np.log(A[int(x), index[1]] + epsilon))
        cross_entropy = np.asarray(cross_entropy_array).reshape(1, Y.shape[1])
        loss = np.mean(cross_entropy)
    else:
        loss = None

    return A, cache, loss"""

def softmax_cross_entropy_loss_der(Y, cache):
    '''
    Computes the derivative of softmax activation and cross entropy loss

    Inputs: 
        Y - numpy.ndarray (1, m) of labels
        cache -  a dictionary with cached activations A of size (n,m)

    Returns:
        dZ - numpy.ndarray (n, m) derivative for the previous layer
    '''
    ### CODE HERE 
    """A = cache['A']
    A = A.T
    for i in range(A.shape[0]):#A.shape[0]):
        #print(A[i])
        A[i][int(Y[0][i])] = A[i][int(Y[0][i])] - 1
        #print(A[i])
    dZ = A.T
    #print(dZ.shape)
    return dZ"""
    n,m = Y.shape
    dZ = cache['A'].copy()
    for index, x in np.ndenumerate(Y):
        dZ[int(x), index[1]] -= 1
    return dZ/m

def initialize_multilayer_weights(net_dims):
    '''
    Initializes the weights of the multilayer network

    Inputs: 
        net_dims - tuple of network dimensions

    Returns:
        dictionary of parameters
    '''
    np.random.seed(0)
    numLayers = len(net_dims)
    parameters = {}
    for l in range(numLayers-1):
        if(l<numLayers-2):
            parameters["W"+str(l+1)] = np.random.randn(net_dims[l+1], net_dims[l])* np.sqrt(2/net_dims[l+1])
            parameters["b"+str(l+1)] = np.random.randn(net_dims[l+1], 1)* np.sqrt(2/net_dims[l+1])
        else:
            parameters["W"+str(l+1)] = np.random.randn(net_dims[l+1], net_dims[l]) * 0.01
            parameters["b"+str(l+1)] = np.random.randn(net_dims[l+1], 1) * 0.01
    """for l in range(numLayers-1):

        parameters["W"+str(l+1)] = np.random.randn(net_dims[l+1], net_dims[l]) * 0.01
        parameters["b"+str(l+1)] = np.random.randn(net_dims[l+1], 1) * 0.01"""
    
    return parameters

def linear_forward(A, W, b):
    '''
    Input A propagates through the layer 
    Z = WA + b is the output of this layer. 

    Inputs: 
        A - numpy.ndarray (n,m) the input to the layer
        W - numpy.ndarray (n_out, n) the weights of the layer
        b - numpy.ndarray (n_out, 1) the bias of the layer

    Returns:
        Z = WA + b, where Z is the numpy.ndarray (n_out, m) dimensions
        cache - a dictionary containing the inputs A
    '''
    ### CODE HERE
    cache = {}
    cache["A"] = A
    Z = np.dot(W, A) + b
    return Z, cache

def layer_forward(A_prev, W, b, activation):
    '''
    Input A_prev propagates through the layer and the activation

    Inputs: 
        A_prev - numpy.ndarray (n,m) the input to the layer
        W - numpy.ndarray (n_out, n) the weights of the layer
        b - numpy.ndarray (n_out, 1) the bias of the layer
        activation - is the string that specifies the activation function

    Returns:
        A = g(Z), where Z = WA + b, where Z is the numpy.ndarray (n_out, m) dimensions
        g is the activation function
        cache - a dictionary containing the cache from the linear and the nonlinear propagation
        to be used for derivative
    '''
    Z, lin_cache = linear_forward(A_prev, W, b)
    if activation == "sigmoid":
        A, act_cache = sigmoid(Z)
    elif activation == "tanh":
        A, act_cache = tanh(Z)
    elif activation == "relu":
        A, act_cache = relu(Z)
    elif activation == "linear":
        A, act_cache = linear(Z)
    
    cache = {}
    cache["lin_cache"] = lin_cache
    cache["act_cache"] = act_cache
    return A, cache

def multi_layer_forward(X, parameters):
    '''
    Forward propgation through the layers of the network

    Inputs: 
        X - numpy.ndarray (n,m) with n features and m samples
        parameters - dictionary of network parameters {"W1":[..],"b1":[..],"W2":[..],"b2":[..]...}
    Returns:
        AL - numpy.ndarray (c,m)  - outputs of the last fully connected layer before softmax
            where c is number of categories and m is number of samples in the batch
        caches - a dictionary of associated caches of parameters and network inputs
    '''
    L = len(parameters)//2  
    A = X
    caches = []
    for l in range(1,L):  # since there is no W0 and b0
        A, cache = layer_forward(A, parameters["W"+str(l)], parameters["b"+str(l)], "relu")
        caches.append(cache)

    AL, cache = layer_forward(A, parameters["W"+str(L)], parameters["b"+str(L)], "linear")
    caches.append(cache)
    return AL, caches

def linear_backward(dZ, cache, W, b):
    '''
    Backward prpagation through the linear layer

    Inputs:
        dZ - numpy.ndarray (n,m) derivative dL/dz 
        cache - a dictionary containing the inputs A, for the linear layer
            where Z = WA + b,    
            Z is (n,m); W is (n,p); A is (p,m); b is (n,1)
        W - numpy.ndarray (n,p)
        b - numpy.ndarray (n, 1)

    Returns:
        dA_prev - numpy.ndarray (p,m) the derivative to the previous layer
        dW - numpy.ndarray (n,p) the gradient of W 
        db - numpy.ndarray (n, 1) the gradient of b
    '''
    A_prev = cache["A"]
    ## CODE HERE
    dA_prev = np.dot(W.T, dZ)
    dW = np.dot(dZ, A_prev.T)
    db = np.sum(dZ, axis=1, keepdims=True)
    return dA_prev, dW, db

def layer_backward(dA, cache, W, b, activation):
    '''
    Backward propagation through the activation and linear layer

    Inputs:
        dA - numpy.ndarray (n,m) the derivative to the previous layer
        cache - dictionary containing the linear_cache and the activation_cache
        activation - activation of the layer
        W - numpy.ndarray (n,p)
        b - numpy.ndarray (n, 1)
    
    Returns:
        dA_prev - numpy.ndarray (p,m) the derivative to the previous layer
        dW - numpy.ndarray (n,p) the gradient of W 
        db - numpy.ndarray (n, 1) the gradient of b
    '''
    lin_cache = cache["lin_cache"]
    act_cache = cache["act_cache"]

    if activation == "sigmoid":
        dZ = sigmoid_der(dA, act_cache)
    elif activation == "tanh":
        dZ = tanh_der(dA, act_cache)
    elif activation == "relu":
        dZ = relu_der(dA, act_cache)
    elif activation == "linear":
        dZ = linear_der(dA, act_cache)
    dA_prev, dW, db = linear_backward(dZ, lin_cache, W, b)
    return dA_prev, dW, db

def multi_layer_backward(dAL, caches, parameters):
    '''
    Back propgation through the layers of the network (except softmax cross entropy)
    softmax_cross_entropy can be handled separately

    Inputs: 
        dAL - numpy.ndarray (n,m) derivatives from the softmax_cross_entropy layer
        caches - a dictionary of associated caches of parameters and network inputs
        parameters - dictionary of network parameters {"W1":[..],"b1":[..],"W2":[..],"b2":[..]...}

    Returns:
        gradients - dictionary of gradient of network parameters 
            {"dW1":[..],"db1":[..],"dW2":[..],"db2":[..],...}
    '''
    L = len(caches)  # with one hidden layer, L = 2
    gradients = {}
    dA = dAL
    activation = "linear"
    for l in reversed(range(1,L+1)):
        dA, gradients["dW"+str(l)], gradients["db"+str(l)] = \
                    layer_backward(dA, caches[l-1], \
                    parameters["W"+str(l)],parameters["b"+str(l)],\
                    activation)
        activation = "relu"
    return gradients

def classify(X, parameters):
    '''
    Network prediction for inputs X

    Inputs: 
        X - numpy.ndarray (n,m) with n features and m samples
        parameters - dictionary of network parameters 
            {"W1":[..],"b1":[..],"W2":[..],"b2":[..],...}
    Returns:
        YPred - numpy.ndarray (1,m) of predictions
    '''
    ### CODE HERE 
    # Forward propagate X using multi_layer_forward
    # Get predictions using softmax_cross_entropy_loss
    # Estimate the class labels using predictions
    A1, cache = multi_layer_forward(X, parameters)
        ## call to softmax cross entropy loss
    # A2, cache1, cost = softmax_cross_entropy_loss(A1, Y)
    # A2 = A2.T
    Ypred = []
    Z=A1
    iter1 = Z.shape[1]
    Z = Z.T
    e_z = Z
    for i in range(iter1):
        print(Z[i])
        e_z[i] = np.exp(Z[i] - np.max(Z[i]))
        e_z[i] = e_z[i]/np.sum(e_z[i])

    A = e_z
    for i in range(A.shape[0]):
        #print(A2[i])
        Ypred.append(np.argmax(A[i]))
    print(Ypred)
    return Ypred
    AL, _ = multi_layer_forward(X, parameters)
    A, _, _ = softmax_cross_entropy_loss(AL)
    labels = np.argmax(A, axis=0)
    Ypred = labels.reshape(1, len(labels))
    return Ypred

def update_parameters(parameters, gradients, epoch, learning_rate, decay_rate=0.01):
    '''
    Updates the network parameters with gradient descent

    Inputs:
        parameters - dictionary of network parameters 
            {"W1":[..],"b1":[..],"W2":[..],"b2":[..],...}
        gradients - dictionary of gradient of network parameters 
            {"dW1":[..],"db1":[..],"dW2":[..],"db2":[..],...}
        epoch - epoch number
        learning_rate - step size for learning
        decay_rate - rate of decay of step size - not necessary - in case you want to use
    '''
    alpha = learning_rate*(1/(1+decay_rate*epoch))
    L = len(parameters)//2

    parameters["W4"] = parameters["W4"]+ - alpha * gradients["dW4"]
    parameters["b4"] = parameters["b4"] - alpha * gradients["db4"]
    ### CODE HERE 
    return parameters, alpha



def softmax_cross_entropy_loss(Z, Y=np.array([])):
    '''
    Computes the softmax activation of the inputs Z
    Estimates the cross entropy loss

    Inputs: 
        Z - numpy.ndarray (n, m)
        Y - numpy.ndarray (1, m) of labels
            when y=[] loss is set to []
    
    Returns:
        A - numpy.ndarray (n, m) of softmax activations
        cache -  a dictionary to store the activations later used to estimate derivatives
        loss - cost of prediction
    '''
    

    maximumZ = np.max(Z, axis = 0, keepdims = True)
    e = np.exp(Z - maximumZ)
    A = e / np.sum(e, axis = 0, keepdims = True)
    cache = {}
    cache["A"] = A
    
    loss = 0
    for i in range(Y.shape[1]):
        x = int(Y[0][i])
        loss += np.log(A[x, i])

    loss = -loss/Y.shape[1]
    return A,cache,loss

def sigmoid(Z):

    A = 1/(1+np.exp(-Z))
    cache = {}
    cache["Z"] = Z
    return A, cache

def sigmoid_der(dA, cache):

    Z = cache["Z"]
    A, cache = sigmoid(Z)
    dZ = dA * A * (1 - A)

    return dZ

def initialize_2layer_weights(n_in, n_h, n_fin):

    W1 = np.random.randn(n_h, n_in) * 0.01
    b1 = np.random.randn(n_h, 1) * 0.01
    W2 = np.random.randn(n_fin, n_h) * 0.01
    b2 = np.random.randn(n_fin, 1) * 0.01

    parameters = {}
    parameters["W1"] = W1
    parameters["b1"] = b1
    parameters["W2"] = W2
    parameters["b2"] = b2

    return parameters

def linear_forward(A, W, b):

    Z = np.dot(W, A) + b
    cache = {}
    cache["A"] = A
    return Z, cache

def layer_forward(A_prev, W, b, activation):

    Z, lin_cache = linear_forward(A_prev, W, b)
    if activation == "sigmoid":
        A, act_cache = sigmoid(Z)
    elif activation == "tanh":
        A, act_cache = tanh(Z)
    elif activation == "relu":
        A, act_cache = relu(Z)
    elif activation == "linear":
        A, act_cache = linear(Z)
    
    cache = {}
    cache["lin_cache"] = lin_cache
    cache["act_cache"] = act_cache

    return A, cache

def cost_estimate(A2, Y):
    m = Y.shape[1]
    cost = -1 * (1/m) * ( np.sum( np.multiply(np.log(A2),Y) ) + np.sum( np.multiply(np.log(1-A2),(1-Y)) ) )

    return cost

def linear_backward(dZ, cache, W, b):

    dW = np.dot(dZ, cache["A"].T)
    db = np.sum(dZ, axis=1, keepdims=True)
    dA_prev = np.dot(W.T, dZ)

    return dA_prev, dW, db

# def layer_backward(dA, cache, W, b, activation):

#     lin_cache = cache["lin_cache"]
#     act_cache = cache["act_cache"]

#     if activation == "sigmoid":
#         dZ = sigmoid_der(dA, act_cache)
#     elif activation == "tanh":
#         dZ = tanh_der(dA, act_cache)
#     dA_prev, dW, db = linear_backward(dZ, lin_cache, W, b)
#     return dA_prev, dW, db


def salt_and_pepper_noise(image,prob):
    output = np.copy(image)
    thres = 1 - prob 
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            rdn = random.random()
            if rdn < prob:
                output[i][j] = 0
            elif rdn > thres:
                output[i][j] = 1
    return output

def classify(X, parameters):
    '''
    Network prediction for inputs X

    Inputs: 
        X - numpy.ndarray (n,m) with n features and m samples
        parameters - dictionary of network parameters 
            {"W1":[..],"b1":[..],"W2":[..],"b2":[..],...}
    Returns:
        YPred - numpy.ndarray (1,m) of predictions
    '''
    ### CODE HERE 
    # Forward propagate X using multi_layer_forward
    # Get predictions using softmax_cross_entropy_loss
    # Estimate the class labels using predictions
    A1, cache = multi_layer_forward(X, parameters)
        ## call to softmax cross entropy loss
    # A2, cache1, cost = softmax_cross_entropy_loss(A1, Y)
    # A2 = A2.T
    Ypred = []
    Z=A1
    iter1 = Z.shape[1]
    Z = Z.T
    e_z = Z
    for i in range(iter1):
        print(Z[i])
        e_z[i] = np.exp(Z[i] - np.max(Z[i]))
        e_z[i] = e_z[i]/np.sum(e_z[i])

    A = e_z
    for i in range(A.shape[0]):
        #print(A2[i])
        Ypred.append(np.argmax(A[i]))
    print(Ypred)
    return Ypred
    AL, _ = multi_layer_forward(X, parameters)
    A, _, _ = softmax_cross_entropy_loss(AL)
    labels = np.argmax(A, axis=0)
    Ypred = labels.reshape(1, len(labels))
    return Ypred

def two_layer_network(X, Y, noisy_X, test_X, test_Y, net_dims, num_iterations=2000, learning_rate=0.1):

    n_in, n_h, n_fin = net_dims
    parameters = initialize_2layer_weights(n_in, n_h, n_fin)
    
    A0 = noisy_X
    train_costs = []
    validation_costs = []
    test_costs = []
    for ii in range(num_iterations):
        # Forward propagation
        ### CODE HERE
        A1, cache1 = layer_forward(A0, parameters["W1"], parameters["b1"], "sigmoid")
        A2, cache2 = layer_forward(A1, parameters["W2"], parameters["b2"], "sigmoid")

        # cost estimation
        train_cost = cost_estimate(A2, X)

        # test_A0 = test_X
        # test_A1, test_cache1 = layer_forward(test_A0, parameters["W1"], parameters["b1"], "sigmoid")
        # test_A2, test_cache2 = layer_forward(test_A1, parameters["W2"], parameters["b2"], "sigmoid")
        # test_cost = cost_estimate(test_A2, test_Y)

        m = Y.shape[1]
        
        # Backward Propagation
        ### CODE HERE
        #dA2 = (1.0/m) * (np.divide(-1*X, A2) + np.divide(1-X, 1-A2))
        dA2 = float(-1.0/A0.shape[1]) * ((X/A2) - ((X-1.0)/(A2-1.0)))
        dA1, dW2, db2 = layer_backward(dA2, cache2, parameters["W2"], parameters["b2"], "sigmoid")
        dA0, dW1, db1 = layer_backward(dA1, cache1, parameters["W1"], parameters["b1"], "sigmoid")
        # dZ2 = sigmoid_der(dA2, cache2)
        # dW2 = np.dot(dZ2, A1.T)
        # db2 = np.sum(dZ2, axis=1, keepdims=True)

        #update parameters
        ### CODE HERE
        parameters["W2"] = parameters["W2"] - learning_rate * dW2 
        parameters["b2"] = parameters["b2"] - learning_rate * db2 
        parameters["W1"] = parameters["W1"] - learning_rate * dW1
        parameters["b1"] = parameters["b1"] - learning_rate * db1 

        if ii % 10 == 0:
            train_costs.append(train_cost)
        if ii % 1 == 0:
            print ("Train Cost at iteration %i is: %f" %(ii, train_cost))
            # print(cache1)
            # print(cache2)
    
    return train_costs, test_costs, validation_costs, parameters
def accuracy(train_Pred,train_label):
    count_train_errors = 0
    for i in range(len(train_Pred[0])):
        # print(train_Pred[0][i])
        # print(train_label[0][i])
        #print (train_Pred[0][i]," ",train_label[0][i])
        if train_Pred[0][i] == train_label[0][i] :
            count_train_errors = count_train_errors + 1
            print (train_Pred[0][i]," ",train_label[0][i])
            
    
    #print(count_train_errors)
    trAcc = count_train_errors/len(train_label[0]) * 100
    return trAcc
def accuracy_multi(train_Pred,train_label):
    count_train_errors = 0
    for i in range(len(train_Pred)):
        # print(train_Pred[0][i])
        # print(train_label[0][i])
        #print (train_Pred[i]," ",train_label[0][i])
        if train_Pred[i] == train_label[0][i] :
            count_train_errors = count_train_errors + 1
            #print (train_Pred[i]," ",train_label[0][i])
            
    
    #print(count_train_errors)
    trAcc = count_train_errors/len(train_label[0]) * 100
    return trAcc
def multi_layer_network(X, Y, net_dims,parameters, num_iterations=500, learning_rate=0.5, decay_rate=0.01):
    '''
    Creates the multilayer network and trains the network

    Inputs:
        X - numpy.ndarray (n,m) of training data
        Y - numpy.ndarray (1,m) of training data labels
        net_dims - tuple of layer dimensions
        num_iterations - num of epochs to train
        learning_rate - step size for gradient descent
    
    Returns:
        costs - list of costs over training
        parameters - dictionary of trained network parameters
    '''
    #parameters = initialize_multilayer_weights(net_dims)
    A0 = X
    costs = []
    valid_costs =[]
    alpha = learning_rate
    for ii in range(num_iterations):
        ### CODE HERE
        # Forward Prop 
        ## call to multi_layer_forward to get activations
        A1, cache = multi_layer_forward(A0, parameters)
        ## call to softmax cross entropy loss
        A2, cache1, cost = softmax_cross_entropy_loss(A1, Y)

        #validationcost
        #VALIDATION
        #V_f,_ = multi_layer_forward(valid_x,parameters)
        ## call to softmax cross entropy loss
       # _,_,v_loss = softmax_cross_entropy_loss(V_f,valid_y)
        

        # Backward Prop
        ## call to softmax cross entropy loss der
        dZ = softmax_cross_entropy_loss_der(Y, cache1)
        ## call to multi_layer_backward to get gradients
        gradients = multi_layer_backward(dZ, cache, parameters)
        ## call to update the parameters
        parameters, alpha = update_parameters(parameters, gradients, ii, learning_rate, decay_rate=decay_rate)
        if ii % 10 == 0:
            costs.append(cost)
            #valid_costs.append(v_loss)
        if ii % 10 == 0:
            print("Cost at iteration %i is: %.05f, learning rate: %.05f" %(ii, cost, alpha))
    
    return costs, parameters,valid_costs

def main():
    # getting the subset dataset from Fashion MNIST
    class_range = [0,1,2,3,4,5,6,7,8,9]
    class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
    train_data, train_label, test_data, test_label = \
            fashion_mnist(noTrSamples=5000,noTsSamples=600,\
            class_range=class_range,\
            noTrPerClass=500, noTsPerClass=60)

    print("train data", train_data.shape)
    print("test data",test_data.shape)
    print(train_label)

    plt.figure(figsize=(10,10))

    for i in range(25):
        plt.subplot(5,5,i+1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(train_data[:,i].reshape(28, 28), cmap=plt.cm.binary)
        #plt.xlabel(class_names[train_label[i]])
    plt.show()

    n_in, m = train_data.shape
    n_fin = 784
    n_h = 500
   
    # initialize learning rate and num_iterations
    learning_rate = 0.01
    num_iterations = 500

    noisyTrdata = np.zeros((n_in, m))
    columns = train_data.shape[1]
    for i in range(0, columns):
        noisy_image = salt_and_pepper_noise(train_data[:,i].reshape(28,28), 0.1)
        noisy_image1 = noisy_image.reshape(784, 1)
        for j in range(noisy_image1.shape[0]):
            noisyTrdata[j][i] = noisy_image1[j][0]

    for i in range(25):
        plt.subplot(5,5,i+1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(noisyTrdata[:,i].reshape(28, 28), cmap=plt.cm.binary)
        #plt.xlabel(class_names[train_label[i]])
    plt.show()

    #layer 1
    net_dims = [n_in, n_h, n_fin]
    train_costs, val_costs, test_costs, parameters_1 = two_layer_network(train_data, train_label, train_data, test_data, test_label, net_dims, num_iterations=num_iterations, learning_rate=learning_rate)
    # print(train_costs)
    # print(val_costs)
    # print(test_costs)

    #layer 2
    
    A1, cache1 = layer_forward(train_data, parameters_1["W1"], parameters_1["b1"], "sigmoid")
    A2,cache2 =  layer_forward(A1, parameters_1["W2"], parameters_1["b2"], "sigmoid")

    for i in range(25):
        plt.subplot(5,5,i+1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(A2[:,i].reshape(28, 28), cmap=plt.cm.binary)
        #plt.xlabel(class_names[int(train_label[i])])
    plt.show()

   
    n_in, m = A1.shape
    n_fin = 500
    n_h = 300
    
    net_dims_2 = [n_in, n_h, n_fin]
    
    train_cost_2,val_costs_2,test_costs_2,parameters_2 = two_layer_network(A1, train_label, A1, test_data, test_label, net_dims_2, num_iterations=num_iterations, learning_rate=learning_rate)
    A_2, cache_2 = layer_forward(A1, parameters_2["W1"], parameters_2["b1"], "sigmoid")
    parameters_1["W2"]= parameters_2["W1"]
    parameters_1["b2"] = parameters_2["b1"]

    ##layer 3
    n_in, m = A_2.shape
    n_fin = 300
    n_h = 100
    
    net_dims_3 = [n_in, n_h, n_fin]

    train_cost_2,val_costs_2,test_costs_2,parameters_3 = two_layer_network(A_2, train_label, A_2, test_data, test_label, net_dims_3, num_iterations=num_iterations, learning_rate=learning_rate)
    A_3, cache_3 = layer_forward(A_2, parameters_3["W1"], parameters_3["b1"], "sigmoid")

    parameters_1["W3"]= parameters_3["W1"]
    parameters_1["b3"] = parameters_3["b1"]

    #layer 4
    n_in, m = A_3.shape
    n_fin = 100
    n_h = 10
    
    net_dims_4 = [n_in, n_h, n_fin]

    train_cost_2,val_costs_2,test_costs_2,parameters_4 = two_layer_network(A_3, train_label, A_3, test_data, test_label, net_dims_4, num_iterations=num_iterations, learning_rate=learning_rate)
    A_4, cache_4 = layer_forward(A_2, parameters_3["W1"], parameters_3["b1"], "sigmoid")

    parameters_1["W4"]= parameters_4["W1"]
    parameters_1["b4"] = parameters_4["b1"]


    #accuracy without fine tuning
    A,cache,loss = softmax_cross_entropy_loss(A_4,train_label)
    print(train_label.shape," train_label ")
    labels = np.argmax(A, axis=0)
    Ypred = labels.reshape(1, len(labels))
    print(len(Ypred))
    print(Ypred.shape, "Ypred Shape")
    print(accuracy(Ypred,train_label))

    #multilayer training fine tuning
    net_dims_5 = [784,500,300,100,10]
    fine_tune_data, fine_label, fine_test_data, fine_test_label = \
            fashion_mnist(noTrSamples=50,noTsSamples=600,\
            class_range=class_range,\
            noTrPerClass=5, noTsPerClass=60)
    num_iterations_2 = 500
    costs, parameters,valid_costs = multi_layer_network(fine_tune_data,fine_label, net_dims_5,parameters_1, \
            num_iterations=num_iterations_2, learning_rate=0.1)
    test_Pred = classify(test_data, parameters)
    
    print(accuracy_multi(test_Pred,test_label))









    # for i in range(25):
    #     plt.subplot(5,5,i+1)
    #     plt.xticks([])
    #     plt.yticks([])
    #     plt.grid(False)
    #     plt.imshow(A2[:,i].reshape(28, 28), cmap=plt.cm.binary)
    #     #plt.xlabel(class_names[int(train_label[i])])
    # plt.show()

    
    
    



if __name__ == "__main__":
    main()