'''
Python version - 3.6
'''
import numpy as np
from load_mnist import fashion_mnist
import matplotlib.pyplot as plt
import pdb
import random

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

def layer_backward(dA, cache, W, b, activation):

    lin_cache = cache["lin_cache"]
    act_cache = cache["act_cache"]

    if activation == "sigmoid":
        dZ = sigmoid_der(dA, act_cache)
    elif activation == "tanh":
        dZ = tanh_der(dA, act_cache)
    dA_prev, dW, db = linear_backward(dZ, lin_cache, W, b)
    return dA_prev, dW, db


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

def two_layer_network(X, Y, noisy_X, test_X, test_Y, net_dims, num_iterations=2000, learning_rate=0.1):

    n_in, n_h, n_fin = net_dims
    parameters = initialize_2layer_weights(n_in, n_h, n_fin)
    
    A0 = noisy_X
    train_costs = []
    validation_costs = []
    test_costs = []
    final_data = X
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
        dA2 = (1.0/m) * (np.divide(-1*X, A2) + np.divide(1-X, 1-A2))
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
        if ii % 100 == 0:
            print ("Train Cost at iteration %i is: %f" %(ii, train_cost))
            # print(cache1)
            # print(cache2)
        final_data = A2
    
    return train_costs, test_costs, validation_costs, parameters, final_data

def deNoise(test_data, parameters):
    A0 = test_data
    A1, cache1 = layer_forward(A0, parameters["W1"], parameters["b1"], "sigmoid")
    A2, cache2 = layer_forward(A1, parameters["W2"], parameters["b2"], "sigmoid")
    return A2

def addNoise(data, noise):
    noisyTrdata = np.zeros(data.shape)
    columns = data.shape[1]
    for i in range(0, columns):
        noisy_image = salt_and_pepper_noise(data[:,i].reshape(28,28), noise)
        noisy_image1 = noisy_image.reshape(784, 1)
        for j in range(noisy_image1.shape[0]):
            noisyTrdata[j][i] = noisy_image1[j][0]
    return noisyTrdata


def main():
    # getting the subset dataset from Fashion MNIST
    class_range = [0,1,2,3,4,5,6,7,8,9]
    class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
    train_data, train_label, test_data, test_label = \
            fashion_mnist(noTrSamples=1500,noTsSamples=60,\
            class_range=class_range,\
            noTrPerClass=150, noTsPerClass=6)

    print("train data", train_data.shape)
    print("test data",test_data.shape)

    plt.figure(figsize=(10,10))

    # for i in range(25):
    #     plt.subplot(5,5,i+1)
    #     plt.xticks([])
    #     plt.yticks([])
    #     plt.grid(False)
    #     plt.imshow(train_data[:,i].reshape(28, 28), cmap=plt.cm.binary)
    #     #plt.xlabel(class_names[train_label[i]])
    # plt.show()

    n_in, m = train_data.shape
    n_fin = 784
    n_h = 256
   
    # initialize learning rate and num_iterations
    learning_rate = 0.05
    num_iterations = 1000

    
    noisyTrdata = addNoise(train_data, 0.5)
    noisyTestdata1 = addNoise(test_data, 0.5)
    
    
    # for i in range(10):
    #     plt.subplot(5,5,i+1)
    #     plt.xticks([])
    #     plt.yticks([])
    #     plt.grid(False)
    #     plt.imshow(test_data[:,i*6].reshape(28, 28), cmap=plt.cm.binary)
    #     plt.xlabel(class_names[int(test_label[0][i*6])])
    # plt.show()

    # for i in range(10):
    #     plt.subplot(5,5,i+1)
    #     plt.xticks([])
    #     plt.yticks([])
    #     plt.grid(False)
    #     plt.imshow(noisyTestdata1[:,i*6].reshape(28, 28), cmap=plt.cm.binary)
    #     plt.xlabel(class_names[int(test_label[0][i*6])])
    # plt.show()


    net_dims = [n_in, n_h, n_fin]
    train_costs, val_costs, test_costs, parameters, final_data = two_layer_network(train_data, train_label, noisyTrdata, test_data, test_label, net_dims, num_iterations=num_iterations, learning_rate=learning_rate)
    # print(train_costs)
    # print(val_costs)
    A1 = deNoise(noisyTestdata1, parameters)
    A = [A1]
    for j in A:
        for i in range(10):
            plt.subplot(5,5,i+1)
            plt.xticks([])
            plt.yticks([])
            plt.grid(False)
            plt.imshow(j[:,i*6].reshape(28, 28), cmap=plt.cm.binary)
            plt.xlabel(class_names[int(test_label[0][i*6])])
        plt.show()

    plt.plot(train_costs,label = "Training  cost")
    plt.ylabel('Cost')
    plt.xlabel('Iterations')
    plt.title("Training Cost vs Iterations")
    plt.show()
   
    
   
    
    
    



if __name__ == "__main__":
    main()