import numpy as np
import gym
import matplotlib.pyplot as plt
from math import inf
import os

class Target_network:

    def __init__(self):
        pass
        
    def sigmoid(self,Z):
        return 1/(1 + np.exp(-Z)) , Z

    def relu(self,Z):
        return np.maximum(0,Z) , Z

    def initialize_parameters_deep(self, layer_dims):

        parameters = {}
        L = len(layer_dims)            

        for l in range(1, L):
            parameters['W' + str(l)] = np.random.randn(layer_dims[l], layer_dims[l - 1]) * 0.01
            parameters['b' + str(l)] = np.zeros((layer_dims[l], 1))
            
        return parameters

    def linear_forward(self,A, W, b):

        # print('A shape in linear forward', A.shape)
        # print("W shape : ", W.shape)
        Z = np.dot(W, A) + b
        # print("b shape : ", b.shape)
        # print("z shape: " , Z.shape)
        cache = (A, W, b)
        
        return Z, cache

    def linear_activation_forward(self, A_prev, W, b, activation):
        
        if activation == "sigmoid":
            Z, linear_cache = self.linear_forward(A_prev, W, b)
            A, activation_cache = self.sigmoid(Z)
        
        elif activation == "relu":
            Z, linear_cache = self.linear_forward(A_prev, W, b)
            A, activation_cache = self.relu(Z)
        
        cache = (linear_cache, activation_cache)

        return A, cache
        
    def L_model_forward(self, X, parameters):

        caches = []
        A = X
        L = len(parameters) // 2                  # number of layers in the neural network

        # print(L)
        # Implement [LINEAR -> RELU]*(L-1). Add "cache" to the "caches" list.
        for l in range(1, L):
            # print(A.shape)
            A_prev = A 
            A, cache = self.linear_activation_forward(A_prev, parameters['W' + str(l)], parameters['b' + str(l)], activation='relu')
            caches.append(cache)    
        
        # Implement LINEAR -> SIGMOID. Add "cache" to the "caches" list.
        AL, cache = self.linear_activation_forward(A, parameters['W' + str(L)], parameters['b' + str(L)], activation='sigmoid')
        caches.append(cache)

        return AL, caches

class Prediction_network(Target_network):

    def __init__(self):
        pass

    def sigmoid_backward(self, dA, cache):

        Z = cache 
        s = 1/(1+np.exp(-Z))
        dZ = dA * s * (1-s)
        return dZ

    def relu_backward(self, dA, cache):

        Z = cache
        # just converting dz to a correct object.
        dZ = np.array(dA, copy=True)
        # When z <= 0, we should set dz to 0 as well. 
        dZ[Z <= 0] = 0
        return dZ

    def compute_cost(self, AL, Y):
        
        # print(Y)
        Y = np.ones((AL.shape[0],1)) * Y
        # print(Y)
        cost = -np.sum(np.multiply(Y, np.log(AL)) + np.multiply(1 - Y, np.log(1 - AL)))
        cost = np.squeeze(cost)      # To make sure your cost's shape is what we expect (e.g. this turns [[17]] into 17).
        
        return cost

    def linear_backward(self, dZ, cache):

        A_prev, W, b = cache

        # print("dZ shape",dZ.shape)
        dW = np.dot(dZ, cache[0].T)
        db = dZ
        dA_prev = np.dot(cache[1].T, dZ)
        
        return dA_prev, dW, db

    def linear_activation_backward(self, dA, cache, activation):

        linear_cache, activation_cache = cache   
        
        if activation == "relu":
            dZ = self.relu_backward(dA, activation_cache)
        elif activation == "sigmoid":
            dZ = self.sigmoid_backward(dA, activation_cache)
        
        dA_prev, dW, db = self.linear_backward(dZ, linear_cache)
        
        return dA_prev, dW, db

    def L_model_backward(self, AL, Y, caches):

        grads = {}
        L = len(caches) # the number of layers
        # m = AL.shape[0]
        Y = np.ones((AL.shape[0],1)) * Y
        # Y = Y.reshape(AL.shape) # after this line, Y is the same shape as AL
        # print(L)
        dAL = - (np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))

        current_cache = caches[-1]
        grads["dA" + str(L-1)], grads["dW" + str(L)], grads["db" + str(L)] = self.linear_activation_backward(dAL, current_cache, "sigmoid")
        
        for l in reversed(range(L-1)):
            # lth layer: (RELU -> LINEAR) gradients.
            # Inputs: "grads["dA" + str(l + 2)], caches". Outputs: "grads["dA" + str(l + 1)] , grads["dW" + str(l + 1)] , grads["db" + str(l + 1)] 
            current_cache = caches[l]
            dA_prev_temp, dW_temp, db_temp = self.linear_activation_backward(grads["dA" + str(l+1)], current_cache, "relu")
            grads["dA" + str(l)] = dA_prev_temp
            grads["dW" + str(l + 1)] = dW_temp
            grads["db" + str(l + 1)] = db_temp

        return grads

    def update_parameters(self, parameters, grads, learning_rate):
        
        L = len(parameters) // 2 # number of layers in the neural network

        # Update rule for each parameter. Use a for loop.
        for l in range(L):
            parameters["W" + str(l + 1)] = parameters["W" + str(l + 1)] - learning_rate * grads["dW" + str(l + 1)]
            parameters["b" + str(l + 1)] = parameters["b" + str(l + 1)] - learning_rate * grads["db" + str(l + 1)]
            
        return parameters

def epsilon_greedy_action(Q,eps):
    if np.random.random() < eps: # explore
        # print(type(Q))
        return np.random.randint(9)
    else: # exploit
        return np.random.choice(np.flatnonzero(Q == Q.max()))

env = gym.make('MsPacman-v0')
observation = env.reset()
state = observation.reshape(-1,1)

target_network = Target_network()
prediction_network = Prediction_network()
learning_rate = 0.0075
gamma = 0.8
c = 100 # not sure
eps = 0.5

layer_dims = (observation.size,30,20,10,env.action_space.n)
parameters = prediction_network.initialize_parameters_deep(layer_dims)
target_parameters = parameters

N_episode = 10000
for i_episode in range(N_episode):
    t = 0
    while(True):
        # print(t)
        env.render()
        if(i_episode % c == 0): target_parameters = parameters
        # print(state.shape)
        Q, caches = prediction_network.L_model_forward(state, parameters)
        # print(Q)
        action = epsilon_greedy_action(Q,eps)
        observation , reward , done , _ = env.step(action)
        state = observation.reshape(-1,1)
        next_Q, _ = target_network.L_model_forward(state, target_parameters)
        target = reward + gamma * next_Q.max()
        cost = prediction_network.compute_cost(Q, target)
        grads =  prediction_network.L_model_backward(Q, target, caches)
        parameters = prediction_network.update_parameters(parameters, grads, learning_rate)
        # print(parameters)
        t+=1

        if done:
            print("Episode finished after {} timesteps".format(t+1))
            break

env.close()