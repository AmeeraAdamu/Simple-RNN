#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np

# Initialize parameters
def initialize_parameters(input_size, hidden_size, output_size):
    parameters = {
        "Wxh": np.random.randn(hidden_size, input_size) * 0.01,  # input to hidden
        "Whh": np.random.randn(hidden_size, hidden_size) * 0.01, # hidden to hidden
        "Why": np.random.randn(output_size, hidden_size) * 0.01, # hidden to output

        "bh": np.zeros((hidden_size, 1)),                        # hidden bias
        "by": np.zeros((output_size, 1))                         # output bias
    }
    return parameters

# Forward propagation
def rnn_forward(x, h_prev, parameters):
    Wxh, Whh, Why = parameters["Wxh"], parameters["Whh"], parameters["Why"]
    bh, by = parameters["bh"], parameters["by"]
    
    h_next = np.tanh(np.dot(Wxh, x) + np.dot(Whh, h_prev) + bh)
    y = np.dot(Why, h_next) + by
    
    cache = (h_next, h_prev, x, parameters)
    
    return y, h_next, cache

# Backward propagation (one step)
def rnn_backward(dy, cache):
    (h_next, h_prev, x, parameters) = cache
    Wxh, Whh, Why = parameters["Wxh"], parameters["Whh"], parameters["Why"]
    
    # Gradient of output
    dWhy = np.dot(dy, h_next.T)
    dby = dy
    
    # Gradient through hidden state
    dh = np.dot(Why.T, dy)
    dh_raw = (1 - h_next ** 2) * dh  # derivative of tanh
    
    dWxh = np.dot(dh_raw, x.T)
    dWhh = np.dot(dh_raw, h_prev.T)
    dbh = dh_raw
    
    dh_prev = np.dot(Whh.T, dh_raw)

    grads = {
        "dWxh": dWxh,
        "dWhh": dWhh,
        "dWhy": dWhy,
        "dbh": dbh,
        "dby": dby,
        "dh_prev": dh_prev
    }
    
    return grads

# Example to test
np.random.seed(1)
input_size = 3
hidden_size = 5
output_size = 2

parameters = initialize_parameters(input_size, hidden_size, output_size)
x = np.random.randn(input_size, 1)
h_prev = np.zeros((hidden_size, 1))

y, h_next, cache = rnn_forward(x, h_prev, parameters)

# Assume random gradient coming from the next layer
dy = np.random.randn(output_size, 1)

grads = rnn_backward(dy, cache)

print("Forward output y:", y)
print("Gradients:", grads)


# In[ ]:





# In[ ]:




