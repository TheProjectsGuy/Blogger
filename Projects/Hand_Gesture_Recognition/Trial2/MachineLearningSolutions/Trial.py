import numpy as np
import pickle
import os

os.system('pwd')

f = open("NeuralNetworks/Results/NN2", 'rb')
nn_info = pickle.load(f)
print(type(nn_info))
print(nn_info)
(nn_arc, params) = nn_info
print(type(nn_arc))
print(nn_arc)
print(type(params))
print(params)
f.close()
