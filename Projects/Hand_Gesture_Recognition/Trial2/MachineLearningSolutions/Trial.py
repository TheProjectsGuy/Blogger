# import numpy as np
# import pickle
# import os
#
# os.system('pwd')
#
# f = open("NeuralNetworks/Results/NN2", 'rb')
# nn_info = pickle.load(f)
# print(type(nn_info))
# print(nn_info)
# (nn_arc, params) = nn_info
# print(type(nn_arc))
# print(nn_arc)
# print(type(params))
# print(params)
# f.close()

import numpy as np

a = np.array([[1,2,3],[6,5,4],[8,9,0]])
print(a)
print(a.sum(axis=1,keepdims=True))
