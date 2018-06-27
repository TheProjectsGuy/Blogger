# import numpy as np
# arr = np.arange(30).reshape((6, 5))
# print(arr)
# sum_mat = np.sum(arr, axis=1, keepdims=True)
# print(sum_mat)
# arr = np.arange(5).reshape((1, 5))
# print(np.sum(arr, axis=1, keepdims=True))
# print("------")
# X = np.arange(6).reshape((3,2))
# print(X)
# W = np.array([0.2, 0.5, 0.6]).reshape((1, -1))
# b = np.array([3])
# def sigmoid(x):
#     # Return sigmoid of an input
#     return np.exp(x)/(1 + np.exp(x))
# ans = sigmoid(np.matmul(W,X) + b)
# print(ans)
# print('-------')
import matplotlib.pyplot as plt
import numpy as np

a = np.array([1,4,3,5,2,6,3,5,3,1])
plt.plot(a)
plt.grid('on')
plt.show()
print(np.arange(10))