"""
Perceptron algorithm to detect 5 finger gesture

Made my : Avneesh Mishra
Test error = 19.8529%
"""
import matplotlib.pyplot as plt
import numpy as np
import os

# Not important right now (until approval)
results_folder = 'Results/Perceptron_5'


def load_data_variables():
    # Load 47 x 38 images
    X = np.load('Data/X.npy')
    Y_one_hot = np.load('Data/Y_one_hot_encoded.npy')
    Y = np.array(Y_one_hot[5,:])
    Y = Y.reshape((1,-1))
    print("DATA DEBUG : Viewing 100 examples")
    view_100_examples(X)
    # Normalize data
    X = X/255
    return X,Y


def view_100_examples(X):
    # View 100 random 47 x 38 images from the dataset
    for i in range(1, 101):
        rno = np.random.randint(X.shape[1])
        img = X[:,rno].reshape(47, 38)
        plt.subplot(10, 10, i)
        plt.imshow(img, cmap='Greys')
        plt.axis('off')
    plt.suptitle("Random Images")
    plt.show()

print("INFO : Loading variables")
X, Y = load_data_variables()
print("DATA DEBUG : Variables loaded, input shape is {x_shape}, output shape is {y_shape}".format(
    x_shape=X.shape, y_shape= Y.shape
))


def shuffle_data(X, Y, iter = 1):
    # Shuffle data
    print("DATA DEBUG : Shuffling data {num_iter} iterations".format(num_iter=iter))
    sh_mat = X
    sh_mat = np.row_stack((sh_mat, Y))
    sh_mat = sh_mat.T
    for i in range(iter):
        np.random.shuffle(sh_mat)
    sh_mat = sh_mat.T
    X = sh_mat[0:X.shape[0], :]
    Y = sh_mat[X.shape[0]:X.shape[0] + Y.shape[0], :]
    return np.array(X), np.array(Y)


def generate_train_dev_test(X, Y, sizes=None):
    # Split given data into train, dev and test sets
    if sizes is None:
        size_train = 2500
        size_dev = 136
        size_test = 136
    else:
        (size_train, size_dev, size_test) = sizes
        assert (size_train + size_test + size_dev <= X.shape[1]), "Size mismatch error {v1}>{v2}".format(
            v1=size_train + size_test + size_dev, v2=X.shape[1]
        )
    x_train = np.array(X[:,:size_train])
    print("DATA DEBUG : x_train is of shape {x_train_shape}".format(
        x_train_shape=x_train.shape
    ))
    y_train = np.array(Y[:,:size_train])
    print("DATA DEBUG : y_train is of shape {y_train_shape}".format(
        y_train_shape=y_train.shape
    ))
    x_dev = np.array(X[:,size_train: size_train + size_dev])
    print("DATA DEBUG : x_dev is of shape {x_dev_shape}".format(
        x_dev_shape=x_dev.shape
    ))
    y_dev = np.array(Y[:, size_train: size_train + size_dev])
    print("DATA DEBUG : y_dev is of shape {y_dev_shape}".format(
        y_dev_shape=y_dev.shape
    ))
    x_test = np.array(X[:, size_train + size_dev: size_train + size_dev + size_test])
    print("DATA DEBUG : x_test is of shape {x_test_shape}".format(
        x_test_shape=x_test.shape
    ))
    y_test = np.array(Y[:, size_train + size_dev: size_train + size_dev + size_test])
    print("DATA DEBUG : y_test is of shape {y_test_shape}".format(
        y_test_shape=y_test.shape
    ))
    rdict = {
        "train": (x_train, y_train),
        "dev": (x_dev, y_dev),
        "test": (x_test, y_test)
    }
    return rdict


X, Y = shuffle_data(X, Y)
data_sets = generate_train_dev_test(X, Y)
(X_train, Y_train) = data_sets["train"]
(X_dev, Y_dev) = data_sets["dev"]
(X_test, Y_test) = data_sets["test"]

# Real perceptron algorithm
# Linear -> Sigmoid


def initialize_parameters(X, Y):
    # Initialize parameters
    n = X.shape[0]
    def initialize_zeros():
        W = np.zeros((1, n))
        b = np.zeros((1, 1))
        rdict = {
            "W": W,
            "b": b
        }
        return rdict
    params = initialize_zeros()
    W = params['W']
    b = params['b']
    W = np.random.rand(*W.shape) * (2/np.sqrt(n - 1))
    b = np.zeros(b.shape)
    print("DATA DEBUG : Weights of shape {w_shape}, sample is {w_sample}".format(
        w_shape=W.shape, w_sample=W
    ))
    print("DATA DEBUG : Bias of shape {b_shape}, sample is {b_sample}".format(
        b_shape=b.shape, b_sample=b
    ))
    params = {
        "W": W,
        "b":b
    }
    return params


params = initialize_parameters(X_train, Y_train)

# Sigmoid functions
def sigmoid(x):
    # Return sigmoid of an input
    return np.exp(x)/(1 + np.exp(x))


print("TEST : Sigmoid (Result should be 0.5) = {sgm_res}".format(sgm_res=sigmoid(0)))


def sigmoid_derivative(x):
    return sigmoid(x) * (1 - sigmoid(x))


print("TEST : Sigmoid Derivative (Result should be 0.235) = {sd_res}".format(
    sd_res=sigmoid_derivative(0.5)))


def forward_propagate(params, X):
    # Forward propagation step
    W = params["W"]
    b = params["b"]
    Z = np.matmul(W, X) + b
    A = sigmoid(Z)
    return A


def generate_forward_propagation_test():
    test_x = np.arange(6).reshape((3, 2))
    test_W = np.array([0.2, 0.5, 0.6]).reshape((1, -1))
    test_b = np.array([3])
    rdict = {
        "W":test_W,
        "b":test_b
    }
    return rdict, test_x


test_params, test_x = generate_forward_propagation_test()
print("TEST : Forward Propagation test (Result should be [[0.9983412, 0.99954738]]) = {fp_res}".format(
    fp_res=forward_propagate(test_params, test_x)
))


def compute_cost(params, X, Y):
    # Compute the cost
    A = forward_propagate(params, X)
    L = np.square(A - Y)
    return np.average(L)/2


def train_parameters_gradient_descent(params, X, Y, num_iter = 100, learning_rate = 0.01, print_cost = True):
    (n, m) = X.shape
    assert (m == Y.shape[1]), "ERROR : Dataset mismatch"
    cost_history = {
        "training_costs": [],
        "test_costs": []
    }
    for i in range(num_iter):
        A = forward_propagate(params, X)
        W = params["W"]
        b = params["b"]
        W = W - (learning_rate/m) * np.sum(X * (A - Y), axis=1, keepdims=True).T
        b = b - (learning_rate/m) * np.sum((A - Y), axis=1, keepdims=True).T
        params["W"] = W
        params["b"] = b
        current_cost = compute_cost(params, X, Y)
        if print_cost and i % 10 == 0:
            print("TRAINING INFO : Iteration {iter}, cost = {cst}".format(
                iter= i, cst=current_cost))
            print("TRAINING DEBUG : W {wsh}, X {xsh}, b {bsh}, A {ash}, Y {ysh}".format(
                    wsh=params["W"].shape, xsh=X.shape, bsh=params["b"].shape,
                    ash=A.shape, ysh= Y.shape))
            cost_test = compute_cost(params, X_test, Y_test)
            cost_history["test_costs"].append(cost_test)
        cost_history["training_costs"].append(current_cost)
    return cost_history, params


number_iterations = 100
c_hist, params = train_parameters_gradient_descent(params, X_train, Y_train, num_iter=number_iterations)
plt.plot(c_hist["training_costs"], 'b-')
plt.title('Cost History')
plt.show()
print("TRAINING DEBUG : Training done.")


def error_test_set(params, test_x, test_y, view_mismatches=True):
    # Try training on test set
    (_, m) = test_y.shape
    A = forward_propagate(params, test_x)
    print("TEST DEBUG : W {wsh}, X {xsh}, b {bsh}, Y {ysh}".format(
        wsh=params["W"].shape, xsh= X.shape, bsh= params["b"].shape, ysh= test_y.shape
    ))
    B = np.zeros_like(A)
    B[A > 0.5] = 1
    D = np.square(test_y - B)
    D = np.array(D)
    # Wherever D is 1, it's an error
    mismatches = D.nonzero()[1]
    print("TEST DEBUG : {num_mis} mismatches (out of {num}) found".format(
        num_mis= mismatches.shape[0], num=test_y.shape[1]))
    if view_mismatches == True:
        lx = int(input("Enter grid dimension (lx) : "))
        ly = int(input("Enter grid dimension (ly) : "))
        for i in range(1, lx*ly + 1):
            plt.subplot(lx,ly,i)
            plt.imshow(test_x[:,mismatches[i]].reshape((47,38)), cmap='Greys')
            plt.title(round(A[0 ,mismatches[i]], 1))
            plt.axis('off')
        plt.suptitle("Mismatches")
        plt.show()
    error = np.sum(D)/m
    return error * 100


error = error_test_set(params, X_test, Y_test, view_mismatches= False)
print("TEST DEBUG : Error is {err}%".format(err=error))
