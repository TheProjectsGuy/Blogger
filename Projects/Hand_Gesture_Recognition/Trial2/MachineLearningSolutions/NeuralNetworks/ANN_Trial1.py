import numpy as np
import pickle
import os
from matplotlib import pyplot as plt

np.random.seed(2)


def load_dataset(dataset_dir_name="Data", x_name="X.npy",
                 y_name="Y_one_hot_encoded.npy", one_hot_index=0, shuffle_data=True, normalize_data = True):
    """
    Loads a dataset stored as a .npy file
    :param dataset_dir_name: Name of the folder in which data is
    :param x_name: Name of file which contains inputs (with .npy extension)
    :param y_name: Name of file which contains outputs (with .npy extension)
    :param one_hot_index: The index you're training for (pass -1 for passing raw output file), should be >= -1
    :param shuffle_data: Shuffle the data after getting it from file
    :return:
        X, Y
        X -> Inputs
        Y -> Outputs
        The output will be shuffled if shuffle_data is True, else it won't be shuffled
    """
    # Load the dataset
    X = np.load("../{main_root}/{f_name}".format(main_root=dataset_dir_name,
                                                 f_name=x_name))
    print("DATA DEBUG : Inputs shape is {in_shape}".format(in_shape=X.shape))
    if one_hot_index == -1:
        # Parse the entire dataset as it is
        Y = np.load("../{main_root}/{f_name}".format(main_root=dataset_dir_name,
                                                     f_name=y_name))
    elif one_hot_index >= 0:
        Y_one_hot = np.load("../{main_root}/{f_name}".format(
            main_root=dataset_dir_name, f_name=y_name
        ))
        Y = np.array(Y_one_hot[one_hot_index, :].reshape((1, -1)))
    else:
        raise IndexError("The index {ind_number} is an illegal index".format(ind_number=one_hot_index))
    print("DATA DEBUG : Output shape is {out_shape}".format(out_shape=Y.shape))
    if normalize_data:
        X = X / 255
    # Shuffle the dataset
    def shuffle_dataset(X, Y):
        buffer_data = np.row_stack((X, Y))
        buffer_data = buffer_data.T
        np.random.shuffle(buffer_data)
        buffer_data = buffer_data.T
        X = buffer_data[0:X.shape[0], :]
        Y = buffer_data[X.shape[0]:X.shape[0] + Y.shape[0], :]
        return X, Y

    if shuffle_data:
        X, Y = shuffle_dataset(X, Y)
    return X, Y


def split_train_dev_test(X, Y, sizes=(2500, 136, 136)):
    """
    Split the dataset into train, dev and test sets
    :param X: Inputs
    :param Y: Outputs
    :param sizes: Distribution tuple
    :return: Dictionary
        "train" : (X_train, Y_train)
        "dev" : (X_dev, Y_dev)
        "test" : (X_test, Y_test)
    """
    # Split the dataset
    X_train = X[:, 0:sizes[0]]
    Y_train = Y[:, 0:sizes[0]]
    X_dev = X[:, sizes[0]:sizes[0] + sizes[1]]
    Y_dev = Y[:, sizes[0]:sizes[0] + sizes[1]]
    X_test = X[:, sizes[0] + sizes[1]:sizes[0] + sizes[1] + sizes[2]]
    Y_test = Y[:, sizes[0] + sizes[1]:sizes[0] + sizes[1] + sizes[2]]
    print("DATA DEBUG : Train input shape {in_shape}, output shape {out_shape}".format(
        in_shape=X_train.shape, out_shape=Y_train.shape
    ))
    print("DATA DEBUG : Test input shape {in_shape}, output shape {out_shape}".format(
        in_shape=X_test.shape, out_shape=Y_test.shape
    ))
    print("DATA DEV : Dev input shape {in_shape}, output shape {out_shape}".format(
        in_shape=X_dev.shape, out_shape=Y_dev.shape
    ))
    rdict = {
        "train": (X_train, Y_train),
        "dev": (X_dev, Y_dev),
        "test": (X_test, Y_test)
    }
    return rdict


def init_params_deep(input_size, layer_tup):
    """
    Initialize the parameters of the DNN
    :param input_size: Size of the input layer
    :param layer_tup:  Tuple of layer sizes (hidden layers + output layer)
    :return:
        layers_info, parameters
        :return layer_size
            The size architecture of the neural netowrk
        :return params
        Dictionary
            "W + str(i)" : Weight of layer i
            "b + str(i)" : Biases of layer i
    """
    # Initialize the neural network with parameters
    layer_size = (input_size, *layer_tup)
    print("DATA DEBUG : Initializing neural network with architecture {arc}".format(arc=layer_size))
    params = {}
    for i in range(1, len(layer_size)):
        params["W" + str(i)] = np.random.rand(layer_size[i], layer_size[i - 1]) * 10
        params["b" + str(i)] = np.zeros((layer_size[i], 1))
    return layer_size, params


def parse_activations(activations):
    """
    Parse activations into function and derivatives
    :param activations: Function which returns a dictionary
        "function" : Activation function
        "derivative" : Derivative of function
    :return:
        activation_function, activation_function_derivatives
    """
    activation_fncs = []
    activation_fnc_der = []
    for fnc in activations:
        activation_fncs.append(fnc()["function"])
        activation_fnc_der.append(fnc()["derivative"])
    return activation_fncs, activation_fnc_der


# Activation functions
def sigmoid():
    """
    Sigmoid activation function
    :return:
        Dictionary
            "function" : sigmoid_function
            "derivative" : sigmoid_derivative
    """
    def sigmoid_function(x):
        """
        Sigmoid function
        :param x:
        :return:
            sigmoid_function(x)
        """
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(x):
        """
        Derivative of sigmoid function
        :param x:
        :return:
            sigmoid'(x)
        """
        return sigmoid_function(x) * (1 - sigmoid_function(x))

    func_dict = {
        "function": sigmoid_function,
        "derivative": sigmoid_derivative
    }
    return func_dict


def relu():
    """
    ReLU (Rectified linear unit) activation function
    :param x:
    :return:
        Dictionary
            "function" : relu_function
            "derivative" : relu_derivative
    """
    def relu_function(x):
        """
        ReLU function
        :param x:
        :return:
            relu(x)
        """
        ret_x = x.copy()
        ret_x[ret_x < 0] = 0
        return ret_x

    def relu_derivative(x):
        """
        Derivative of ReLU function
        :param x:
        :return:
            relu'(x)
        """
        ret_d = x.copy()
        ret_d[x <= 0] = 0
        ret_d[x > 0] = 1
        return ret_d

    func_dict = {
        "function": relu_function,
        "derivative": relu_derivative
    }
    return func_dict


# Forward propagation step
def forward_propagate_deep(params, activations, input):
    """
    Perform forward propagation of the neural network
    :param params: Dictionary of weights and biases of the network
            params["W" + str(l)] : Weights of layer l
            params["b" + str(l)] : Biases of layer l
    :param activations: Array of activation functions of every layer
    :param input: Inputs given to the neural network
    :return:
        A_final, cache
        A_final : Final output values of the DNN after forward propagation
        cache : Dictionary
                cache["Z" + str(l)] : The weighed sum
                cache["A" + str(l)] : The value after passing the weighed sums through the activation function
    """
    # Forward propagation process
    cache = {"A0": input}
    L = int(len(params.keys()) / 2)
    for i in range(1, L + 1):
        cache["Z" + str(i)] = params["W" + str(i)] @ cache["A" + str(i - 1)] + params["b" + str(i)]
        act = activations[i - 1]()["function"]
        cache["A" + str(i)] = act(cache["Z" + str(i)])
    A_final = cache["A" + str(L)]
    return A_final, cache


# Cost function
def cost_function(A_pred, Y):
    """
    The Chi Squared cost function employed for knowing the goodness of fit
    :param A_pred: Predictions made
    :param Y: Actual output from the dataset
    :return:
        Cost
    """
    diff_vect = A_pred - Y
    diff_vect = np.square(diff_vect)
    cost_val = np.average(diff_vect)/2
    return cost_val


# Backward propagation gradient
def backward_propagation_grads(cache, params, activations, Y):
    """
    The backward propagation gradient generator
    :param cache: Output and weghed sum of every neuron of every layer in the neural network
            cache["A" + str(l)] : Output of layer l
            cache["Z" + str(l)] : Weighed sum over the previous layer
    :param params: The Weights and Biases of the neural network
            params["W" + str(l)] : Weights of layer l. From layer l-1 to layer l
            params["b" + str(l)] : Biases of layer l
    :param activations: The list of activation functions of each layer
    :param Y: Outputs
    :return: return the gradients (changes to apply)
        Dictionary
            grads["dW" + str(l)] : Gradient of weights of layer l
            grads["db" + str(l)] : Gradient of biases of layer l
    """
    # Main variables
    L = int(len(params.keys()) / 2)  # Number of layers in the neural network
    m = cache["A0"].shape[1]  # Number of training examples
    # Calculate the multiplying factors
    del_vals = {}
    if activations[L - 1] != sigmoid:
        activation_derivative = activations[L-1]()["derivative"]
        del_vals["del" + str(L)] = ((cache["A" + str(L)] - Y) / (m * cache["A" + str(L)] * (1 - cache["A" + str(L)]))) * \
                               activation_derivative(cache["Z" + str(L)])
    else:
        del_vals["del" + str(L)] = ((cache["A" + str(L)] - Y) / (m))
    for l in range(L - 1, 0, -1):  # Go backward from layer L-1 to 1 to calculate del_vals["del" + l]
        activation_derivative = activations[l-1]()["derivative"]
        del_vals["del" + str(l)] = (params["W" + str(l + 1)].T @ del_vals["del" + str(l + 1)]) * \
                                   activation_derivative(cache["Z" + str(l)])
    # Calculate final derivatives
    grads = {}
    # Final backward step
    for l in range(L, 0, -1):
        grads["dW" + str(l)] = del_vals["del" + str(l)] @ cache["A" + str(l - 1)].T
        grads["db" + str(l)] = del_vals["del" + str(l)] @ np.ones((m, 1))

    return grads


# Back propagation step
def back_propagation_deep(cache, params, activations, Y, learning_rate=0.01, reg_lambda = 1000):
    """
    Perform one step of backward propagation
    :param cache: Output and weghed sum of every neuron of every layer in the neural network
            cache["A" + str(l)] : Output of layer l
            cache["Z" + str(l)] : Weighed sum over the previous layer
    :param params: The Weights and Biases of the neural network
            params["W" + str(l)] : Weights of layer l. From layer l-1 to layer l
            params["b" + str(l)] : Biases of layer l
    :param activations: The list of activation functions of each layer
    :param Y: Output
    :param learning_rate: The learning_rate to use
    :return: The new parameters of the neural network
        Dictionary
            params["W" + str(l)] : The weights of layer l
            params["b" + str(l)] : The biases of layer l
    """
    # Backward propagation step
    L = int(len(params.keys()) / 2)
    m = Y.shape[1]
    grads = backward_propagation_grads(cache, params, activations, Y)
    for l in range(L, 0, -1):  # Adjust gradients from layer L to 1
        params["W" + str(l)] = params["W" + str(l)] * (1 - learning_rate * reg_lambda/m)\
                               - learning_rate * grads["dW" + str(l)]
        params["b" + str(l)] = params["b" + str(l)] - learning_rate * grads["db" + str(l)]
    return params


# Test functions
def _test__functions():
    """
    Test functions
    :return:
    """
    def _test_init():
        X, Y = load_dataset(one_hot_index=0)
        data_dict = split_train_dev_test(X, Y)

    def _test_forward_propagation():
        params = init_params_deep(3, (1, 2))
        inp = np.array([[1], [3], [5]])
        activations = [relu, relu]
        Z2, cache = forward_propagate_deep(params, activations, inp)
        print(Z2)


# Load data into memory
X, Y = load_dataset()
datasets = split_train_dev_test(X, Y)
X_train, Y_train = datasets["train"]
X_dev, Y_dev = datasets["dev"]
X_test, Y_test = datasets["test"]
# Load weights and activation functions
architecture_nn, params = init_params_deep(X.shape[0], (50, 50, 1))
activations = [relu, relu, sigmoid]
# Training the network
num_iter = 20
debug_iter_num = 10
cost_tracker = {
    "train_x" : [],
    "train_cost" : [],
    "eval_x" : [],
    "eval_cost" : []
}
# Hyperparameters
learning_rate = 0.01
reg_param_lambda = 1000

for i in range(num_iter):
    # Forward propagate
    A_pred, cache = forward_propagate_deep(params, activations, X_train)
    # Note the cost
    cost_iter = cost_function(A_pred, Y_train)
    cost_tracker["train_x"].append(i)
    cost_tracker["train_cost"].append(cost_iter)
    if (i+1) % debug_iter_num == 0:
        print("TRAIN DEBUG : Cost at iteration {it_num} is {cost}".format(cost=cost_iter, it_num=i+1))
        pred_test, _ = forward_propagate_deep(params, activations, X_test)
        cost_test = cost_function(pred_test, Y_test)
        cost_tracker["eval_x"].append(i)
        cost_tracker["eval_cost"].append(cost_test)
    # Back propagation
    params = back_propagation_deep(cache, params, activations, Y_train, learning_rate)

plt.plot(cost_tracker["train_x"], cost_tracker["train_cost"], 'b-',
         cost_tracker["eval_x"], cost_tracker["eval_cost"], 'g-')
plt.show()


def error_test_set(test_x, test_y, params):
    A_pred, _ = forward_propagate_deep(params, activations, test_x)
    predictions = np.zeros_like(A_pred)
    predictions[A_pred > 0.5] = 1
    diff_vector = predictions - test_y
    diff_vector = np.square(diff_vector)
    mismatch_vector = diff_vector[diff_vector == 1].reshape((1, -1))
    print("{err}% mismatch error".format(err=diff_vector.shape[1]/mismatch_vector.shape[1]))


error_test_set(X_test, Y_test, params)
user_input = input("Save the input ? [Y/N] : ")
if user_input == 'Y':
    f_name = input("Enter file name : ")
    f_name = "Results/{fn}".format(fn=f_name)
    with open(f_name, 'wb') as file:
        pickle.dump([architecture_nn, params], file)
