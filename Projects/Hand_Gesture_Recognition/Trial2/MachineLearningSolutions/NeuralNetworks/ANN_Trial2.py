"""
Implemented mini batch gradient descent
"""
import numpy as np
import cv2 as cv
import pickle
import os

from jedi.api import exceptions
from matplotlib import pyplot as plt

np.random.seed(2)


def load_dataset(dataset_dir_name="../Data", x_name="X.npy",
                 y_name="Y.npy", one_hot_index=-1, shuffle_data=True, normalize_data=True):
    """
    Loads a dataset stored as a .npy file
    :param dataset_dir_name: Name of the folder in which data is
    :param x_name: Name of file which contains inputs (with .npy extension)
    :param y_name: Name of file which contains outputs (with .npy extension)
    :param one_hot_index: The index you're training for (pass -1 for passing raw output file), should be >= -1
    :param shuffle_data: Shuffle the data
    :param normalize_data: Normalize the input data
    :return:
        X, Y
        X -> Inputs
        Y -> Outputs
        The output will be shuffled if shuffle_data is True, else it won't be shuffled
    """
    # Load the dataset
    print("DATA DEBUG : Checking directory \"{r}/{x}\" for inputs and \"{r}/{y}\" for outputs".format(
        r=dataset_dir_name, x=x_name, y=y_name
    ))
    X = np.load("{main_root}/{f_name}".format(main_root=dataset_dir_name,
                                              f_name=x_name))
    print("DATA DEBUG : Inputs shape is {in_shape}".format(in_shape=X.shape))
    if one_hot_index == -1:
        # Parse the entire dataset as it is
        Y = np.load("{main_root}/{f_name}".format(main_root=dataset_dir_name,
                                                  f_name=y_name))
    elif one_hot_index >= 0:
        Y_one_hot = np.load("{main_root}/{f_name}".format(
            main_root=dataset_dir_name, f_name=y_name
        ))
        Y = np.array(Y_one_hot[one_hot_index, :].reshape((1, -1)))
    else:
        raise IndexError("The index {ind_number} is an illegal index".format(ind_number=one_hot_index))
    print("DATA DEBUG : Output shape is {out_shape}".format(out_shape=Y.shape))
    if normalize_data:
        X = X / 255
    if shuffle_data:
        X, Y = shuffle_dataset(X, Y)
    Y_true_p = Y.nonzero()[1].reshape(1, -1).shape[1] / Y.shape[1]
    print("DATA DEBUG : {tp}% data is true".format(tp=Y_true_p * 100))
    return X, Y


def shuffle_dataset(X, Y):
    buffer_data = np.row_stack((X, Y))
    buffer_data = buffer_data.T
    np.random.shuffle(buffer_data)
    buffer_data = buffer_data.T
    X = buffer_data[0:X.shape[0], :]
    Y = buffer_data[X.shape[0]:X.shape[0] + Y.shape[0], :]
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
            The size architecture of the neural network. Sizes of every layer including input layer
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
        params["W" + str(i)] = np.random.randn(layer_size[i], layer_size[i - 1]) * 2 / np.sqrt(layer_size[i - 1])
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


def tanh():
    """
    Tan Hyperbolic activation function
    :return:
        Dictionary
            "function" : Tan hyperbolic function
            "derivative" : Derivative of tan hyperbolic function
    """

    def tanh_function(x):
        return np.tanh(x)

    def tanh_derivative(x):
        return np.square(1 / (np.cosh(x)))

    func_dict = {
        "function": tanh_function,
        "derivative": tanh_derivative
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
        cache["Z" + str(i)] = np.matmul(params["W" + str(i)], cache["A" + str(i - 1)]) + params["b" + str(i)]
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
    cost_val = np.average(diff_vect) / 2
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
        activation_derivative = activations[L - 1]()["derivative"]
        del_vals["del" + str(L)] = ((cache["A" + str(L)] - Y) / (m * cache["A" + str(L)] * (1 - cache["A" + str(L)]))) * \
                                   activation_derivative(cache["Z" + str(L)])
    else:
        del_vals["del" + str(L)] = ((cache["A" + str(L)] - Y) / (m))
    for l in range(L - 1, 0, -1):  # Go backward from layer L-1 to 1 to calculate del_vals["del" + l]
        activation_derivative = activations[l - 1]()["derivative"]
        del_vals["del" + str(l)] = np.matmul(params["W" + str(l + 1)].T, del_vals["del" + str(l + 1)]) * \
                                   activation_derivative(cache["Z" + str(l)])
    # Calculate final derivatives
    grads = {}
    # Final backward step
    for l in range(L, 0, -1):
        grads["dW" + str(l)] = np.matmul(del_vals["del" + str(l)], cache["A" + str(l - 1)].T)
        grads["db" + str(l)] = np.matmul(del_vals["del" + str(l)], np.ones((m, 1)))
    return grads


# Back propagation step
def back_propagation_deep(cache, params, activations, Y, learning_rate=0.01, reg_lambda=0.2):
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
    :param reg_lambda: Regularization parameter
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
        params["W" + str(l)] = params["W" + str(l)] * (1 - learning_rate * reg_lambda / m) \
                               - learning_rate * grads["dW" + str(l)]
        params["b" + str(l)] = params["b" + str(l)] * (1 - learning_rate * reg_lambda / m) \
                               - learning_rate * grads["db" + str(l)]
    return params


def error_test_set(test_x, test_y, params, activations, threshold=0.5,
                   show_mismatch_images=True):
    """
    To test the performance of the neural network
    :param test_x: Inputs
    :param test_y: Desired outputs
    :param params: Parameters of the neural network
    :param activations: Activation functions
    :param threshold: Threshold value wanted
    :param show_mismatch_images: If you want to view mismatch images
    :return:
    None
    """
    predictions, _ = forward_propagate_deep(params, activations, test_x)
    pred_test = np.zeros_like(predictions)
    pred_test[predictions > threshold] = 1
    difference_vector = pred_test - test_y
    difference_vector = np.square(difference_vector)
    diff_indices = difference_vector.nonzero()[1]
    print("TEST DEBUG : {num_mismatch} mismatches found at indices {ind}".format(
        num_mismatch=len(diff_indices), ind=diff_indices
    ))
    if show_mismatch_images:
        for ind in diff_indices:
            print("TEST DEBUG : Testing image at index {i}".format(i=ind))
            img = test_x[:, ind].reshape((47, 38))
            cv.imshow("Index {i}, Y = {y}, A = {a}".format(
                i=ind, y=test_y[:, ind], a=predictions[:, ind]
            ), img)
            while True:
                key = cv.waitKey(0) & 0xff
                if key == 27:
                    exit(0)
                elif key == ord('n'):
                    cv.destroyWindow("Index {i}, Y = {y}, A = {a}".format(
                        i=ind, y=test_y[:, ind], a=predictions[:, ind]))
                    break
    cv.destroyAllWindows()


def divide_into_mini_batches(X, Y, mini_batch_size, debugger_output=False):
    """
    Divide the passed set into buckets (mini batches) of size mini_batch_size
    :param X: Input examples
    :param Y: Output of the examples
    :param mini_batch_size: The mini batch size
    :param debugger_output: Show debugger output
    :return:
        tuple(X_mini_batches, Y_mini_batches)
            X_mini_batches : Array of input batches
            Y_mini_batches : Array of output batches
    """
    m = Y.shape[1]
    n_batches = m // mini_batch_size
    if debugger_output:
        print("DATA DEBUG : Dividing {num} examples into batches of size {batch_s}. {n_b} full batches".format(
            num=m, batch_s=mini_batch_size, n_b=n_batches
        ))
    X_mini_batches = []
    Y_mini_batches = []
    for i in range(n_batches):
        X_mini_batches.append(X[:, i * mini_batch_size: (i + 1) * mini_batch_size])
        Y_mini_batches.append(Y[:, i * mini_batch_size: (i + 1) * mini_batch_size])
    if m % mini_batch_size != 0:
        X_mini_batches.append(X[:, n_batches * mini_batch_size:])
        Y_mini_batches.append(Y[:, n_batches * mini_batch_size:])
    return (X_mini_batches, Y_mini_batches)


# Network configurations dictionary
net_config = {
    "data": {
        "dir_name": "../Data/Global",
        "x_name": "X.npy",
        "y_name": "Y_one_hot_encoded.npy",
        "one_hot_index": 1,
        "dist_train_dev_test": (900, 0, 21)
    },
    "nn_arc": {
        "layers": (100, 50, 5, 1),
        "activations": [tanh, relu, tanh, sigmoid]
    },
    "hyperparameters": {
        "training_iterations": 20,
        "debug_num_iter": 100,
        "learning_rate": 0.01,
        "regularization_parameter": 1.2,
        "num_epochs": 100,
        "mini_batch_size": 90,
        "threshold": 0.9
    },
    "testing": {
        "test_image": "../Data/4_Fingers/M_23.jpg",
        "false_examples": "../../DataCollector/Mask_data_collector/Data_Distribution_Generated/False",
        "true_examples": "../../DataCollector/Mask_data_collector/Data_Distribution_Generated/True",
        "show_mismatches": True,
        "live_trial_mode": True
    }
}

if __name__ == '__main__':
    # Load data into memory
    X, Y = load_dataset(dataset_dir_name=net_config["data"]["dir_name"], x_name=net_config["data"]["x_name"],
                        y_name=net_config["data"]["y_name"], one_hot_index=net_config["data"]["one_hot_index"])
    datasets = split_train_dev_test(X, Y, net_config["data"]["dist_train_dev_test"])
    X_train, Y_train = datasets["train"]
    X_dev, Y_dev = datasets["dev"]
    X_test, Y_test = datasets["test"]
    # Load weights and activation functions
    nn_architecture = {
        "layers": net_config["nn_arc"]["layers"],
        "activations": net_config["nn_arc"]["activations"]
    }
    architecture_nn, params = init_params_deep(X.shape[0], nn_architecture["layers"])
    activations = nn_architecture["activations"]
    # Training the network
    # Hyperparameters
    num_iter = net_config["hyperparameters"]["training_iterations"]
    debug_iter_num = net_config["hyperparameters"]["debug_num_iter"]
    learning_rate = net_config["hyperparameters"]["learning_rate"]
    reg_param_lambda = net_config["hyperparameters"]["regularization_parameter"]
    num_epochs = net_config["hyperparameters"]["num_epochs"]
    mini_batch_size = net_config["hyperparameters"]["mini_batch_size"]
    cost_tracker = {
        "train_x": [],
        "train_cost": [],
        "eval_x": [],
        "eval_cost": []
    }
    print("TRAIN DEBUG : {it} training iterations required".format(
        it=int(num_iter * num_epochs * X_train.shape[1] / mini_batch_size)
    ))
    # Main training process
    for train_iter_num in range(num_iter):
        X_training, Y_training = shuffle_dataset(X_train, Y_train)
        (X_training_batches, Y_training_batches) = divide_into_mini_batches(X_training, Y_training, mini_batch_size)
        num_batches = len(Y_training_batches)
        for batch_index in range(len(Y_training_batches)):
            X_training_batch = X_training_batches[batch_index]
            Y_training_batch = Y_training_batches[batch_index]
            for epoch_count in range(num_epochs):
                # Forward Propagate
                Y_pred_mini_batch, cache_mini_batch = forward_propagate_deep(params, activations, X_training_batch)
                # Note the cost
                cost_iter = cost_function(Y_pred_mini_batch, Y_training_batch)
                i = train_iter_num * num_batches * num_epochs + batch_index * num_epochs + epoch_count
                cost_tracker["train_x"].append(i)
                cost_tracker["train_cost"].append(cost_iter)
                if i % debug_iter_num == 0 or i == 0:
                    pred_test, _ = forward_propagate_deep(params, activations, X_test)
                    eval_cost = cost_function(pred_test, Y_test)
                    cost_tracker["eval_x"].append(i)
                    cost_tracker["eval_cost"].append(eval_cost)
                    print("TRAIN DEBUG : Cost at iteration {it_num} is {cost}\t Test cost is {test_cost}".format(
                        cost=cost_iter, it_num=i, test_cost=eval_cost))
                # Back propagation
                params = back_propagation_deep(cache_mini_batch, params, activations, Y_training_batch,
                                               learning_rate, reg_param_lambda)

    # print(params)
    # print(forward_propagate_deep(params, activations, X))

    plt.plot(cost_tracker["train_x"], cost_tracker["train_cost"], 'b-',
             cost_tracker["eval_x"], cost_tracker["eval_cost"], 'g-')
    plt.legend(["Training", "Evaluation"])
    plt.show()

    error_test_set(X_test, Y_test, params, activations,
                   show_mismatch_images=net_config["testing"]["show_mismatches"])

    # Test segment
    try:
        img = cv.imread(net_config["testing"]["test_image"])
        img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        cv.imshow("Test image", img)
        img = cv.resize(img, None, fx=1 / 5, fy=1 / 5)
        X_img = np.ndarray.reshape(img, (-1, 1))
        pred_test, _ = forward_propagate_deep(params, activations, X_img)
        print("Prediction on test image is ", pred_test)
        cv.destroyAllWindows()
    except Exception:
        cv.destroyAllWindows()
    # Test over all the examples in the directories
    try:
        # Parse everything in the file
        c = 0
        t = 0
        f_dir = net_config["testing"]["false_examples"]
        tp = 0  # True positive
        fp = 0  # False positive
        fn = 0  # False negative
        for filename in os.listdir("{r}".format(r=f_dir)):
            img = cv.imread("{r}/{f}".format(r=f_dir, f=filename))
            img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
            img = cv.resize(img, None, fx=1 / 5, fy=1 / 5)
            X_img = np.reshape(img, (-1, 1))
            pred_test, _ = forward_propagate_deep(params, activations, X_img)
            if pred_test > net_config["hyperparameters"]["threshold"]:
                c += 1
            t += 1
        print("TEST DEBUG : {count} files (out of {tot}) predicted true in 'False'".format(count=c, tot=t))
        fp = c
        c = 0
        t = 0
        f_dir = net_config["testing"]["true_examples"]
        for filename in os.listdir("{r}".format(r=f_dir)):
            img = cv.imread("{r}/{f}".format(r=f_dir, f=filename))
            img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
            img_r = cv.resize(img, None, fx=1 / 5, fy=1 / 5)
            X_img = np.reshape(img_r, (-1, 1))
            pred_test, _ = forward_propagate_deep(params, activations, X_img)
            if pred_test > net_config["hyperparameters"]["threshold"]:
                c += 1
            t += 1
        print("TEST DEBUG : {count} files (out of {tot}) predicted true in 'True'".format(count=c, tot=t))
        tp = c
        fn = t - c
        prec_score = tp / (tp + fp)
        rec_score = tp / (tp + fn)
        print("TEST DEBUG : Precision is {prec} and Recall is {rec}. F1 score is {f_1}".format(
            prec=prec_score, rec=rec_score, f_1=2 * (prec_score * rec_score) / (prec_score + rec_score)
        ))
    except FileNotFoundError:
        print("File not found")
    if net_config["testing"]["live_trial_mode"]:    # Try out your own images
        input("Put all test images in folder \"Trial_Images\" and press enter")
        # Check out the trial folder and report result for every image
        for filename in os.listdir("Trial_Images"):
            full_filename = "{r}/{f}".format(r="Trial_Images", f=filename)
            img = cv.imread(full_filename)
            img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
            img_s = cv.resize(img, None, fx=1/5, fy=1/5)
            _, img_s = cv.threshold(img_s, 127, 255, cv.THRESH_BINARY)
            img_x = img_s.reshape((-1, 1))
            pred, _ = forward_propagate_deep(params, activations, img_x)
            cv.imshow("{f} A = {a}".format(f=filename, a=pred), img)
            print("TEST DEBUG : File \"{f}\"".format(f=full_filename))
            cv.waitKey(0)
            cv.destroyAllWindows()
    user_input = input("Save the network architecture and parameters ? [Y/N] : ")
    if user_input == 'Y' or user_input == 'y':
        f_name = input("Enter file name : ")
        f_name = "Results/{fn}".format(fn=f_name)
        with open(f_name, 'wb') as file:
            pickle.dump([architecture_nn, params], file)
