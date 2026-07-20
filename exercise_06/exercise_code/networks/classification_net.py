import numpy as np
import os
import pickle

from exercise_code.networks.layer import affine_forward, affine_backward, Sigmoid, Tanh, LeakyRelu, Relu
from exercise_code.networks.base_networks import Network
from exercise_code.networks.utils import conv_forward_naive, conv_backward_naive, max_pool_forward_naive, max_pool_backward_naive


class ClassificationNet(Network):
    """
    A fully-connected classification neural network with configurable 
    activation function, number of layers, number of classes, hidden size and
    regularization strength. 
    """

    def __init__(self, activation=Sigmoid, num_layer=2,
                 input_size=3 * 32 * 32, hidden_size=100,
                 std=1e-3, num_classes=10, reg=0, **kwargs):
        """
        :param activation: choice of activation function. It should implement
            a forward() and a backward() method.
        :param num_layer: integer, number of layers. 
        :param input_size: integer, the dimension D of the input data.
        :param hidden_size: integer, the number of neurons H in the hidden layer.
        :param std: float, standard deviation used for weight initialization.
        :param num_classes: integer, number of classes.
        :param reg: float, regularization strength.
        """
        super().__init__("cifar10_classification_net")

        self.activation = activation()
        self.reg_strength = reg

        self.cache = None

        self.memory = 0
        self.memory_forward = 0
        self.memory_backward = 0
        self.num_operation = 0

        # Initialize random gaussian weights for all layers and zero bias
        self.num_layer = num_layer
        self.std = std
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_classes = num_classes
        self.reset_weights()

    def forward(self, X):
        """
        Performs the forward pass of the model.

        :param X: Input data of shape N x D. Each X[i] is a training sample.
        :return: Predicted value for the data in X, shape N x 1
                 1-dimensional array of length N with the classification scores.
        """

        self.cache = {}
        self.reg = {}
        X = X.reshape(X.shape[0], -1)
        # Unpack variables from the params dictionary
        for i in range(self.num_layer - 1):
            W, b = self.params['W' + str(i + 1)], self.params['b' + str(i + 1)]

            # Forward i_th layer
            X, cache_affine = affine_forward(X, W, b)
            self.cache["affine" + str(i + 1)] = cache_affine

            # Activation function
            X, cache_sigmoid = self.activation.forward(X)
            self.cache["sigmoid" + str(i + 1)] = cache_sigmoid

            # Store the reg for the current W
            self.reg['W' + str(i + 1)] = np.sum(W ** 2) * self.reg_strength

        # last layer contains no activation functions
        W, b = self.params['W' + str(self.num_layer)],\
               self.params['b' + str(self.num_layer)]
        y, cache_affine = affine_forward(X, W, b)
        self.cache["affine" + str(self.num_layer)] = cache_affine
        self.reg['W' + str(self.num_layer)] = np.sum(W ** 2) * self.reg_strength

        return y

    def backward(self, dy):
        """
        Performs the backward pass of the model.

        :param dy: N x 1 array. The gradient wrt the output of the network.
        :return: Gradients of the model output wrt the model weights
        """

        # Note that last layer has no activation
        cache_affine = self.cache['affine' + str(self.num_layer)]
        dh, dW, db = affine_backward(dy, cache_affine)
        self.grads['W' + str(self.num_layer)] = \
            dW + 2 * self.reg_strength * self.params['W' + str(self.num_layer)]
        self.grads['b' + str(self.num_layer)] = db

        # The rest sandwich layers
        for i in range(self.num_layer - 2, -1, -1):
            # Unpack cache
            cache_sigmoid = self.cache['sigmoid' + str(i + 1)]
            cache_affine = self.cache['affine' + str(i + 1)]

            # Activation backward
            dh = self.activation.backward(dh, cache_sigmoid)

            # Affine backward
            dh, dW, db = affine_backward(dh, cache_affine)

            # Refresh the gradients
            self.grads['W' + str(i + 1)] = dW + 2 * self.reg_strength * \
                                           self.params['W' + str(i + 1)]
            self.grads['b' + str(i + 1)] = db

        return self.grads

    def save_model(self):
        self.eval()
        directory = 'models'
        model = {self.model_name: self}
        if not os.path.exists(directory):
            os.makedirs(directory)
        pickle.dump(model, open(directory + '/' + self.model_name + '.p', 'wb'))

    def get_dataset_prediction(self, loader):
        self.eval()
        scores = []
        labels = []
        
        for batch in loader:
            X = batch['image']
            y = batch['label']
            score = self.forward(X)
            scores.append(score)
            labels.append(y)
            
        scores = np.concatenate(scores, axis=0)
        labels = np.concatenate(labels, axis=0)

        preds = scores.argmax(axis=1)
        acc = (labels == preds).mean()

        return labels, preds, acc
    
    def eval(self):
        """sets the network in evaluation mode, i.e. only computes forward pass"""
        self.return_grad = False
        
        self.reg = {}
        self.cache = {}
        
    def reset_weights(self):
        self.params = {'W1':self.std * np.random.randn(self.input_size, self.hidden_size),
                       'b1': np.zeros(self.hidden_size)}

        for i in range(self.num_layer - 2):
            self.params['W' + str(i + 2)] = self.std * np.random.randn(self.hidden_size,
                                                                  self.hidden_size)
            self.params['b' + str(i + 2)] = np.zeros(self.hidden_size)

        self.params['W' + str(self.num_layer)] = self.std * np.random.randn(self.hidden_size,
                                                                  self.num_classes)
        self.params['b' + str(self.num_layer)] = np.zeros(self.num_classes)

        self.grads = {}
        self.reg = {}
        for i in range(self.num_layer):
            self.grads['W' + str(i + 1)] = 0.0
            self.grads['b' + str(i + 1)] = 0.0
        


class MyOwnNetwork(ClassificationNet):
    def __init__(self, activation=Sigmoid, num_layer=2,
                 input_size=3 * 32 * 32, hidden_size=100,
                 std=1e-3, num_classes=10, reg=0,
                 dropout=0.5, num_filters=32, filter_size=3,
                 conv_stride=1, conv_pad=1, **kwargs):

        super().__init__(activation=activation, num_layer=num_layer,
                         input_size=input_size, hidden_size=hidden_size,
                         std=std, num_classes=num_classes, reg=reg, **kwargs)

        self.dropout = dropout
        self.num_filters = num_filters
        self.filter_size = filter_size
        self.conv_stride = conv_stride
        self.conv_pad = conv_pad

        self.conv_param = {'stride': self.conv_stride, 'pad': self.conv_pad}
        self.pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}

        self._init_my_params()

    def _init_my_params(self):
        C, H, W = 3, 32, 32
        F = self.num_filters
        HH = self.filter_size
        WW = self.filter_size
        stride = self.conv_stride
        pad = self.conv_pad

        H_conv = 1 + (H + 2 * pad - HH) // stride
        W_conv = 1 + (W + 2 * pad - WW) // stride

        H_pool = 1 + (H_conv - self.pool_param['pool_height']) // self.pool_param['stride']
        W_pool = 1 + (W_conv - self.pool_param['pool_width']) // self.pool_param['stride']

        flat_dim = F * H_pool * W_pool

        self.params = {}
        self.params['Wc1'] = self.std * np.random.randn(F, C, HH, WW)
        self.params['bc1'] = np.zeros(F)

        self.params['W1'] = self.std * np.random.randn(flat_dim, self.hidden_size)
        self.params['b1'] = np.zeros(self.hidden_size)

        for i in range(self.num_layer - 2):
            self.params['W' + str(i + 2)] = self.std * np.random.randn(self.hidden_size, self.hidden_size)
            self.params['b' + str(i + 2)] = np.zeros(self.hidden_size)

        self.params['W' + str(self.num_layer)] = self.std * np.random.randn(self.hidden_size, self.num_classes)
        self.params['b' + str(self.num_layer)] = np.zeros(self.num_classes)

        self.grads = {}
        for k in self.params:
            self.grads[k] = 0.0

        self.conv_out_shape = (F, H_conv, W_conv)
        self.pool_out_shape = (F, H_pool, W_pool)

    def forward(self, X):
        self.cache = {}
        self.reg = {}

        N = X.shape[0]
        X_img = X.reshape(N, 3, 32, 32)

        Wc = self.params['Wc1']
        bc = self.params['bc1']

        out_conv, cache_conv = conv_forward_naive(X_img, Wc, bc, self.conv_param)
        self.cache['conv1'] = cache_conv

        out_pool, cache_pool = max_pool_forward_naive(out_conv, self.pool_param)
        self.cache['pool1'] = cache_pool

        X = out_pool.reshape(N, -1)

        for i in range(self.num_layer - 1):
            Wi = self.params['W' + str(i + 1)]
            bi = self.params['b' + str(i + 1)]

            X, cache_affine = affine_forward(X, Wi, bi)
            self.cache['affine' + str(i + 1)] = cache_affine

            X, cache_act = self.activation.forward(X)
            self.cache['activation' + str(i + 1)] = cache_act

            if self.dropout > 0:
                if getattr(self, 'return_grad', True):
                    mask = (np.random.rand(*X.shape) > self.dropout) / (1.0 - self.dropout)
                    X = X * mask
                    self.cache['dropout' + str(i + 1)] = mask
                else:
                    self.cache['dropout' + str(i + 1)] = None

            self.reg['W' + str(i + 1)] = np.sum(Wi ** 2) * self.reg_strength

        W_last = self.params['W' + str(self.num_layer)]
        b_last = self.params['b' + str(self.num_layer)]
        out, cache_affine_last = affine_forward(X, W_last, b_last)
        self.cache['affine' + str(self.num_layer)] = cache_affine_last
        self.reg['W' + str(self.num_layer)] = np.sum(W_last ** 2) * self.reg_strength
        self.reg['Wc1'] = np.sum(self.params['Wc1'] ** 2) * self.reg_strength

        return out

    def backward(self, dy):
        self.grads = {}

        cache_affine_last = self.cache['affine' + str(self.num_layer)]
        dh, dW, db = affine_backward(dy, cache_affine_last)
        self.grads['W' + str(self.num_layer)] = dW + 2 * self.reg_strength * self.params['W' + str(self.num_layer)]
        self.grads['b' + str(self.num_layer)] = db

        for i in range(self.num_layer - 2, -1, -1):
            if self.dropout > 0:
                mask = self.cache.get('dropout' + str(i + 1))
                if mask is not None:
                    dh = dh * mask

            cache_act = self.cache['activation' + str(i + 1)]
            dh = self.activation.backward(dh, cache_act)

            cache_affine = self.cache['affine' + str(i + 1)]
            dh, dW, db = affine_backward(dh, cache_affine)

            self.grads['W' + str(i + 1)] = dW + 2 * self.reg_strength * self.params['W' + str(i + 1)]
            self.grads['b' + str(i + 1)] = db

        N = self.cache['conv1'][0].shape[0]
        dh_reshaped = dh.reshape((N,) + self.pool_out_shape)
        cache_pool = self.cache['pool1']
        dconv = max_pool_backward_naive(dh_reshaped, cache_pool)

        cache_conv = self.cache['conv1']
        dx, dWc, dbc = conv_backward_naive(dconv, cache_conv)

        self.grads['Wc1'] = dWc + 2 * self.reg_strength * self.params['Wc1']
        self.grads['bc1'] = dbc

        return self.grads
