import numpy as np


def binarize(X, y, a_percentile, b_percentile):
    """ Splits data to be smaller than the a_percentil and larger than b_percentile
    :param x: input
    :param y: labels
    :param a_percentile:
    :param b_percentile:
    :return:
    :rtype: X, Y
    """
    data_index = ((a_percentile >= y) | (y >= b_percentile))
    y = y[data_index]
    x = X[data_index[:, 0]]

    y[y <= a_percentile] = 0
    y[y >= b_percentile] = 1

    return x, np.expand_dims(y, 1)


def test_accuracy(y_pred, y_true):
    """ Compute test error / accuracy
    Params:
    ------
    y_pred: model prediction
    y_true: ground truth values
    return:
    ------
    Accuracy / error on test set
    """

    # Apply threshold
    threshold = 0.50

    y_binary = np.zeros_like((y_pred))
    y_binary[y_pred >= threshold] = 1
    y_binary[y_pred < threshold] = 0

    # Get final predictions.
    y_binary = y_binary.flatten().astype(int)
    y_true = y_true.flatten().astype(int)

    acc = (y_binary == y_true).mean()
    return acc


import numpy as np


def conv_forward_naive(x, w, b, conv_param):
    """
    x: (N, C, H, W)
    w: (F, C, HH, WW)
    b: (F,)
    conv_param: {'stride': int, 'pad': int}
    """
    stride = conv_param['stride']
    pad = conv_param['pad']

    N, C, H, W = x.shape
    F, _, HH, WW = w.shape

    H_out = 1 + (H + 2 * pad - HH) // stride
    W_out = 1 + (W + 2 * pad - WW) // stride

    x_padded = np.pad(x, ((0, 0), (0, 0),
                          (pad, pad), (pad, pad)), mode='constant')

    out = np.zeros((N, F, H_out, W_out))

    for n in range(N):
        for f in range(F):
            for i in range(H_out):
                h_start = i * stride
                h_end = h_start + HH
                for j in range(W_out):
                    w_start = j * stride
                    w_end = w_start + WW
                    window = x_padded[n, :, h_start:h_end, w_start:w_end]
                    out[n, f, i, j] = np.sum(window * w[f]) + b[f]

    cache = (x, w, b, conv_param, x_padded)
    return out, cache


def conv_backward_naive(dout, cache):
    """
    dout: (N, F, H_out, W_out)
    cache: from conv_forward_naive
    """
    x, w, b, conv_param, x_padded = cache
    stride = conv_param['stride']
    pad = conv_param['pad']

    N, C, H, W = x.shape
    F, _, HH, WW = w.shape
    _, _, H_out, W_out = dout.shape

    dx_padded = np.zeros_like(x_padded)
    dw = np.zeros_like(w)
    db = np.zeros_like(b)

    for n in range(N):
        for f in range(F):
            for i in range(H_out):
                h_start = i * stride
                h_end = h_start + HH
                for j in range(W_out):
                    w_start = j * stride
                    w_end = w_start + WW

                    d_out_val = dout[n, f, i, j]
                    window = x_padded[n, :, h_start:h_end, w_start:w_end]

                    dw[f] += window * d_out_val
                    dx_padded[n, :, h_start:h_end, w_start:w_end] += w[f] * d_out_val

    for f in range(F):
        db[f] = np.sum(dout[:, f, :, :])

    dx = dx_padded[:, :, pad:pad + H, pad:pad + W]
    return dx, dw, db


def max_pool_forward_naive(x, pool_param):
    """
    x: (N, C, H, W)
    pool_param: {'pool_height': int, 'pool_width': int, 'stride': int}
    """
    N, C, H, W = x.shape
    pool_height = pool_param['pool_height']
    pool_width = pool_param['pool_width']
    stride = pool_param['stride']

    H_out = 1 + (H - pool_height) // stride
    W_out = 1 + (W - pool_width) // stride

    out = np.zeros((N, C, H_out, W_out))

    for n in range(N):
        for c in range(C):
            for i in range(H_out):
                h_start = i * stride
                h_end = h_start + pool_height
                for j in range(W_out):
                    w_start = j * stride
                    w_end = w_start + pool_width
                    window = x[n, c, h_start:h_end, w_start:w_end]
                    out[n, c, i, j] = np.max(window)

    cache = (x, pool_param)
    return out, cache


def max_pool_backward_naive(dout, cache):
    """
    dout: (N, C, H_out, W_out)
    cache: from max_pool_forward_naive
    """
    x, pool_param = cache
    N, C, H, W = x.shape
    pool_height = pool_param['pool_height']
    pool_width = pool_param['pool_width']
    stride = pool_param['stride']

    _, _, H_out, W_out = dout.shape

    dx = np.zeros_like(x)

    for n in range(N):
        for c in range(C):
            for i in range(H_out):
                h_start = i * stride
                h_end = h_start + pool_height
                for j in range(W_out):
                    w_start = j * stride
                    w_end = w_start + pool_width

                    window = x[n, c, h_start:h_end, w_start:w_end]
                    m = np.max(window)

                    mask = (window == m)
                    dx[n, c, h_start:h_end, w_start:w_end] += mask * dout[n, c, i, j]

    return dx

