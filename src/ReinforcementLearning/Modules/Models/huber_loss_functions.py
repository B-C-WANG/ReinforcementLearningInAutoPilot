# coding: utf-8
# Type: Private Author: Baochuan Wang
import numpy as np
import tensorflow as tf


def huber_loss_tf_graph(y, y_hat, delta):
    half_delta_squared = 0.5 * delta * delta
    error = y - y_hat
    abs_error = tf.abs(error)

    less_than = 0.5 * tf.square(error)
    more_than = (delta * abs_error) - half_delta_squared

    diff = abs_error - less_than
    # TODO:将判断语句改成计算图

    if abs_error < delta:
        loss = less_than
    else:
        loss = more_than

    return tf.reduce_sum(loss)


def huber_loss(y, y_hat, delta):
    """
    Huber loss 优于均方根误差偏差小于阈值,使用平方误差,大于阈值采用线性误差
    Compute the Huber Loss as part of the model graph
    Huber Loss is more robust to outliers. It is defined as:
     if |y - y_hat| < delta :
        0.5 * (y - y_hat)**2
    else :
        delta * |y - y_hat| - 0.5 * delta**2
    Attributes:
        y (Tensor[-1, 1]): Target value
        y_hat(Tensor[-1, 1]): Estimated value
        delta (float): Outliers threshold
    Returns:
        CNTK Graph Node
    """
    half_delta_squared = 0.5 * delta * delta
    error = y - y_hat
    abs_error = abs(error)

    less_than = 0.5 * np.square(error)
    more_than = (delta * abs_error) - half_delta_squared
    if abs_error < delta:
        loss = less_than
    else:
        loss = more_than

    return np.sum(loss)
