""" model.py
model samples for CIFAR-10
@author: Meringue
@date: 2018/11/05
"""

import tensorflow as tf

def max_pool(feature_map):
    return tf.nn.max_pool(feature_map, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

def conv_relu(feature_map, weight, bias):
    conv = tf.nn.conv2d(feature_map, weight, strides=[1, 1, 1, 1], padding='SAME')
    return tf.nn.relu(conv + bias)

def conv_pool_relu(feature_map, weight, bias):
    conv = tf.nn.conv2d(feature_map, weight, strides=[1, 1, 1, 1], padding='SAME')
    pool = tf.nn.max_pool(conv, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    return tf.nn.relu(pool + bias)

def fc_relu(fc_input, weight, bias):
    fc = tf.matmul(fc_input, weight) 
    return tf.nn.relu(fc + bias)

def cifar10_5layers(input_image, keep_prob, init_method=tf.truncated_normal_initializer(stddev=1e-2)):
    """ 
    model definition with 5 layers.

    Args:
        input_image: input image tensor.
        init_method: initialization method. 
                     The default is tf.truncated_normal_initializer(1e-2)
    Return:
        model: computation graph of the defined model.
    """
    with tf.variable_scope("conv1"):
        W1 = tf.get_variable(name="W1", shape=[5,5,3,32], dtype=tf.float32, \
                             initializer=init_method)
        b1 = tf.get_variable(name="b1", shape=[32], dtype=tf.float32, \
                             initializer=tf.constant_initializer(0.01))
        conv1 = conv_pool_relu(input_image, W1, b1)
    with tf.variable_scope("conv2"):
        W2 = tf.get_variable(name="W2", shape=[5,5,32,64], dtype=tf.float32, \
                             initializer=init_method)
        b2 = tf.get_variable(name="b2", shape=[64], dtype=tf.float32, \
                             initializer=tf.constant_initializer(0.01))
        conv2 = conv_pool_relu(conv1, W2, b2)
	conv2 = tf.nn.dropout(conv2, keep_prob)
    with tf.variable_scope("conv3"):
        W3 = tf.get_variable(name="W3", shape=[5,5,64,128], dtype=tf.float32, \
                             initializer=init_method)
        b3 = tf.get_variable(name="b3", shape=[128], dtype=tf.float32, \
                             initializer=tf.constant_initializer(0.01))
        conv3 = conv_pool_relu(conv2, W3, b3)
	conv3 = tf.nn.dropout(conv3, keep_prob)
    with tf.variable_scope("fc1"):
        W4 = tf.get_variable(name="W4", shape=[4*4*128,256], dtype=tf.float32, \
                             initializer=init_method)
        b4 = tf.get_variable(name="b4", shape=[256], dtype=tf.float32, \
                             initializer=tf.constant_initializer(0.01))
        conv3_flat = tf.reshape(conv3, [-1, 4*4*128])
        fc1 = fc_relu(conv3_flat, W4, b4)
	fc1 = tf.nn.dropout(fc1, keep_prob)
    with tf.variable_scope("output"):
        W5 = tf.get_variable(name="W5", shape=[256,10], dtype=tf.float32, \
                             initializer=init_method)
        b5 = tf.get_variable(name="b5", shape=[10], dtype=tf.float32, \
                             initializer=tf.constant_initializer(0.01))
	y_logit = tf.matmul(fc1, W5) + b5
    return y_logit, tf.nn.softmax(y_logit, name="softmax")

def cifar10_8layers(input_image, keep_prob, init_method=tf.truncated_normal_initializer()):
    """ 
    model definition with 8 layers.

    Args:
        input_image: input image tensor.
        init_method: initialization method. 
                     The default is tf.truncated_normal_initializer()
        keep_prop: keep propobality in dropout.
    Return:
        model: computation graph of the defined model.
    """
    with tf.variable_scope("conv1_1"):
        W1_1 = tf.get_variable(name="W1_1", shape=[3,3,3,32], dtype=tf.float32, \
                             initializer=init_method)
        b1_1 = tf.get_variable(name="b1_1", shape=[32], dtype=tf.float32, \
                             initializer=tf.constant_initializer(0.01))
        conv1_1 = conv_relu(input_image, W1_1, b1_1)
    with tf.variable_scope("conv1_2"):
        W1_2 = tf.get_variable(name="W1_2", shape=[3,3,32,32], dtype=tf.float32, \
                             initializer=init_method)
        b1_2 = tf.get_variable(name="b1_2", shape=[32], dtype=tf.float32, \
                             initializer=tf.constant_initializer(0.01))
        conv1_2 = max_pool(conv_relu(conv1_1, W1_2, b1_2))
	#conv1_2 = tf.nn.dropout(conv1_2, keep_prob)
    with tf.variable_scope("conv2_1"):
        W2_1 = tf.get_variable(name="W2_1", shape=[3,3,32,64], dtype=tf.float32, \
                             initializer=init_method)
        b2_1 = tf.get_variable(name="b2_1", shape=[64], dtype=tf.float32, \
                             initializer=tf.constant_initializer(0.01))
        conv2_1 = conv_relu(conv1_2, W2_1, b2_1)
        #conv2_1 = tf.nn.dropout(conv2_1, keep_prob)
    with tf.variable_scope("conv2_2"):
        W2_2 = tf.get_variable(name="W2_2", shape=[3,3,64,64], dtype=tf.float32, \
                             initializer=init_method)
        b2_2 = tf.get_variable(name="b2_2", shape=[64], dtype=tf.float32, \
                             initializer=tf.constant_initializer(0.01))
        conv2_2 = max_pool(conv_relu(conv2_1, W2_2, b2_2))
        #conv2_2 = tf.nn.dropout(conv2_2, keep_prob)
    with tf.variable_scope("conv3_1"):
        W3_1 = tf.get_variable(name="W3_1", shape=[3,3,64,128], dtype=tf.float32, \
                             initializer=init_method)
        b3_1 = tf.get_variable(name="b3_1", shape=[128], dtype=tf.float32, \
                             initializer=tf.constant_initializer(0.01))
        conv3_1 = conv_relu(conv2_2, W3_1, b3_1)
        #conv3_1 = tf.nn.dropout(conv3_1, keep_prob)
    with tf.variable_scope("conv3_2"):
        W3_2 = tf.get_variable(name="W3_2", shape=[3,3,128,128], dtype=tf.float32, \
                             initializer=init_method)
        b3_2 = tf.get_variable(name="b3_2", shape=[128], dtype=tf.float32, \
                             initializer=tf.constant_initializer(0.01))
        conv3_2 = max_pool(conv_relu(conv3_1, W3_2, b3_2))
        conv3_2 = tf.nn.dropout(conv3_2, keep_prob)
    with tf.variable_scope("fc1"):
        W4 = tf.get_variable(name="W4", shape=[4*4*128,256], dtype=tf.float32, \
                             initializer=init_method)
        b4 = tf.get_variable(name="b4", shape=[256], dtype=tf.float32, \
                             initializer=tf.constant_initializer(0.01))
        conv3_flat = tf.reshape(conv3_2, [-1, 4*4*128])
        fc1 = fc_relu(conv3_flat, W4, b4)
        fc1 = tf.nn.dropout(fc1, keep_prob)
    with tf.variable_scope("fc2"):
        W5 = tf.get_variable(name="W5", shape=[256,512], dtype=tf.float32, \
                             initializer=init_method)
        b5 = tf.get_variable(name="b5", shape=[512], dtype=tf.float32, \
                             initializer=tf.constant_initializer(0.01))
        fc2 = fc_relu(fc1, W5, b5)
        fc2 = tf.nn.dropout(fc2, keep_prob)
    with tf.variable_scope("output"):
        W6 = tf.get_variable(name="W6", shape=[512,10], dtype=tf.float32, \
                             initializer=init_method)
        b6 = tf.get_variable(name="b6", shape=[10], dtype=tf.float32, \
                             initializer=tf.constant_initializer(0.01))
        y_logit = tf.matmul(fc2, W6) + b6
    return y_logit, tf.nn.softmax(y_logit, name="softmax")
