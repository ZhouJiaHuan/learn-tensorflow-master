import tensorflow as tf 
from model import cifar10_5layers, cifar10_8layers
from cifar10 import load_test_data
import os

model_path ="./models/cifar10-5layers_Gaussian-50000"

if __name__ == "__main__":
    images_test, _, labels_test = load_test_data()
    X = tf.placeholder(tf.float32, [None, 32, 32, 3])
    Y_ = tf.placeholder(tf.float32, [None, 10])
    keep_prob = tf.placeholder(tf.float32)
    logit, Y = cifar10_5layers(X, keep_prob)
    saver = tf.train.Saver()
    with tf.Session() as sess:
        saver.restore(sess, model_path)
        sess.run(Y, feed_dict={X: images_test, Y_: labels_test, keep_prob: 1})
        correct_prediction = tf.equal(tf.argmax(Y, 1), tf.argmax(Y_, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        print("accuracy = ", accuracy)

