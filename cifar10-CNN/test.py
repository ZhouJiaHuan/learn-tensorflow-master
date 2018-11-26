#coding: utf-8
from __future__ import print_function
import tensorflow as tf 
from model import cifar10_5layers, cifar10_8layers
from cifar10 import load_test_data
import os
import argparse

def parse_args():
    	"""
	parse input arguments.
	"""
	parse = argparse.ArgumentParser(description="CIFAR-10 test") 
	parse.add_argument("--model", dest="model_name", 
					   help="model name: 'cifar10-5layers' or 'cifar10-8layers'",
					   default="cifar10-5layers")
	parse.add_argument("--path", dest="model_path",
					   help="trained model file path: ***.data***",
					   default="models/cifar10-5layers_Gaussian-10000.data-00000-of-00001")
	args = parse.parse_args()
	return args


def main():
    args = parse_args()
    model_name = args.model_name
    model_path = args.model_path.split(".")[0]
    # 加载测试数据集
    images_test, _, labels_test = load_test_data()
    print("images_test.shape = ", images_test.shape)

    # 构建模型（模型也可以直接从.meta文件中恢复）
    X = tf.placeholder(tf.float32, [None, 32, 32, 3])
    Y_ = tf.placeholder(tf.float32, [None, 10])
    keep_prob = tf.placeholder(tf.float32)
    if model_name == "cifar10-5layers":
        _, Y = cifar10_5layers(X, keep_prob)
    elif model_name == "cifar10-8layers":
        _, Y = cifar10_8layers(X, keep_prob)
    else:
        print("wrong model name!")
        print("model name: 'cifar10-5layers' or 'cifar10-8layers'")
        raise KeyError

    correct_prediction = tf.equal(tf.argmax(Y, 1), tf.argmax(Y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    
    saver = tf.train.Saver()
    with tf.Session() as sess:
        # 恢复模型权重
        saver.restore(sess, model_path)
        # 测试，注意测试阶段keep_prob=1
        print("test accuracy = ", sess.run(accuracy, feed_dict={X: images_test, Y_: labels_test, keep_prob: 1}))

if __name__ == "__main__":
    main()