#coding: utf-8
from __future__ import print_function
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import time
import os
import sys
from cifar10 import load_training_data, load_test_data, load_label_names
from cifar10 import split_train_data, create_mini_batches
import download
from model import cifar10_5layers, cifar10_8layers
import argparse

init_methods = {"Gaussian": tf.truncated_normal_initializer(stddev=1e-2), 
                "Xavier": tf.contrib.layers.xavier_initializer_conv2d(),
                "He": tf.contrib.layers.variance_scaling_initializer()}

def parse_args():
	"""
	parse input arguments.
	"""
	parse = argparse.ArgumentParser(description="CIFAR-10 training") 
	parse.add_argument("--model", dest="model_name", 
					   help="model name: 'cifar10-5layers' or 'cifar10-8layers'",
					   default="cifar10-5layers")
	parse.add_argument("--init", dest="init_method",
					   help="initialization method for weights, 'Gaussian', 'Xavier' or 'He'",
					   default="Gaussian")
	args = parse.parse_args() # 获取所有的参数
	return args

def main():
    """
    # 下载并解压数据集(已下载可忽略)
    data_url = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
    data_path = "./data/"
    download.maybe_download_and_extract(data_url,data_path)
    """
    args = parse_args()
    # 导入数据集并显示数据集信息
    class_names= load_label_names()
    images_train, _, labels_train = load_training_data()
    images_test, _, labels_test = load_test_data()
    images_train, labels_train, images_valid, labels_valid = \
        split_train_data(images_train, labels_train, shuffle=True)

    print("classes names:", class_names)
    print("shape of training images:", images_train.shape)
    print("shape of training labels (one-hot):", labels_train.shape)
    print("shape of valid images:", images_valid.shape)
    print("shape of valid labels (one-hot):", labels_valid.shape)
    print("shape of test images:", images_test.shape)
    print("shape of testing labels (one-hot):", labels_test.shape)

    # 将数据集分成mini-batches.
    images_train_batches, labels_train_batches = create_mini_batches(images_train, \
                                                                    labels_train, \
                                                                    shuffle=True)
    print("shape of one batch training images:", images_train_batches[0].shape)
    print("shape of one batch training labels:", labels_train_batches[0].shape)
    print("shape of last batch training images:", images_train_batches[-1].shape)
    print("shape of last batch training labels:", labels_train_batches[-1].shape)

    img_size = images_train.shape[1]
    num_channels = images_train.shape[-1]
    num_classes = len(class_names)
    batch_size = images_train_batches[0].shape[0]
    num_batches = len(images_train_batches)

    # 创建模型
    X = tf.placeholder(tf.float32, [None, img_size, img_size, num_channels])
    Y_ = tf.placeholder(tf.float32, [None, num_classes])
    keep_prob = tf.placeholder(tf.float32)

    init_method = init_methods[args.init_method]
    if args.model_name == "cifar10-5layers":
        logit, Y = cifar10_5layers(X, keep_prob, init_method)
    elif args.model_name == "cifar10-8layers":
        logit, Y = cifar10_8layers(X, keep_prob, init_method)

    # 交叉熵损失和准确率
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logit, labels=Y_)
    cross_entropy = tf.reduce_mean(cross_entropy)
    correct_prediction = tf.equal(tf.argmax(Y, 1), tf.argmax(Y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    # 训练过程
    tr_step = []; tr_acc = []; tr_loss = []
    va_step = []; va_acc = []; va_loss = []
    train_steps =50000 
    lr = 0.0001
    train_step = tf.train.AdamOptimizer(lr).minimize(cross_entropy)
    init = tf.global_variables_initializer()
    saver = tf.train.Saver()
    with tf.Session() as sess:
        sess.run(init)
        for i in range(train_steps+1):
            # 获取一个mini batch数据
            j = i%num_batches # j用于记录第几个mini batch
            batch_X = images_train_batches[j]
            batch_Y = labels_train_batches[j]
            if j == num_batches-1: # 遍历一遍训练集（1个epoch）
                images_train_batches, labels_train_batches = create_mini_batches(images_train, \
                                                                                labels_train, \
                                                                                shuffle=True)

            # 训练一次
            sess.run(fetches=train_step, feed_dict={X: batch_X, Y_: batch_Y, keep_prob: 0.5})
            
            # 每100次打印并记录一组训练结果
            if i % 100 == 0:
                train_accuracy, train_loss = sess.run([accuracy, cross_entropy], \
                                                      feed_dict={X: batch_X, Y_: batch_Y, keep_prob: 1})
                print("steps =", i, "train loss =", train_loss, " train accuracy =", train_accuracy)
                tr_step.append(i)
                tr_acc.append(train_accuracy)
                tr_loss.append(train_loss)

            # 每500次打印并记录一次测试结果（验证集）
            if i % 500 == 0:
                valid_accuracy, valid_loss = sess.run([accuracy, cross_entropy], \
                                                      feed_dict={X: images_valid, Y_: labels_valid, keep_prob: 1})
                va_step.append(i)
                va_acc.append(valid_accuracy)
                va_loss.append(valid_loss)
                print("steps =", i, "validation loss =", valid_loss, " validation accuracy =", valid_accuracy)
            
            # 每10000次保存一次训练模型到本地
            if i % 10000 == 0 and i > 0:
                model_name = args.model_name + "_" + args.init_method
                model_name = os.path.join("./models", model_name)
                saver.save(sess, model_name, global_step=i)

    # 保存训练日志到本地  
    train_log = "train_log_" + args.model_name + "_" + args.init_method + ".txt" 
    train_log = os.path.join("./results", train_log)
    with open(train_log, "w") as f:
        f.write("steps\t" + "accuracy\t" + "loss\n")
        for i in range(len(tr_step)):
            row_data = str(tr_step[i]) + "\t" + str(round(tr_acc[i],3)) + "\t" + str(round(tr_loss[i],3)) + "\n"
            f.write(row_data)

    valid_log = "valid_log_" + args.model_name + "_" + args.init_method + ".txt" 
    valid_log = os.path.join("./results", valid_log)            
    with open(valid_log, "w") as f:
        f.write("steps\t" + "accuracy\t" + "loss\n")
        for i in range(len(va_step)):
            row_data = str(va_step[i]) + "\t" + str(round(va_acc[i],3)) + "\t" + str(round(va_loss[i],3)) + "\n"
            f.write(row_data)

if __name__ == "__main__":
    main()
