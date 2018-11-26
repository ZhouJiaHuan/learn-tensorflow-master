"""plot the training log
author: Meringue
date: 9/1/2018
"""
import numpy as np 
import matplotlib.pyplot as plt
import sys

def read_log(log_file):
    """
    read the log file [steps \t accuracy \t loss].

    Args:
        log_file[str]: train log file.
    Return:
        steps [list]: steps.
        accuracy [list]: training or validation accuracy.
        loss [list]: training or validation loss.
    """
    steps = []; accuracy = []; loss = []
    with open(log_file, "r") as f:
        for index, row_data in enumerate(f.readlines()):
            if index == 0: # ignore the head title
                continue
            step, acc, los = row_data.rstrip().split("\t")
            steps.append(int(step))
            accuracy.append(float(acc))
            loss.append(float(los))
    return steps, accuracy, loss

def smooth(data_list, k=3):
    """
    smooth the data_list.

    Args:
        data_list: input list
        k:number for smooth.
    Return:
        data_smooth: data with smooth.
    """
    data_smooth = data_list[:k-1]
    for i in range(len(data_list)-k+1):
        temp = sum(data_list[i:i+k])/k
        data_smooth.append(temp)
    return data_smooth

def plot_log(train_log, valid_log, smooth_train=0):
    """
    plot the training process (train + valid)

    Ars:
        train_log [str]: train log path.
        valid_log [str]: validation log path.
        smooth_train [int]: value for smoothing the training data.
    """
    tr_steps, tr_acc, tr_loss = read_log(train_log)
    va_steps, va_acc, va_loss = read_log(valid_log)
    if smooth_train > 0:
        tr_acc = smooth(tr_acc, smooth_train)
        tr_loss = smooth(tr_loss)
    plt.figure()
    plt.subplot(1,2,1)
    plt.plot(tr_steps, tr_acc, 'r-', label="train accuracy")
    plt.plot(va_steps, va_acc, 'b:', label="valid accuracy")
    plt.ylim((0, 1))
    plt.xlabel("steps"); plt.ylabel("accuracy")
    plt.legend()
    plt.subplot(1,2,2)
    plt.plot(tr_steps, tr_loss, 'r-', label="train loss")
    plt.plot(va_steps, va_loss, 'b:', label="valid loss")
    plt.ylim((0, 3))
    plt.xlabel("steps"); plt.ylabel("loss")
    plt.legend()
    plt.show()
    
if __name__ == "__main__":
    train_log = "./logs/train_log_cifar10-8layers_He.txt"
    valid_log = "./logs/valid_log_cifar10-8layers_He.txt"
    plot_log(train_log, valid_log, smooth_train=3)