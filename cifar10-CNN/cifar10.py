# coding: utf-8
"""
this code is used to process the CIFAR-10 dataset.
it should be compatible with both Python2 and Python3.
"""

import numpy as np
import pickle
import os
import sys

python_version = sys.version[0]
data_path = os.getcwd()
data_path = os.path.join(data_path,'data/')

def _get_file_path(filename=''):
    """ 
    Return the full path of a data-file for the data-set.
    If filename=="" then return the directory of the files.
    """
    return os.path.join(data_path,'cifar-10-batches-py/',filename)
    
def _unpickle(filename):
    """ 
    Unpickle the given file and return the data.
    Note that the appropriate dir-name is prepended
    the filename.
    """
    file_path = _get_file_path(filename)
    print("Loading data: " + file_path)
    with open(file_path, mode='rb') as file:
        if python_version == "2":
            data = pickle.load(file)
        else:
            data = pickle.load(file, encoding="bytes")
    return data

def _convert_images(raw):
    """
    Convert images from unpickled data (10000, 3072)
    to a 4-dim array
    
    Args:
        raw: unpackled data from cifar10, eg: (10000,3072)
    return:
        a 4-dim array: (img_num, height, width, channel)
    """
    num_channels = 3
    img_size = 32
    raw_float = np.array(raw, dtype=float)/255.0
    images = raw_float.reshape([-1,num_channels,img_size,img_size])
    images = images.transpose([0, 2, 3, 1])
    return images

def _load_data(filename):
    """
    Load a pickled data-file from the CIFAR-10 data set
    and return the converted images (see above) and the 
    class-number for each image.
    """
    data = _unpickle(filename)
    if python_version == "2":
        raw_images = data['data'] # delete 'b' when using python2
        labels = np.array(data['labels']) # delete 'b' when using python2
    else:
        raw_images = data[b'data']
        labels = np.array(data[b'labels'])  
    images = _convert_images(raw_images)
    return images, labels

def load_label_names():
    """
    Load the names for the classes in the CIFAR-10 data set.
    Returns a list with the names. 
    Example: names[3] is the name associated with class-number 3.
    """
    raw = _unpickle("batches.meta")
    if python_version == "2":
        label_names = [x.decode('utf-8') for x in raw['label_names']]
    else:
        label_names = raw[b'label_names']
    return label_names

def _one_hot_encoded(class_numbers, num_classes=None):
    """
    Generate the One-Hot encoded class-labels from an array of integers.

    For example, if class_number=2 and num_classes=4 then
    the one-hot encoded label is the float array: [0. 0. 1. 0.]

    Args:
        class_numbers: array of integers with class-numbers.
        num_classes: number of classes. If None then use max(cls)-1.
    Return:
        2-dim array of shape: [len(cls), num_classes]
    """
    if num_classes is None:
        num_classes = np.max(class_numbers)+1
        
    return np.eye(num_classes, dtype=float)[class_numbers]  

def load_training_data():
    """
    Load all the training-data for the CIFAR-10 data-set.
    The data-set is split into 5 data-files which are merged here.

    Returns:
        images: training images
        labels: label of training images
        one_hot_labels: one-hot labels.
    """ 
    num_files_train = 5
    images_per_file = 10000
    num_classes = 10
    img_size = 32
    num_channels = 3
    num_images_train = num_files_train*images_per_file
    
    # 32bit的Python使用内存超过2G之后,此处会报MemoryError(最好用64位)
    images = np.zeros(shape=[num_images_train, img_size, img_size, num_channels], dtype=float)
    labels = np.zeros(shape=[num_images_train], dtype=int)
    
    begin = 0
    for i in range(num_files_train):
        images_batch, labels_batch = _load_data(filename="data_batch_"+str(i+1)) # _load_data2 in python2
        num_images = len(images_batch)
        end = begin + num_images
        images[begin:end,:] = images_batch
        labels[begin:end] = labels_batch
        begin = end
    one_hot_labels = _one_hot_encoded(class_numbers=labels,num_classes=num_classes)
    return images, labels, one_hot_labels

def load_test_data():
    """
    Load all the test-data for the CIFAR-10 data-set.
    Returns:
    the images, class-numbers and one-hot encoded class-labels.
    """
    num_classes = 10
    images, labels = _load_data(filename="test_batch") # _load_data2 in python2
    return images, labels, _one_hot_encoded(class_numbers=labels, num_classes=num_classes)

def split_train_data(images_train, one_hot_labels_train, ratio = 0.1, shuffle = False):
    """
    split valid data from train data with specified ratio.
    
    Arguments:
        images_train: train data (50000, 32, 32, 3).
        one_hot_labels_train: one-hot labels of train data (50000, 10). 
        ratio: valid data ratio.
        shuffle: shuffle or not.  
    Return:
        images_train: splitted train data.
        one_hot_labels_train: train data labels.
        images_valid: valid data
        one_hot_labels_valid: valid data labels
    """
    num_train = images_train.shape[0]
    num_valid = int(np.math.floor(num_train * ratio))

    if shuffle:
        permutation = list(np.random.permutation(num_train))
        images_train = images_train[permutation, ]
        one_hot_labels_train = one_hot_labels_train[permutation, ]

    images_valid = images_train[-num_valid:, ]
    one_hot_labels_valid = one_hot_labels_train[-num_valid:, ]
    images_train = images_train[0:-num_valid, ]
    one_hot_labels_train = one_hot_labels_train[0:-num_valid, ]
    return images_train, one_hot_labels_train, images_valid, one_hot_labels_valid

def create_mini_batches(X, Y, mini_batch_size = 128, shuffle=False):
    """
    Create a list of minibatches from the training images.
    
    Arguments:
        X: numpy.ndarry images shaped (num_images, height, width, channels).
           for example: (50000, 32, 32, 3).
        Y: one-hot labels of images shaped (num_images, num_classes).
           for example: (50000,10) 
        mini_batch_size: Mini-batch size .
        shuffle: Shuffling the images or not.
    Return:
        mini_batches_X: a list of all mini-batches images, each element in
                        it is an numpy.ndarray containing one batch of images.
        mini_batches_Y: a list of all mini-batches one-hot labels, 
                        each element in it is an one-hot label.
    """
    m = X.shape[0]
    mini_batches_X = []
    mini_batches_Y = []
    
    if shuffle:
        permutation = list(np.random.permutation(m))
        X = X[permutation, ]
        Y = Y[permutation, ]
        
    num_complete_minibathes = int(np.math.floor(m/mini_batch_size))

    for k in range(0, num_complete_minibathes):
        mini_batch_X = X[k*mini_batch_size:(k+1)*mini_batch_size, ]
        mini_batch_Y = Y[k*mini_batch_size:(k+1)*mini_batch_size, ]
        
        mini_batches_X.append(mini_batch_X)
        mini_batches_Y.append(mini_batch_Y)
        
    if m % mini_batch_size != 0:
        mini_batch_X = X[num_complete_minibathes*mini_batch_size:, ]
        mini_batch_Y = Y[num_complete_minibathes*mini_batch_size:, ]
        mini_batches_X.append(mini_batch_X)
        mini_batches_Y.append(mini_batch_Y)

    return mini_batches_X, mini_batches_Y