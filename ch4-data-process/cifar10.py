"""
an example of processing CIFAR-10 dataset.
@author: Meringue
@data: 2018/10/31
"""

import tensorflow as tf 
import matplotlib.pyplot as plt 
LABEL_BYTES = 1
IMAGE_SIZE = 32
IMAGE_DEPTH = 3
IMAGE_BYTES = IMAGE_SIZE *IMAGE_SIZE * IMAGE_DEPTH
NUM_CLASSES = 10

def read_cifar10(data_file, batch_size):
    """
    从CIFAR-10数据文件读取批样例

    Args:
        data_file: CIFAR-10数据集文件
        batch_size: 批数据大小
    Return:
        images: 形如[batch_size, image_size, image_size, 3]的批数据
        labels: 形如[batch_size, NUM_CLASSES]的标签批数据
    """
    record_bytes = LABEL_BYTES + IMAGE_BYTES
    # 创建文件名列表
    data_files = tf.gfile.Glob(data_file) # 查找匹配的文件并以列表形式返回
    # print("data files = ", data_files)
    # 创建文件名队列
    file_queue = tf.train.string_input_producer(data_files, shuffle=True)
    # 创建reader
    reader = tf.FixedLengthRecordReader(record_bytes=record_bytes)
    _, value = reader.read(file_queue)
    # 解析出类别标签和图片信息
    record = tf.decode_raw(value, tf.uint8) # 将字符串数据重新变回原数据格式
    record = tf.reshape(record, [record_bytes])
    label = tf.cast(tf.slice(record, [0], [LABEL_BYTES]), tf.int32)
    depth_major = tf.slice(record, [LABEL_BYTES], [IMAGE_BYTES])
    depth_major = tf.reshape(depth_major, [IMAGE_DEPTH, IMAGE_SIZE, IMAGE_SIZE])
    image = tf.cast(tf.transpose(depth_major, [1,2,0]), tf.float32)
    # 创建样例队列
    example_queue = tf.RandomShuffleQueue(capacity=16*batch_size, 
                                          min_after_dequeue=8*batch_size,
                                          dtypes=[tf.float32, tf.int32],
                                          shapes=[[IMAGE_SIZE,IMAGE_SIZE,IMAGE_DEPTH], [1]])
    num_threads = 16
    # 创建样例队列的入队操作
    example_enqueue_op = example_queue.enqueue([image, label])
    # 将定义的16个线程全部添加到queue runnner中
    tf.train.add_queue_runner(tf.train.queue_runner.QueueRunner(
        example_queue, [example_enqueue_op]*num_threads))
    images, labels = example_queue.dequeue_many(batch_size)
    labels = tf.reshape(labels, [batch_size, 1])
    zeros = tf.constant(0, shape=[batch_size,1])
    indices = tf.reshape(tf.range(0, batch_size, 1), [batch_size, 1])
    labels = tf.sparse_to_dense(tf.concat(values=[zeros, labels], axis=1), 
                                [batch_size, NUM_CLASSES], 1.0, 0.0)
    assert len(images.get_shape()) == 4
    assert images.get_shape()[0] == batch_size
    assert images.get_shape()[-1] == 3
    assert len(labels.get_shape()) == 2
    assert labels.get_shape()[0] == batch_size
    assert labels.get_shape()[1] == NUM_CLASSES

    return images, labels

if __name__ == "__main__":
    cifar10_dir = "E:\\Database\\cifar-10-batches-py\\"
    # print(tf.gfile.Glob(cifar10_dir+"data_batch_*"))
    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        images, labels = read_cifar10(cifar10_dir+"data_batch_*", batch_size=1)
        print(images.get_shape())
        print(labels.get_shape())
        # print(sess.run(images))
        print(sess.run(labels))
        # images_np = images.eval()
        # print(images_np.shape)
    #plt.figure()
    #plt.imshow(images_np)
    #plt.show()



    
    
    