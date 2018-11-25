"""reader.py
an example of reading data in TFRecord format
catch the OutOfRangeError in reader.py
@author: Meringue
@date: 2018/10/23
"""
import tensorflow as tf 
# 创建文件名队列，并指定遍历4次数据集
filename_queue = tf.train.string_input_producer(["stat.tfrecord"], num_epochs=4)
# 创建读取TFRecords文件的reader
reader = tf.TFRecordReader()
# 读取一条序列号样例
_, serialized_example = reader.read(filename_queue)
# 将一条序列号样例转换为其包含的所有特征张量
features = tf.parse_single_example(
    serialized_example,
    features={
        "id": tf.FixedLenFeature([], tf.int64),
        "age": tf.FixedLenFeature([], tf.int64),
        "income": tf.FixedLenFeature([], tf.float32),
        "outgo": tf.FixedLenFeature([], tf.float32),
    }
)
# 聚合两种初始化操作，使用协调器管理多线程需要先执行
# tf.local_variables_initializer()
init_op = tf.group(tf.global_variables_initializer(),
                   tf.local_variables_initializer())
# 创建协调器
coord = tf.train.Coordinator()

with tf.Session() as sess:
    sess.run(init_op)
    # 启动执行入队操作的后台进程
    threads =  tf.train.start_queue_runners(sess=sess, coord=coord)
    print("Threads: %s" % threads)
    try:
        for i in range(10):
            # 读取数据记录
            if not coord.should_stop():
                example = sess.run(features)
                print(example)
    except tf.errors.OutOfRangeError: # 捕获异常
        print("Catch OutOfRangeError")
    finally:
        coord.request_stop()
        print("Finish reading")
    coord.join(threads)