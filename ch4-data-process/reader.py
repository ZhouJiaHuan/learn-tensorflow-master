"""reader.py
an example of reading data in TFRecord format
@author: Meringue
@date: 2018/10/23
"""
import tensorflow as tf 
# 创建文件名队列
filename_queue = tf.train.string_input_producer(["stat.tfrecord"])
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

init_op = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init_op)
    # 启动执行入队操作的后台进程
    tf.train.start_queue_runners(sess=sess)
    for i in range(2):
        # 读取数据记录
        example = sess.run(features)
        print(example)
