"""ex1.py
an example of reading CSV file.
@author: Meringue
@date: 2018/09/07
"""
import tensorflow as tf 

# 创建文件名队列
filename_list = ["stat0.csv", "stat1.csv"]
filename_queue = tf.train.string_input_producer(filename_list, shuffle=True)
# 创建读取CSV文件的TextLineReader
reader = tf.TextLineReader()
# 从文件名队列中取出CSV文件中的一条或多条记录保存在value中
#_, value = reader.read(filename_queue) #单条数据
_, value = reader.read_up_to(filename_queue, num_records=5) #多条数据
# 设置数据记录的默认值
record_defaults = [[0.0], [0.0], [0.0], [0.0]]
# 使用decode_csv方法将数据记录的每个字段都转换成特征张量
idd, age, income, outgo = tf.decode_csv(value, record_defaults=record_defaults)
# 将所有特征张量组合在一起形成一条记录
features = tf.stack([idd, age, income, outgo], axis=1) #根据需要设置stack的方式（0或1）

with tf.Session() as sess:
    print("start reading......")
    # 创建一个线程调节器
    coord = tf.train.Coordinator()
    # 启动Graph中所有队列的线程
    threads = tf.train.start_queue_runners(coord=coord)
    # 读取数据
    for i in range(20):
        row_data = sess.run(features)
        print("batch %i = " % int(i), row_data)
    # 请求停止线程
    coord.request_stop()
    # 等待线程结束
    coord.join(threads=threads)