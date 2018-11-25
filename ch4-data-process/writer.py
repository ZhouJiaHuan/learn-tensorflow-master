"""writer.py
an example of writing TFRecords file.
run this code to generate 'stat.tfrecord'
@author: Meringue
@date: 2018/10/29

data:
id  age income  outgo
1   24  2048.0  1024.0
2   48  4096.0  2048.0
"""
import tensorflow as tf 
# 创建向TFRecord文件写数据记录的writer
writer = tf.python_io.TFRecordWriter("stat.tfrecord")
# 两轮循环构造输入样例
for i in range(1,3):
    # 创建样例
    example = tf.train.Example(
        features = tf.train.Features(
            feature = {
                "id": tf.train.Feature(int64_list=
                                       tf.train.Int64List(value=[i])),
                "age": tf.train.Feature(int64_list=
                                        tf.train.Int64List(value=[i*24])),
                "income": tf.train.Feature(float_list=
                                           tf.train.FloatList(value=[i*2048.0])),
                "outgo": tf.train.Feature(float_list=
                                          tf.train.FloatList(value=[i*1024.0]))
            }
        )
    )
    # 将样例序列化后写入指定文件
    writer.write(example.SerializeToString())
# 关闭输出流
writer.close()
