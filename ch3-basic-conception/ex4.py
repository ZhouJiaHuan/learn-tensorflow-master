"""计算节点（operation）、存储节点（Variable）、数据节点（Placeholder）
auther: Meringue
date: 2018/7/8
"""

import tensorflow as tf 
x = tf.placeholder(tf.float32) # 数据节点：用于存放用户指定的输入
W = tf.Variable(1.0) # 存储节点：存放变量
b = tf.Variable(1.0)
y = W*x+b
with tf.Session() as sess:
    tf.global_variables_initializer().run() # 调用operation的run方法
    fetch = y.eval(feed_dict={x: 3.0}) # x需要指定输入
    print("fetch = ", fetch)
    z = tf.add(W, b).eval() # 调用Tensor的eval()方法
    print("z = ", z)