"""计算节点：Operation
auther: Meringue
date: 2018/7/8
"""

import tensorflow as tf 

with tf.name_scope("AddExample"): # 命名空间
    a = tf.Variable(1.0, name="a")
    b = tf.Variable(2.0, name="b")
    c = tf.add(a, b, name="Add")
    print("c = ", c) # 张量c的属性

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer()) # 如果定义了变量需要初始化
    print("c_value = ", c.eval())