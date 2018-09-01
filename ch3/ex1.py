"""张量的基本属性和成员方法
auther: Meringue
date: 2018/7/5
"""

import tensorflow as tf 

a = tf.constant([1, 1])
b = tf.constant([2, 2], name="const_b")
c = tf.add(a, b)
d = tf.multiply(c, c)

with tf.Session() as sess:
    # 张量的名称
    print("a.name = ", a.name)
    print("b.name = ", b.name)
    print("c.name = ", c.name)
    print("d.name = ", d.name)

    # 张量所属的数据流图
    print("a.graph = ", a.graph)
    print("b.graph = ", b.graph)
    print("c.graph = ", c.graph)
    print("d.graph = ", d.graph)

    # 张量的shape
    print("c.shape = ", c.shape) # 通过属性访问
    print("c.shape = ", c.get_shape()) # 通过成员方法访问

    # 张量的取值
    print("a = ", a.eval()) #通过成员方法访问
    print("b = ", b.eval())
    print("c = ", c.eval())
    print("d = ", sess.run(d)) # sess.run()获得
    
    # 张量的后置运算和前置运算
    print("a.consumers = ", a.consumers()) #后置运算（成员方法）
    print("b.consumers = ", b.consumers())
    print("c.consumers = ", c.consumers())
    print("c.ops = ", c.op) # 前置运算（属性）