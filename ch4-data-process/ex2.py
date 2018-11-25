"""ex2.py
an example of creating and initializing model parameters.
@author: Meringue
@date: 2018/11/05

tf.Variable()的初值可以是张量、生成张量的方法（符合某一分布、符合某种生成规则
的序列张量、常量等）、通convert\_to\_tensor方法转化的数据类型（列表、元组等）、
已经初始化的变量。
"""
import tensorflow as tf 
W1 = tf.Variable(tf.random_normal(shape=[1,4], mean=0, stddev=1.0), name="W1")
W2 = tf.Variable(tf.constant(-1.0, shape=[1,4]), name="W2")
W3 = tf.Variable(tf.linspace(start=1.0, stop=5.0, num=3), name="W3")
W4 = tf.Variable(W1.initialized_value()*2, name="w4") # 调用W1的初值

with tf.Session() as sess:
    #sess.run(tf.global_variables_initializer())
    #sess.run(tf.variables_initializer([W1, W2, W3, W4])) # 给定需要初始化的变量集合
    sess.run(tf.variables_initializer(tf.trainable_variables())) # 变量的trainable默认为True
    print("W1 = ", W1.eval())
    print("W2 = ", W2.eval())
    print("W3 = ", W3.eval())
    print("W4 = ", W4.eval())
    print(sess.run(tf.trainable_variables()))