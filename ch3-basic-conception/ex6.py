"""线性模型
auther: Meringue
date: 2018/7/15
"""
import tensorflow as tf 
import numpy as np 

# 生成模拟数据
x_data = np.float32(np.random.rand(2, 100))
y_data = np.matmul([0.100, 0.200], x_data) + 0.300

# 构造线性模型
b = tf.Variable(tf.zeros([1]))
W = tf.Variable(tf.random_uniform([1,2], -1, 1))
y = tf.matmul(W, x_data) + b

# 损失函数
loss = tf.reduce_mean(tf.square(y-y_data))
optimizer = tf.train.GradientDescentOptimizer(0.5)
train = optimizer.minimize(loss)
with tf.Session() as sess:
    tf.global_variables_initializer().run()
    for step in range(200+1):
        train.run()
        if step%10 == 0:
            print("step =", step, "W = ", W.eval(), "b = ", b.eval())


