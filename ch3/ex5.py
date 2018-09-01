"""线性模型
auther: Meringue
date: 2018/7/8
"""

import numpy as np 
import tensorflow as tf 
import matplotlib.pyplot as plt

# 定义超参数
learning_rate = 2e-3
max_train_steps = 1000
batch_size = 30

# 输入数据
train_X = np.arange(0, batch_size, dtype=np.float32).reshape([batch_size,1])
train_Y = 0.8*train_X + 2 * (np.random.rand(batch_size,1)-1)

# 构建模型
X = tf.placeholder(tf.float32, [None, 1])
W = tf.Variable(tf.random_normal([1, 1], name="weight"))
b = tf.Variable(tf.zeros([1], name="bias"))
Y = tf.matmul(X, W) + b

# 损失函数
Y_ = tf.placeholder(tf.float32, [None, 1])
loss = tf.reduce_sum(tf.pow(Y-Y_, 2))/batch_size

# 优化器
optimizer = tf.train.GradientDescentOptimizer(learning_rate)
train_op = optimizer.minimize(loss)

# 训练
steps = []
train_loss = []
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print("start training...")
    for step in range(max_train_steps):
        sess.run(train_op, feed_dict={X: train_X, Y_: train_Y})
        if step%10 == 0:
            ls = sess.run(loss, feed_dict={X: train_X, Y_: train_Y})
            steps.append(step)
            train_loss.append(ls)
            print("Step=%d, loss=%.4f, W=%.4f, b=%.4f" %(step, ls, W.eval(), b.eval()))
    weitht, bias = sess.run([W, b])
    final_loss = sess.run(loss, feed_dict={X: train_X, Y_: train_Y})
    print("Step=%d, loss=%.4f, W=%.4f, b=%.4f" %(max_train_steps, final_loss, W.eval(), b.eval()))

# 模型可视化
plt.figure(1)
plt.plot(steps, train_loss, label="train loss")
#plt.ylim([0, 30])
plt.legend()
plt.figure(2)
plt.plot(train_X, train_Y, "ro", label="train data")
plt.plot(train_X, train_X*weitht+bias, label="fitted line") 
plt.legend()
plt.show()