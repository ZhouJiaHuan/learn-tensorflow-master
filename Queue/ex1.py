"""
an example of Queue in TensorFlow
@author: Meringue
@date: 2018/11/07
"""
import tensorflow as tf 
# 创建一个先进先出队列，指定最多保存2个元素，类型为整数
q = tf.FIFOQueue(4, "int32")
# 初始化队列中的元素
init = q.enqueue_many(([0 ,10, 20, 30], ))
# 执行出队操作
x = q.dequeue()
# 将元素+1后入队
y = x + 1
q_inc = q.enqueue([y])

with tf.Session() as sess:
    init.run() # 运行初始化队列
    for _ in range(5):
        # 执行出队+1操作
        v, _ = sess.run([x, q_inc])
        print(v)