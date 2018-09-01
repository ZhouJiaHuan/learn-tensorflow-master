"""稀疏张量
auther: Meringue
date: 2018/7/5
"""

import tensorflow as tf 
sp = tf.SparseTensor(indices=[[0,0], [0,3], [1,3]], values=[1,2,3], dense_shape=[3,4])
sp_dense = tf.sparse_to_dense([[0,0], [0,3], [1,3]], [3,4], [1,2,3])
with tf.Session() as sess:
    print("sp = ", sp.eval())
    print("sp_dense = ", sp_dense.eval())
    
# 稀疏张量也有一套自己的运算操作，此处不多介绍