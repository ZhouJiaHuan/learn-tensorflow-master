""" an example of tf.app.run()
author: Meringue
date: 2018/09/07
"""
import tensorflow as tf

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string("str", "train", "str is a string")
tf.app.flags.DEFINE_float("lr", 0.001, "learning rate")
tf.app.flags.DEFINE_bool("flag", True, "True or False")

def main(argv=None):
    print("string = ", FLAGS.str)
    print("learning rate = ", FLAGS.lr)
    print("flag = ", FLAGS.flag)

if __name__ == "__main__":
    tf.app.run()
