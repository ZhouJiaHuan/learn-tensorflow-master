## ex1.py: run with "python ex1.py --help" for help
tf.app.run()用于加载由tf.app.flags.FLAGS定义的参数并调用main()函数或指定函数(可以在FCN.tensorflow-master中看到该用法)

主函数中的tf.app.run()会默认调用main()函数并传递参数，因此必须在main()函数中**设置一个参数位置**。也可以在tf.app.run()中传入**指定的函数名**：
```python
def test(args=None):
    # test function
if __name__ == "__main__":
    tf.app.run(test)
```