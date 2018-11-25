# cifar-10-Tensorflow
cifar-10 training and test with CNN  with Tensorflow.

## dependence
- Python 2.7 or Python3.x (modify some codes, see codes)
- numpy
- matplotlib
- TensorFlow >= 1.3.0

## Experimental results
- learning rate = 0.001
- training methods = Adam
- steps = 50,000
- GPU = NVIDIA 1070

### cifar10\_5layers
- Gaussian initialization
```
python train.py --model "cifar10-5layers" --init "Gauss"

test accuracy = 0.7814
```

![cifar10-5layers_Gaussian](/figures/cifar10-5layers_Gaussian.png)

- Xavier initialization
```
python train.py --model "cifar10-5layers" --init "Xavier"

test accuracy = 0.8081
```
![cifar10-5layers_Xavier](/figures/cifar10-5layers_Xavier.png)
- He initialization
```
python train.py --model "cifar10-5layers" --init "He"

test accuracy = 0.7968
```
![cifar10-5layers_He](/figures/cifar10-5layers_He.png)

### cifar10\_8layers
- Gaussian initialization
```
python train.py --model "cifar10-8layers" --init "Gauss"

test accuracy = 0.7552
```

![cifar10-8layers_Gaussian](/figures/cifar10-8layers_Gaussian.png)

- Xavier initialization
```
python train.py --model "cifar10-8layers" --init "Xavier"

test accuracy = 0.8175
```

![cifar10-8layers_Xavier](/figures/cifar10-8layers_Xavier.png)

- He initialization
```
python train.py --model "cifar10-8layers" --init "He"

test accuracy = 0.7703
```

![cifar10-8layers_He](/figures/cifar10-8layers_He.png)