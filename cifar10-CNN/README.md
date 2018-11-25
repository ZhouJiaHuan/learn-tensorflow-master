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

test accuracy =
training time =
```

- Xavier initialization
```
python train.py --model "cifar10-5layers" --init "Xavier"

test accuracy =
training time =
```

- He initialization
```
python train.py --model "cifar10-5layers" --init "He"

test accuracy =
training time =
```

### cifar10\_8layers
- Gaussian initialization
```
python train.py --model "cifar10-8layers" --init "Gauss"

test accuracy =
training time =
```
- Xavier initialization
```
python train.py --model "cifar10-8layers" --init "Xavier"

test accuracy =
training time =
```
- He initialization
```
python train.py --model "cifar10-8layers" --init "He"

test accuracy =
training time =
```