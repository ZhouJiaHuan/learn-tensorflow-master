echo "model = cifar10-5layers, init method = Gaussian"
python train.py --model 'cifar10-5layers' --init 'Gaussian'
echo "model = cifar10-5layers, init method = Xavier"
python train.py --model 'cifar10-5layers' --init 'Xavier'
echo "model = cifar10-5layers, init method = He"
python train.py --model 'cifar10-5layers' --init 'He'

echo "model = cifar10-8layers, init method = Gaussian"
python train.py --model 'cifar10-8layers' --init 'Gaussian'
echo "model = cifar10-8layers, init method = Xavier"
python train.py --model 'cifar10-8layers' --init 'Xavier'
echo "model = cifar10-8layers, init method = He"
python train.py --model 'cifar10-8layers' --init 'He'
