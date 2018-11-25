model_name="cifar10-5layers"
model_path="models/cifar10-5layers_Gaussian-50000.data-00000-of-00001"
echo "model_name = $model_name, model path = $model_path"
python test.py --model $model_name --path $model_path

model_name="cifar10-5layers"
model_path="models/cifar10-5layers_Xavier-50000.data-00000-of-00001"
echo "model_name = $model_name, model path = $model_path"
python test.py --model $model_name --path $model_path

model_name="cifar10-5layers"
model_path="models/cifar10-5layers_He-50000.data-00000-of-00001"
echo "model_name = $model_name, model path = $model_path"
python test.py --model $model_name --path $model_path

model_name="cifar10-8layers"
model_path="models/cifar10-8layers_Gaussian-50000.data-00000-of-00001"
echo "model_name = $model_name, model path = $model_path"
python test.py --model $model_name --path $model_path

model_name="cifar10-8layers"
model_path="models/cifar10-8layers_Xavier-50000.data-00000-of-00001"
echo "model_name = $model_name, model path = $model_path"
python test.py --model $model_name --path $model_path

model_name="cifar10-8layers"
model_path="models/cifar10-8layers_He-50000.data-00000-of-00001"
echo "model_name = $model_name, model path = $model_path"
python test.py --model $model_name --path $model_path
