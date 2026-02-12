import warnings
warnings.filterwarnings('ignore')
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import cifar10

cifar10_nn = cifar10.CIFAR10([32, 10], 20)

cifar10_nn.report()
cifar10_nn.export2csv()
cifar10_nn.export2onnx()
cifar10_nn.report2file()


## from file
cifar10_nn_from_file = cifar10.CIFAR10FromDir("./CIFAR10-mlp")
cifar10_nn_from_file.report()