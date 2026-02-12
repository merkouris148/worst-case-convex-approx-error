import warnings
warnings.filterwarnings('ignore')
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import fashion

fashion_nn = fashion.Fashion()

fashion_nn.report()
fashion_nn.export2csv()
fashion_nn.export2onnx()


## from file
fashion_nn_from_file = fashion.FashionFromDir("./Fashion-mlp")
fashion_nn_from_file.report()