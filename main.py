import numpy as np
from train import MNIST,load_data
from PIL import Image
import paddle

def load_image(img_path):
    img = Image.open(img_path)
    img.show()
    img = img.resize((28,28))
    img = np.array(img).reshape(1,1,28,28).astype('float32')
    img = 1.0 - img/255
    return img

def main():
    model = MNIST()
    params_file_path = 'mnist.pdparams'
    img_path = './example_9.jpg'
    model_dict =  paddle.load("mnist.pdparams")
    model.load_dict(model_dict)

    model.eval()
    tensor_img = load_image(img_path)
    results = model(paddle.to_tensor(tensor_img))
    lab = np.argsort(results.numpy())
    print('本次预测的字体是{}'.format(lab[0][1]))
    print(' ')
    
main()