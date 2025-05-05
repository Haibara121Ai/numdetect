import json
import gzip
import random
import numpy as np
import paddle
from paddle.nn import Linear, MaxPool2D

def load_data(mode='train'):
    datafile = './mnist.json.gz'
    print('loading mnist dataset from{}……'.format(datafile))
    data = json.load(gzip.open(datafile))
    train_set,val_set,eval_set = data

    IMG_ROWS = 28
    IMG_COLS = 28

    if mode == 'train':
        imgs = train_set[0]
        labels = train_set[1]
    elif mode == 'valid':
        imgs = val_set[0]
        labels = train_set[1]
    elif mode == 'eval':
        imgs = eval_set[0]
        labels = eval_set[1]
    imgs_length = len(imgs)

    assert len(imgs) == len(labels)

    print('数据集校验正常')
    index_list = list(range(imgs_length))

    BATCHSIZE = 100

    def data_generator():
        if mode == 'train':
            random.shuffle(index_list)
        imgs_list = []
        labels_list = []
        for i in index_list:
            img = np.reshape(imgs[i],[1,IMG_ROWS,IMG_COLS]).astype('float32')
            label = np.reshape(labels[i],[1]).astype('int64')
            imgs_list.append(img)
            labels_list.append(label)
            if len(imgs_list) == BATCHSIZE:
                yield np.array(imgs_list),np.array(labels_list)
                imgs_list = []
                labels_list = []
            if len(imgs_list) > 0:
                yield np.array(imgs_list),np.array(labels_list)
    return data_generator

class MNIST(paddle.nn.Layer):
    def __init__(self):
        super(MNIST,self).__init__()
        self.conv1 = paddle.nn.Conv2D(1, 20, 5, stride=1, padding=2)
        self.relu1 = paddle.nn.ReLU()
        self.pool1 = MaxPool2D(2, 2)
        self.conv2 = paddle.nn.Conv2D(20, 20, 5, stride=1, padding=2)
        self.relu2 = paddle.nn.ReLU()
        self.pool2 = MaxPool2D(2, 2)
        self.fc = Linear(980, 10)

    def forward(self,inputs):
        x = self.conv1(inputs)
        x = self.relu1(x)
        
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)
        x = paddle.reshape(x,[x.shape[0],980])
        x = self.fc(x)
        return x

# def load_img(img_path):
#     im = Image.open(img_path).convert('L')
#     im.show()
#     im = im.resize((28,28))
#     im = np.array(im).reshape(1,1,28,28).astype(np.float32)
#     im = 1.0-im/255
#     return im
def train():
    model =  MNIST()
    model.train()
    train_load = load_data('train')
    optimizer = paddle.optimizer.SGD(learning_rate = 0.01,parameters = model.parameters())

    EPOCH_NUM = 3
    for epoch_id in range(EPOCH_NUM):
        for batch_id,data in enumerate(train_load()):
            image_data,label_data = data
            image = paddle.to_tensor(image_data)
            label = paddle.to_tensor(label_data)

            predict = model(image)

            loss = paddle.nn.functional.cross_entropy(predict,label)
            avg_loss = paddle.mean(loss)

            if batch_id % 50 == 0:
                print("epoch:{},batch:{},loss is :{}".format(epoch_id,batch_id,avg_loss.numpy()))

            avg_loss.backward()
            optimizer.minimize(avg_loss)
            model.clear_gradients()

    paddle.save(model.state_dict(), 'mnist.pdparams')
