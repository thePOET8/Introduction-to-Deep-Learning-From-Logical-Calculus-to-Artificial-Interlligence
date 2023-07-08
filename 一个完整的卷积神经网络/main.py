import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPool2D
from tensorflow.keras.utils import to_categorical
#此处原文为”from keras.utils import np_utils“，现版本已被移除。
#在最新版本的Keras中，np_utils模块已经被移除
#因此导入from keras.utils import np_utils会导致AttributeError错误。
#如果你想要进行独热编码的转换，可以使用to_categorical函数来实现。这个函数现在已经被移到了tensorflow.keras.utils模块中。

from keras.datasets import mnist
(train_samples, train_labels), (test_samples, test_labels) = mnist.load_data()
#使用4个不同变量加载训练样本、训练标签、测试样本、测试标签
train_samples=train_samples.reshape(train_samples.shape[0],28,28,1)
test_samples=test_samples.reshape(test_samples.shape[0],28,28,1)
#改变保存MNIST数据集的数组的结构，生成了一个（60000，28，28，1）维的数组
#第一维实际上是样本的数量，第二维和第三维用于表示尺寸28×28的图像，最后一维表示通道
train_samples= train_samples.astype('float32')
test_samples= test_samples.astype('float32')
#将数组中的条目声明为float32类型，Numpy（可以显著加速计算）需要类声明
train_samples = train_samples/255
test_samples = test_samples/255
#将数组条目从0-255的范围归一化为0~1的范围（可以解释为一个像素的灰度百分比）



c_train_labels = to_categorical(train_labels, 10)
c_test_labels = to_categorical(test_labels, 10)
#数据预处理

convnet = Sequential()
#创建一个空模型，其他行将填充网络规格
convnet.add(Convolution2D(32, 3, 3, activation='relu', input_shape=(28,28,1)))
#添加第一层，本例中是添加一个卷积层，需要生成32个特征图，使用ReLU为激活函数，并且具有3×3的感受野
convnet.add(MaxPool2D(pool_size=(1,1)))
#定义一个最大池化层，池大小为（1，1）
convnet.add(Convolution2D(32, 3, 3, activation='relu'))
#指定第三层，也是卷积层，感受野为3×3
convnet.add(MaxPool2D(pool_size=(2,2)))
#定义一个最大池化层，池大小为（2，2）
convnet.add(Dropout(0.3))
#丢弃”层“，只是上一层与下一层之间连接的修改方式，连接修改为对所有连接包含值为0.3的丢弃率。
convnet.add(Flatten())
#压平张量（将固定大小的矩阵转化为向量
convnet.add(Dense(10, activation='softmax'))
#将压平的向量馈送到最后一层，一个标准的全连接前馈层，接受的输入数量就是压平向量中的分量数，并且输出10个值（10个输出神经元）
#使用的Softmax激活函数是适用与两个以上类的逻辑函数版本

convnet.compile(loss='mean_squared_error', optimizer='sgd', metrics=('accuracy'))
#编译模型。训练方法为’sgd‘（随机梯度下降），误差指定为MSE，此外要求Keras在训练计算准确率
convnet.fit(train_samples, c_train_labels, batch_size=32, epochs=20, verbose=1)
#使用的批大小为32，训练20次epoch，verbose标志设置为1，表示将输出训练的详细信息

metrics = convnet.evaluate(test_samples, c_test_labels, verbose=1)
print()
print("%s: %.2f%%" % (convnet.metrics_names[1], metrics[1]*100))
preddiction = convnet.predict(test_samples)
#如果想要使用它进行预测，在此应该使用一些新的样本，除了第一维之外，必须与test_samples具有相同的维度，而第一维用于保存各个训练样本。
#变量prediction将与c_test_labels具有完全相同的维度

#本代码在原书基础上基于版本改变做了一些改变