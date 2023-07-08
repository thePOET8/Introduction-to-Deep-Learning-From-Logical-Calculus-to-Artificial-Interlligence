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
from keras.optimizers import Adam


(train_samples, train_labels), (test_samples, test_labels) = mnist.load_data()
#使用4个不同变量加载训练样本、训练标签、测试样本、测试标签
train_samples=train_samples.reshape(train_samples.shape[0],28,28,1)
test_samples=test_samples.reshape(test_samples.shape[0],28,28,1)
train_samples= train_samples.astype('float32')
test_samples= test_samples.astype('float32')
train_samples = train_samples/255
test_samples = test_samples/255

c_train_labels = to_categorical(train_labels, 10)
c_test_labels = to_categorical(test_labels, 10)
#数据预处理

convnet = Sequential()
convnet.add(Convolution2D(32, 3, 3, activation='relu', input_shape=(28,28,1)))
convnet.add(MaxPool2D(pool_size=(1,1)))
convnet.add(Convolution2D(32, 3, 3, activation='relu'))
convnet.add(MaxPool2D(pool_size=(2,2)))
convnet.add(Dropout(0.3))
convnet.add(Flatten())
convnet.add(Dense(10, activation='softmax'))

convnet.compile(loss='mean_squared_error', optimizer=Adam(learning_rate=0.001), metrics=('accuracy'))
convnet.fit(train_samples, c_train_labels, batch_size=32, epochs=30, verbose=1)

metrics = convnet.evaluate(test_samples, c_test_labels, verbose=1)
print()
print("%s: %.2f%%" % (convnet.metrics_names[1], metrics[1]*100))
preddiction = convnet.predict(test_samples)