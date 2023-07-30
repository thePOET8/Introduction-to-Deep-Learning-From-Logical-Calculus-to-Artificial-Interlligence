from keras.layers import Input, Dense
from keras.models import Model
from keras.datasets import mnist
import numpy as np
(x_train, _), (x_test, _) = mnist.load_data() #从keras加载MNIST数据集

#对MNIST数据集预处理
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0
noise_rate = 0.05

#引入噪点
x_train_noisy = x_train + noise_rate * np.random.normal(loc=0.0, scale=1.0, size=x_train.shape)
x_test_noisy = x_test + noise_rate * np.random.normal(loc=0.0, scale=1.0, size=x_test.shape)
x_test_noisy = np.clip(x_test_noisy, 0.0, 1.0)
x_train_noisy = np.clip(x_train_noisy, 0.0, 1.0)
#loc为平均值，scale为标准差

#
x_train = x_train.reshape((len(x_train),np.prod(x_train.shape[1:])))
x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))
x_train_noisy = x_train_noisy.reshape((len(x_train_noisy), np.prod(x_train_noisy.shape[1:])))
x_test_noisy = x_test_noisy.reshape((len(x_test_noisy), np.prod(x_test_noisy.shape[1:])))
assert  x_train_noisy.shape[1]  ==  x_test_noisy.shape[1]  #检查噪点代码和测试向量大小是否相同


#构建实际的编码器
inputs = Input(shape=(x_train_noisy.shape[1],))
encode1 = Dense(128, activation='relu')(inputs)
encode2 = Dense(64, activation='tanh')(encode1)
encode3 = Dense(32, activation='relu')(encode2)
decode3 = Dense(64, activation='relu')(encode3)
decode2 = Dense(128, activation='relu')(decode3)
decode1 = Dense(x_train_noisy.shape[1], activation='relu')(decode2)


#
autoencoder = Model(inputs, decode1)
autoencoder.compile(optimizer='sgd', loss='mean_squared_error', metrics=['accuracy'])
autoencoder.fit(x_train,x_train_noisy,epochs=5,batch_size=256,shuffle=True)


metrics = autoencoder.evaluate(x_test_noisy, x_test, verbose=1)
print()
print("%s:%2f%%" % (autoencoder.metrics_names[1], metrics[1]*100))
print()
results = autoencoder.predict(x_test)
all_AE_weights_shapes = [x.shape for x in autoencoder.get_weights()]
print(all_AE_weights_shapes)
ww=len(all_AE_weights_shapes)
deeply_encoded_MNIST_weight_matrix = autoencoder.get_weights()
[int((ww/2))]
for weight_matrix in deeply_encoded_MNIST_weight_matrix:
    print(weight_matrix.shape)
autoencoder.save_weights("all_AE_weight.h5")