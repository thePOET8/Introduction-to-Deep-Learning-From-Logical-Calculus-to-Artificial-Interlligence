import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers.core import Dense
TARGET_VARIABLE ="user_action"   # 希望预测的目标变量
TRAIN_TEST_SPLIT=0.5    #训练集中的数据点比例设置为0.5
HIDDEN_LAYER_SIZE=30
raw_data = pd.read_csv("data.csv")

mask = np.random.rand(len(raw_data)) < TRAIN_TEST_SPLIT   #定义随机的数据抽样，用于获取训练-测试拆分
tr_dataset = raw_data[mask]
te_dataset = raw_data[~mask]

tr_data = np.array(tr_dataset.drop(TARGET_VARIABLE,axis=1))
tr_label = np.array(tr_dataset[[TARGET_VARIABLE]])
te_data = np.array(te_dataset.drop(TARGET_VARIABLE,axis=1))
te_label = np.array(te_dataset[[TARGET_VARIABLE]])
#将训练和测试数据框拆分成标签和数据，然后将其转换为Numpy数组
#对原文中提供的代码进行了修改：训练数据tr_data和tr_label是从整个raw_data DataFrame中创建的，而不是训练集tr_dataset。这可能会导致数据泄漏，即模型是在后来被用于测试集的数据上训练的。

ffnn = Sequential()  #ffnn中初始化一个新的顺序模型
ffnn.add(Dense(HIDDEN_LAYER_SIZE, input_shape=(3,),activation="sigmoid"))  #指定输入层（接受三维向量作为单个数据输入）以及隐藏层大小（通过变量HIDDEN_LAYER_SIZE=30指定）
ffnn.add(Dense(1, activation="sigmoid"))     #将从上一层获取隐藏层大小（Keras会自动执行）
ffnn.compile(loss="mean_squared_error", optimizer="sgd", metrics =['accuracy'])    #指定误差函数（MSE）、优化器（随机梯度下降）以及要计算的指标，同时编译模型，从我们指定的位置聚合、收集Python需要的的所有其他内容
ffnn.fit(tr_data, tr_label, epochs=150, batch_size=2, verbose=1)   #针对tr_data对神经网络进行训练，使用tr_labels，执行150次epoch，在一个小批中包含包含两个样本，verbose=1表示在每次训练epoch之后，它将输出准确度和损。

metrics = ffnn.evaluate(te_data, te_label, verbose=1)   #使用te_labels针对te_data来评估模型
print("%s: %.2f%%" % (ffnn.metrics_names[1], metrics[1]*100))    #以格式化字符串的形式输出精确度

new_data = np.array(pd.read_csv("new_data.csv"))
results = ffnn.predict(new_data)
print(results)



