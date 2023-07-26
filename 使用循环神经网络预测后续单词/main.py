#使用循环神经网络预测后续单词
import numpy as np
#导入库
from keras.layers import Dense, Activation
from keras.layers import LSTM
#from keras.layers.recurrent import SimpleRNN
#(In newer versions of Keras (2.0 and above), the recurrent layers have been moved to the 'keras.layers' module. So, instead of 'keras.layers.recurrent', you should import the specific recurrent layer you want to use from 'keras.layers'.)
from keras.models import Sequential

#定义超参数
hidden_neurons = 50   #使用50个隐藏单元
my_optimizer ="sgd"   #使用随机梯度下降优化器
batch_size = 60       #在一次随机梯度下降迭代中使用60个示例
error_function = "mean_squared_error"   #告诉Keras使用MSE的误差函数
output_nonlinearity = "softmax"         #“softmax”激活函数
cycles = 5
epochs_per_cycle = 3
context = 3

#文字处理部分
def create_tesla_text_from_file(textfile="tesla.txt"):
    clean_text_chunks = []
    with open(textfile, 'r', encoding='utf=8') as text:
        for line in text:
            clean_text_chunks.append(line)
    clean_text = ("".join(clean_text_chunks)).lower()
    text_as_list = clean_text.split()
    return text_as_list
text_as_list = create_tesla_text_from_file()
#把纯文本文件“tesla.txt"打开（UIF-8编码），
#把打开的行聚集在一个称为”clean_text_chunks"的列表中，
#然后合并在一个称为”clean_text“的大字符串中，再切割成单个的单词，并储存在名为text_as_list的列表中


#到此所有的文本都位于一个列表中，其中每个单独的元素是一个单词

#处理单词重复的问题
distinct_words = set(text_as_list)
numbers_of_words = len(distinct_words)  #计算文本中的单词数
word2index = dict((w,1) for i,w in enumerate(distinct_words))    #创建字典，以唯一的单词为键，以它们在文本中的位置为值
index2word = dict((i,w) for i,w in enumerate(distinct_words))    #创建字典，以位置作为键，将单词作为值

def create_word_indices_for_text(text_as_list):
    input_words = []
    label_word = []
    for i in range(0,len(text_as_list) - context):
        input_words.append((text_as_list[i:i+context]))
        label_word.append((text_as_list[i+context]))
    return input_words, label_word
input_words,label_word = create_word_indices_for_text(text_as_list)


input_vectors = np.zeros((len(input_words), context, numbers_of_words), dtype=np.int16)
vectorized_labels = np.zeros((len(input_words), numbers_of_words), dtype=np.int16)

#对张量进行”爬网“
for i, input_w in enumerate(input_words):
    for j, w in enumerate(input_w):
        input_vectors[i, j, word2index[w]] = 1
        vectorized_labels[i, word2index[label_word[i]]] = 1

#使用Keras函数指定完整的简单循环神经网络
model = Sequential()
model.add(LSTM(hidden_neurons, return_sequences=False,
input_shape=(context,numbers_of_words), unroll=True))
model.add(Dense(numbers_of_words))
model.add(Activation(output_nonlinearity))
model.compile(loss=error_function, optimizer=my_optimizer)

for cycle in range(cycles):
    print("> - <" * 50)
    print("Cycle: %d" % (cycle+1))
    model.fit(input_vectors, vectorized_labels, batch_size= batch_size, epochs= epochs_per_cycle)
    test_index = np.random.randint(len(input_words))
    test_words = input_words[test_index]
    print("Generating test from test index %s: %s" % (test_index, test_words))
    input_for_test = np.zeros((1, context, numbers_of_words))
    for i, w in enumerate(test_words):
        input_for_test[0, i, word2index[w]]= 1
    predictions_all_matrix = model.predict(input_for_test, verbose = 0)[0]
    predictions_word = index2word[np.argmax(predictions_all_matrix)]
    print("THE COMPLETE RESULTING SENTENCE IS: %s %s" % (''.join(test_words), predictions_word))
    print()