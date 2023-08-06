# CBOW Word2vec实现示例
from keras.models import Sequential
from keras.layers.core import Dense
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

text_as_list = ["who", "are", "you", "that", "you", "do", "not", "know", "your", "history"]
embedding_size = 300
context = 2

#与循环神经网络对应的代码相同
distinct_words = set(text_as_list)
number_of_words = len(distinct_words)
word2index = dict((w, i) for i, w in enumerate(distinct_words))
index2word = dict((i, w) for i, w in enumerate(distinct_words))

# 创造一个函数，该函数生成两个列表，一个是主单词列表，一个是给定单词的语境单词列表（由列表构成的列表）
def create_word_context_and_main_words_lists(text_as_list):
    input_words = []
    label_words = []
    for i in range(0, len(text_as_list)):
        label_words.append(text_as_list[i])
        context_list = []
        if i >= context and i < (len(text_as_list) - context):
            context_list.append(text_as_list[i - context:i])
            context_list.append(text_as_list[i + 1:i + 1 + context])
            context_list = [x for subl in context_list for x in subl]
        elif i < context:
            context_list.append(text_as_list[:i])
            context_list.append(text_as_list[i + 1:i + 1 + context])
            context_list = [x for subl in context_list for x in subl]
        elif i >= (len(text_as_list) - context):
            context_list.append(text_as_list[i - context:i])
            context_list.append(text_as_list[i + 1:])
            context_list = [x for subl in context_list for x in subl]
        input_words.append(context_list)
    return input_words, label_words

input_words, label_words = create_word_context_and_main_words_lists(text_as_list)

input_vectors = np.zeros((len(text_as_list), number_of_words), dtype=np.int16)
vectorized_labels = np.zeros((len(text_as_list), number_of_words), dtype=np.int16)

for i, input_w in enumerate(input_words):
    for j, w in enumerate(input_w):
        input_vectors[i, word2index[w]] = 1
        vectorized_labels[i, word2index[label_words[i]]] = 1

# 初始化Keras模型，并对其进行训练
word2vec = Sequential()
word2vec.add(Dense(embedding_size, input_shape=(number_of_words,), activation="linear", use_bias=False))
word2vec.add(Dense(number_of_words, activation="softmax", use_bias=False))
word2vec.compile(loss='mean_squared_error', optimizer="sgd", metrics=['accuracy'])
word2vec.fit(input_vectors, vectorized_labels, epochs=1500, batch_size=10, verbose=1)

# 之后，提取权重
word2vec.save_weights("all_weight.h5")
embedding_size_weight_matrix = word2vec.get_weights()[0]

pca = PCA(n_components=2)
pca.fit(embedding_size_weight_matrix)
results = pca.transform(embedding_size_weight_matrix)
x = np.transpose(results).tolist()[0]
y = np.transpose(results).tolist()[1]
n = list(word2index.keys())

fig, ax = plt.subplots()
ax.scatter(x, y)
for i, txt in enumerate(n):
    ax.annotate(txt, (x[i], y[i]))

plt.savefig('word_vectors_in_2D_space.png')
plt.show()