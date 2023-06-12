from flask import Flask, jsonify, request,render_template
from keras.models import load_model
import jieba
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from keras.preprocessing.text import Tokenizer
from keras.utils import pad_sequences
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
#导包
import random
import torch
from torch import nn
from collections import Counter
import torch.nn.functional as F
import tensorflow as tf
 
#MLP
class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(5, 100)
        self.fc2 = nn.Linear(100, 100)
        self.fc3 = nn.Linear(100, 2500)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Load your .h5 model
model = load_model("cnn_model.h5")
model2 = load_model("lstm_model.h5")
# 加载模型
model3 = torch.load('mlp.pt')

df2 = pd.read_csv('movie_lable.csv', engine='python',encoding='utf-8',skip_blank_lines=True)

df=pd.read_excel('combine_df.xlsx')
# 拟合tokenizer
tokenizer = Tokenizer(num_words=1000)

movie_synopses = [' '.join(jieba.cut(s)) for s in df['des']]

tokenizer.fit_on_texts(movie_synopses)


# 处理词汇表

#这里我想因为标签和类别的长度不一样 我想标签都取前两个 类别取前两个 然后分词 类别只有一个的情况就补一个词
#然后导演不分词 把导演名字看成一个词
#剪裁类型函数
def cut_category(category):
  num, index = 0, 0
  for i in category:
    if i == '/':
      num += 1
    if num == 2:
      return category[:index]
    index += 1
  if num == 0:
    return category + str('/单一')
  else:
    return category 
  
#剪裁标签函数
def cut_tag(tag):
  num, index = 0, 0
  for i in tag:
    if i == '/':
      num += 1
    if num == 1:
      return tag[:index]
    index += 1
  return tag
  
# print(df.head(2))
#通过该模型，做一个有标号的训练集
df_list = np.array(df2).tolist() #直接把源文件转化成一个list准备添加标号
cnt = 0
#处理所有的类型和标签
for i in df_list:
  i[3] = cut_category(i[3])
  i[5] = cut_tag(i[5])
  # print(i[3])
  # print(i[5])
  cnt += 1
#list重新转换成df
df2 = pd.DataFrame(df_list, columns=['name', 'director',  'rating', 'category', 'duration', 'tag' ,'lables'])
df3 = df2
print(df2.head(5))

# 分词并过滤停用词
stopwords = set(pd.read_csv('stopwords.txt', sep='\t', header=None, engine='python',quoting=3,encoding='utf-8')[0])
#电影名没用，不分词
#导演看成一个词不分词
df2['category'] = df2['category'].apply(lambda x: ' '.join([w for w in jieba.cut(x) if w not in stopwords]))
#tag看成一个词

# 将电影名称、时长、评分和数字特征组合成新的数据框
# word_df = df[['director', 'tag', 'category', 'duration', 'rating']] #这个是去除掉电影名之后的电影属性，因为电影名没啥用
word_df = df2[['tag', 'category', 'duration', 'rating']] #这个是输入的向量df
lables_df = df2['lables'] #这是输出df
select_df = df2[['name','director', 'tag', 'category', 'duration', 'rating','lables']] #这个df是为了到时候好查找电影名

print(word_df.head(3))
print(lables_df.head(3))

#将每个电影的属性全部拼成一个分词好的字符串
line_df = list() #将每部电影属性 所有有用的词 都搞成一列 存在line_df中
select_list = np.array(select_df).tolist()
list_df = np.array(word_df).tolist()
for i in list(list_df):
    line = ''
    for j in i:
        line += ' '
        line += str(j)
    line_df.append(line)
# 分词并统计词频
#重新写分词函数
#拆一行的函数
def split_movie(line):
  res = []
  j = 0
  length = len(line)
  for i in range(length):
    if line[j] == ' ':
      j += 1
      continue
    if line[i] == ' ' or i == length:
      res.append(line[j:i])
      j = i
  res.append(line[-3:])
  return res


words = []
for line in line_df:
    # words += jieba.cut(line)
    #手动分词
    word_list = split_movie(line)
    for word in word_list:
      words.append(word)
word_freq = Counter(words)

print(words[:10])
vocab_size = 1510 # 设定词汇表大小 理论上来讲 每部电影4个词加上评分和数字应该6个 10000条电影 60000个词 但是很多重复的 所以 取6500个
vocab = ["<PAD>", "<UNK>"] + [word for word, _ in word_freq.most_common(vocab_size - 2)]
word_to_idx = {word: idx for idx, word in enumerate(vocab)}
idx_to_word = {idx: word for word, idx in word_to_idx.items()}

print(vocab_size)
print(len(vocab))

#传一行用户需要预测的数据进来 转化成tensor
def get_tensor(line):
   # 将电影数据转换为数字序列
    max_len = 5 # 设定最大序列长度 每部电影最大50个词

    #将一行电影拆开成一个个词 做成一个向量

    line_words = split_movie(line)
    seq = [word_to_idx.get(word, 1) for word in line_words][:max_len]
    seq += [0] * (max_len - len(seq))

    print(seq)
    
    return torch.tensor(seq)
    

# ---------------------------------------------------------------------

app = Flask(__name__)



@app.route('/')
def index():
    return render_template('index.html')

@app.route('/cnn')
def cnn():
    return render_template('cnn.html')

@app.route('/lstm')
def lstm():
    return render_template('lstm.html')

@app.route('/mlp')
def mlp():
   return render_template('mlp.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get input data from JSON payload
    print("进入了预测函数")
    # data = request.get_json(force=True)
    # print(data['input'])
    # input_data = preprocess_input(data['input'])
    input_data = request.form['input'] 
    synopsis_model2 =input_data

    synopsis_model2 = ' '.join(jieba.cut(synopsis_model2))
    sequence_model2 = tokenizer.texts_to_sequences([synopsis_model2])
    padded_sequence_model2 = pad_sequences(sequence_model2, maxlen=50)


    pred = model.predict(padded_sequence_model2)

    pred_class = np.argmax(pred)

    predicted_label = df[df['lables'] == pred_class][:10]

    print('预测的电影为:', predicted_label['name'],predicted_label['label'])
    # similar_movies_indices = np.argsort(pred)[0][-10:]

    # similar_movies_model2 = df.iloc[similar_movies_indices]
    
    # print(similar_movies_model2)

    return render_template('predict.html', data=predicted_label)

@app.route('/predict2', methods=['POST'])
def predict2():
    # Get input data from JSON payload
    print("进入了预测函数2")
    # data = request.get_json(force=True)
    # print(data['input'])
    # input_data = preprocess_input(data['input'])
    input_data = request.form['input'] 
    synopsis_model2 =input_data

    synopsis_model2 = ' '.join(jieba.cut(synopsis_model2))
    sequence_model2 = tokenizer.texts_to_sequences([synopsis_model2])
    padded_sequence_model2 = pad_sequences(sequence_model2, maxlen=50)


    pred = model.predict(padded_sequence_model2)


    pred_class = np.argmax(pred)

    predicted_label = df[df['lables'] == pred_class][:10]

    print('预测的电影为:', predicted_label['name'],predicted_label['label'])
    # similar_movies_indices = np.argsort(pred)[0][-10:]

    # similar_movies_model2 = df.iloc[similar_movies_indices]
    
    # print(similar_movies_model2)

    return render_template('predict.html', data=predicted_label)

@app.route('/predict3', methods=['POST'])
def predict3():
    tag = request.form['tag'] 
    category1 = request.form['category1'] 
    category2 = request.form['category2'] 
    duration = request.form['duration'] 
    rating = request.form['rating'] 
    line = tag + " " + category1 + " " + category2 + " " + duration + " " + rating
    print("line:" , end=" ")
    print(line)
    ans = model3(get_tensor(line).reshape(-1,5).float())
    # print(ans)
    # 对张量进行softmax操作，dim=0代表按列计算
    probs = torch.softmax(ans, dim=1)

    # 输出概率最大的类别
    max_class = torch.argmax(probs).item()
    print(max_class)
    # 使用 iloc 方法进行条件筛选
    selected_rows = df3[df3['lables'] == max_class]
    # print("df3:")
    # print(df3.head(5))
    # print("selected_rows:")
    # print(selected_rows)
    return render_template('predict.html', data=selected_rows)


if __name__ == '__main__':
    app.run(debug=True)