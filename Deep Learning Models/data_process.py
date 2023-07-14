import os
import pandas as pd
from fastai import *
from fastai.vision import *
from fastai.text import *
from fastai.callback import *
from pytorch_pretrained_bert import BertTokenizer
from keras.preprocessing.sequence import pad_sequences
from pandas import Series
import pickle
Max_Len = 20


def complete(x):
    result = []
    data_zero = np.zeros(128)
    for i in x:
        data_Len = len(i[1])
        for j in i[1].values:
            result.append(j)
        for j in range(Max_Len - data_Len):
            result.append(data_zero)
    return result


data_path = 'dataset'#数据集路径
bert_pre_model = 'bert-base-uncased/pytorch_model.bin'#预训练模型文件
bert_config = 'bert-base-uncased/bert_config.json'#配置文件
bert_pre_tokenizer = 'bert-base-chinese-vocab.txt'#词表

# 读取训练数据 os.path.join(data_dir, "train.txt")
df = pd.read_csv(os.path.join(data_path, "project.csv"), delimiter=',')

# 提取语句并处理
print(df.head())
sentencses = ['[CLS] ' + sent + ' [SEP]' for sent in df.project.values]
labels = df.label.values
print("第一句话:", sentencses[0])
tokenizer = BertTokenizer.from_pretrained(bert_pre_tokenizer, do_lower_case=True)
tokenized_sents = [tokenizer.tokenize(sent) for sent in sentencses]
print("tokenized的第一句话:", tokenized_sents[0])

# 定义句子最大长度（512）
MAX_LEN = 128
# 定义每个人最多可填的honour数量
MAX_Honour = 20

# 将分割后的句子转化成数字  word-->idx
input_ids = [tokenizer.convert_tokens_to_ids(sent) for sent in tokenized_sents]
print("转化后的第一个句子:", input_ids[0])

# 做PADDING,这里使用keras的包做pad，也可以自己手动做pad,truncating表示大于最大长度截断
# 大于128做截断，小于128做PADDING
input_ids = pad_sequences(input_ids, maxlen=MAX_LEN, dtype="long", truncating="post", padding="post")
print("Padding 第一个句子:", input_ids[0])

# 将ndarray类型数据转化为pandas类型
input_ids = list(input_ids)
out = pd.DataFrame()
out["uid"] = df["i"]
out["project"] = input_ids
# print(input_ids)

# 对相同uid的honour数据进行形状拼接
out_group = out.groupby("uid")
x = Series(out_group.project)


f = open("data.pickle", mode='wb')
result = {}
for j in x:
    result[j[0]] = j[1].values

pickle.dump(result, f)
out = out_group.apply()
# out.to_csv("number_project.csv")


# 建立mask
attention_masks = []
for seq in input_ids:
    seq_mask = [float(i > 0) for i in seq]
    attention_masks.append(seq_mask)
print("第一个attention mask:", attention_masks[0])
