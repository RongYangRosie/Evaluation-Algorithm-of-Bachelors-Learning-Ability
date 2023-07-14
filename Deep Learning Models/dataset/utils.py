import os
import pandas as pd
from fastai import *
from fastai.vision import *
from fastai.text import *
from fastai.callback import *
from pytorch_pretrained_bert import BertTokenizer
from keras.preprocessing.sequence import pad_sequences
from pandas import Series
MAX_LEN = 128


def project2vector(text, bert_pre_tokenizer):
    sentencses = ['[CLS] ' + sent + ' [SEP]' for sent in text.project.values]
    tokenizer = BertTokenizer.from_pretrained(bert_pre_tokenizer, do_lower_case=True)
    tokenized_sents = [tokenizer.tokenize(sent) for sent in sentencses]
    input_ids = [tokenizer.convert_tokens_to_ids(sent) for sent in tokenized_sents]
    input_ids = pad_sequences(input_ids, maxlen=MAX_LEN, dtype="long", truncating="post", padding="post")
    return input_ids


def honour2vector(text, bert_pre_tokenizer):
    sentencses = ['[CLS] ' + sent + ' [SEP]' for sent in text.honour.values]
    tokenizer = BertTokenizer.from_pretrained(bert_pre_tokenizer, do_lower_case=True)
    tokenized_sents = [tokenizer.tokenize(sent) for sent in sentencses]
    input_ids = [tokenizer.convert_tokens_to_ids(sent) for sent in tokenized_sents]
    input_ids = pad_sequences(input_ids, maxlen=MAX_LEN, dtype="long", truncating="post", padding="post")
    return input_ids
