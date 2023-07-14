import torch
from torch.autograd import Variable


from pytorch_pretrained_bert import BertTokenizer,  BertModel

import logging

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

text = "[CLS] Who was Jim Henson ? [SEP] Jim Henson was a puppeteer [SEP]"
tokenized_text = tokenizer.tokenize(text)

masked_index = 8
tokenized_text[masked_index] = '[MASK]'
assert tokenized_text == ['[CLS]', 'who', 'was', 'jim', 'henson', '?', '[SEP]', 'jim', '[MASK]', 'was', 'a', 'puppet', '##eer', '[SEP]']

indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)

print(indexed_tokens, len(indexed_tokens))

segments_ids = [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1]

tokens_tensor = torch.tensor([indexed_tokens])
segments_tensors = torch.tensor([segments_ids])

model = BertModel.from_pretrained('bert-base-uncased')
import ipdb
ipdb.set_trace()
model.eval()

tokens_tensor = tokens_tensor.to('cuda')
segments_tensors = segments_tensors.to('cuda')
model.to('cuda')

with torch.no_grad():
    encoded_layers, pooled_output = model(tokens_tensor, segments_tensors)
    print(pooled_output, pooled_output.shape)
