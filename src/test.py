import torch
import numpy as np
from collections import OrderedDict
from transformers import BertForSequenceClassification
from my_bert_model import BertModel

model = torch.load('bert_data/bert-base-uncased-pytorch_model.bin')
#print(model.keys())
print(type(model))
n_para = 0
for key in model.keys():
    # print(model[key].size(), key, '\n')
    n_para += np.array(list(model[key].size())).prod()

model2 = OrderedDict([k.replace('LayerNorm.gamma', 'LayerNorm.weight'), v] for k, v in model.items())
model = OrderedDict([k.replace('LayerNorm.beta', 'LayerNorm.bias'), v] for k, v in model2.items())
for key in model.keys():
    print(model[key].size(), key, '\n')

print(model['bert.embeddings.token_type_embeddings.weight'])
print(n_para-2-2*768-30522*768-768-768-768-768*768-30522)