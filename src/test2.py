import torch
import numpy as np
from collections import OrderedDict
from my_bert_model import BertConfig, BertModel, BertGenerationEncoderDecoder, BertGenerationEncoder, BertGenerationDecoder
from torch.nn import Parameter

DEVICE = 'cpu'

def load_my_state_dict(model, state_dict):
    own_state = model.state_dict()
    for name, param in state_dict.items():
        if name not in own_state:
            continue
        if isinstance(param, Parameter):
            # backwards compatibility for serialized parameters
            param = param.data
        own_state[name].copy_(param)


config = BertConfig()
model_encoder = BertGenerationEncoder(config)
config.is_decoder = True
model_decoder = BertGenerationDecoder(config)
my_model = BertGenerationEncoderDecoder(model_encoder, model_decoder)
keys = my_model.state_dict()

'''
print(type(keys))
for key in keys:
    #print(keys[key], key, '\n')
    print(key, '\n')
'''

input_ids = torch.tensor([[1, 1, 0, 3, 7], [2, 4, 6, 8, 10]]).to(DEVICE)
outputs = my_model(input_ids=input_ids, decoder_input_ids=input_ids)


'''
config = BertConfig()
config.is_decoder = True
model = torch.load('bert_data/bert-base-uncased-pytorch_model.bin')
model2 = OrderedDict([k.replace('LayerNorm.gamma', 'LayerNorm.weight'), v] for k, v in model.items())
model = OrderedDict([k.replace('LayerNorm.beta', 'LayerNorm.bias'), v] for k, v in model2.items())
my_model = BertForSequenceClassification(config, num_labels=2)
keys = my_model.state_dict()
print(type(keys))
for key in keys:
    #print(keys[key], key, '\n')
    print(key, '\n')

#load_my_state_dict(my_model, state_dict=model)
#keys = my_model.state_dict()
#print(type(keys))
#for key in keys:
    #print(keys[key], key, '\n')
    #print(key[0], '\n')
'''