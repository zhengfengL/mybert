import torch
import numpy as np
from torch import nn
from my_bert_model import BertConfig, BertModel

config = BertConfig()
model = BertModel(config)
