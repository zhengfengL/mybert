import math
import os
import warnings
from dataclasses import dataclass
from typing import Optional, Tuple
from collections import OrderedDict

import torch
from torch import nn
from torch.nn import CrossEntropyLoss, MSELoss, Parameter    
from torch.utils.data import RandomSampler, SequentialSampler, DataLoader, TensorDataset 
from torch.optim import Adam
from sklearn.model_selection import train_test_split

import numpy as np
import pandas as pd
import my_bert_model

from transformers import BertTokenizer, BertConfig
from my_bert_model import BertModel, BertForSequenceClassification

MAX_LEN = 512
BATCH_SIZE = 8
EPOCHS = 5
COLA_DIR = "./CoLA-archive/cola_public/raw/"
VOCAB_DIR = ""
DEVICE = 'cuda:6'

if __name__ == "__main__":
    #print(torch.cuda.is_available())
    device = torch.device(DEVICE if torch.cuda.is_available() else 'cpu')
    '''
    test CoLa
    '''
    '''Load dataset and construct train, dev and test sets'''
    df = pd.read_csv(os.path.join(COLA_DIR, "in_domain_train.tsv"), delimiter='\t', header=None, \
        names=['sentence_source', 'label', 'label_notes', 'sentence'])
    df = df[:2000]

    '''We just need sentences and labels'''

    sentences = ['[CLS] '+ sent + ' [SEP]' for sent in df.sentence.values]
    labels = df.label.values
    #print(sentences[0], labels[0], len(sentences), len(labels))

    '''Tokenization'''
    print('Tokenize.')
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    tok_sentences = [tokenizer.tokenize(sent) for sent in sentences]  
    #print(tok_sentences[0], labels[0], len(sentences), len(labels))
    input_ids = [tokenizer.convert_tokens_to_ids(sent) for sent in tok_sentences]
    #print(input_ids[20])
    #print(tokenizer.convert_ids_to_tokens(input_ids[20]))

    '''Pad to the max length or truncate and Build the mask'''
    print('Pad the inputs and construct the mask.')
    pad_input_id = []
    attention_masks = []
    for input_id in input_ids:
        if len(input_id) >= MAX_LEN:
            input_id = input_id[:MAX_LEN]
            seq_mask = np.ones([MAX_LEN, MAX_LEN])
        else:
            input_id = input_id + [0 for i in range(MAX_LEN - len(input_id))]
            seq_mask = [[float(i>0) and float(j>0) for i in input_id] for j in input_id]
            #seq_mask = [float(i>0) for i in input_id]
        pad_input_id.append(input_id)
        attention_masks.append(seq_mask)
    input_ids = pad_input_id
    # print(input_ids[0], len(input_ids[0]))
    # print(attention_masks[0])
    

    '''Split the set and turn them into tensors'''
    print('Split the set.')
    train_inputs, validation_inputs, train_labels, validation_labels = train_test_split(input_ids, labels, random_state=2020, test_size=0.1)
    train_masks, validation_masks, _, _ = train_test_split(attention_masks, input_ids, random_state=2020, test_size=0.1)
    print('Turn data into tensors.')
    #print(tokenizer.convert_ids_to_tokens(train_inputs[22]), train_labels[22])
    train_inputs = torch.tensor(train_inputs)
    validation_inputs = torch.tensor(validation_inputs)
    train_labels = torch.tensor(train_labels)
    validation_labels = torch.tensor(validation_labels)
    train_masks = torch.tensor(train_masks)
    validation_masks = torch.tensor(validation_masks)
    #print(train_inputs[22], train_labels[22])
    #print(train_inputs[0].dtype)

    '''Construct the dataloader'''
    train_data = TensorDataset(train_inputs, train_masks, train_labels)
    train_sampler =  RandomSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=BATCH_SIZE)
    validation_data = TensorDataset(validation_inputs, validation_masks, validation_labels)
    validation_sampler = SequentialSampler(validation_data)
    validation_dataloader = DataLoader(validation_data, sampler=validation_sampler, batch_size=BATCH_SIZE)

    '''load parameters and create optimizer'''
    def load_my_state_dict(model, state_dict):
        own_state = model.state_dict()
        for name, param in state_dict.items():
            if name not in own_state:
                continue
            if isinstance(param, Parameter):
                # backwards compatibility for serialized parameters
                param = param.data
            own_state[name].copy_(param)

    model_config = BertConfig.from_pretrained('bert-base-uncased')
    model = BertForSequenceClassification(model_config, num_labels=2).to(device)
    param_dict = torch.load('bert_data/bert-base-uncased-pytorch_model.bin')
    param_dict = OrderedDict([k.replace('LayerNorm.gamma', 'LayerNorm.weight'), v] for k, v in param_dict.items())
    param_dict = OrderedDict([k.replace('LayerNorm.beta', 'LayerNorm.bias'), v] for k, v in param_dict.items())
    load_my_state_dict(model, param_dict)
    keys = model.state_dict()
    print(type(keys))
    #for key in keys:
    #    print(keys[key], key, '\n')
    #print(model.named_parameters())
    #print(list(model.named_parameters()))

    param_optimizer = list(model.named_parameters())
    # for key in model.named_parameters():
    #    print(key[0], '\n')
    optimizer = Adam(model.classifier.parameters(), lr=1e-5) 

    '''fine tuning'''
    def accuracy(preds, labels):
        pred_flat = np.argmax(preds, axis=1).flatten()
        labels_flat = labels.flatten()
        print('pred_flat: ', pred_flat)
        print('labels_flat:', labels_flat)
        return np.sum(pred_flat == labels_flat) / len(labels_flat)

    epochs = EPOCHS
    for epoch in range(epochs):
        print('EPOCH ', epoch, ' begins.')
        model.train()
        tr_loss = 0
        nb_tr_examples, nb_tr_steps = 0, 0
        for step, batch in enumerate(train_dataloader):
            batch = tuple(t.to('cpu') for t in batch) 
            b_input_ids, b_input_mask, b_labels = batch
            #if step==1:
            #    print(b_input_ids)
            #    print(b_input_mask)
            optimizer.zero_grad()

            loss = model(
                b_input_ids.to(device), 
                token_type_ids=None, 
                attention_mask=b_input_mask.to(device), 
                labels=b_labels.to(device)
                )[0]
            loss.backward()
            optimizer.step()

            tr_loss += loss.item()
            nb_tr_examples += b_input_ids.size(0)
            nb_tr_steps += 1
        print("Train loss: {}".format(tr_loss / nb_tr_steps))

        model.eval()
        eval_loss, eval_accuracy = 0, 0
        nb_eval_examples, nb_eval_steps = 0, 0
        #for batch in validation_dataloader:
        for batch in train_dataloader:
            batch = tuple(t.to('cpu') for t in batch)
            b_input_ids, b_input_mask, b_labels = batch
            with torch.no_grad():
                probs = model(
                    b_input_ids.to(device), 
                    token_type_ids=None, 
                    attention_mask=b_input_mask.to(device)
                    )[1]
            probs = probs.cpu().numpy()
            print('probs: ', probs)
            label_ids = b_labels.to('cpu').numpy()
            tmp_eval_accuracy = accuracy(probs, label_ids)
            eval_accuracy += tmp_eval_accuracy
            nb_eval_steps += 1
        print("Train loss: {}".format(tr_loss / nb_tr_steps))
        print("Validation Accuracy: {}".format(eval_accuracy / nb_eval_steps))


    keys = model.state_dict()
    print(type(keys))
    #for key in keys:
    #    print(keys[key], key, '\n')
    


    