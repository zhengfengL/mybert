import math
import os
import warnings
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import CrossEntropyLoss, MSELoss

import numpy as np
import pandas as pd

from transformers import BertTokenizer

DEVICE = 'cuda:6'

class BertConfig():
    r"""
    Args:
        vocab_size (:obj:`int`, `optional`, defaults to 30522):
            Vocabulary size of the BERT model. Defines the number of different tokens that can be represented by the
            :obj:`inputs_ids` passed when calling :class:`~transformers.BertModel` or
            :class:`~transformers.TFBertModel`.
        hidden_size (:obj:`int`, `optional`, defaults to 768):
            Dimensionality of the encoder layers and the pooler layer.
        num_hidden_layers (:obj:`int`, `optional`, defaults to 12):
            Number of hidden layers in the Transformer encoder.
        num_attention_heads (:obj:`int`, `optional`, defaults to 12):
            Number of attention heads for each attention layer in the Transformer encoder.
        intermediate_size (:obj:`int`, `optional`, defaults to 3072):
            Dimensionality of the "intermediate" (often named feed-forward) layer in the Transformer encoder.
        hidden_dropout_prob (:obj:`float`, `optional`, defaults to 0.1):
            The dropout probabilitiy for all fully connected layers in the embeddings, encoder, and pooler.
        attention_probs_dropout_prob (:obj:`float`, `optional`, defaults to 0.1):
            The dropout ratio for the attention probabilities.
        max_position_embeddings (:obj:`int`, `optional`, defaults to 512):
            The maximum sequence length that this model might ever be used with.
            Typically set this to something large just in case (e.g., 512 or 1024 or 2048).
        type_vocab_size (:obj:`int`, `optional`, defaults to 2):
            The vocabulary size of the :obj:`token_type_ids` passed when calling :class:`~transformers.BertModel` or
            :class:`~transformers.TFBertModel`.
        layer_norm_eps (:obj:`float`, `optional`, defaults to 1e-12):
            The epsilon used by the layer normalization layers.
        is_encoder_decoder (:obj:`bool`, `optional`, defaults to :obj:`False`):
            Whether the model is used as an encoder/decoder or not.
        is_decoder (:obj:`bool`, `optional`, defaults to :obj:`False`):
            Whether the model is used as decoder or not (in which case it's used as an encoder).
        add_cross_attention (:obj:`bool`, `optional`, defaults to :obj:`False`):
            Whether cross-attention layers should be added to the model. Note, this option is only relevant for models
            that can be used as decoder models within the `:class:~transformers.EncoderDecoderModel` class, which
            consists of all models in ``AUTO_MODELS_FOR_CAUSAL_LM``.
    Examples:
        >>> from transformers import BertModel, BertConfig
        >>> # Initializing a BERT bert-base-uncased style configuration
        >>> configuration = BertConfig()
        >>> # Initializing a model from the bert-base-uncased style configuration
        >>> model = BertModel(configuration)
        >>> # Accessing the model configuration
        >>> configuration = model.config
    """
    model_type = "bert"

    def __init__(
        # configuration from the original paper
        self,
        # vababulary size of words
        vocab_size=30522,
        # number of hidden output's dimensions 
        hidden_size=768,
        # number of hidden layers
        num_hidden_layers=12,
        # type of activation function, GELU from the original paper 
        hidden_act = 'gelu',
        # number of heads of multihead attention
        num_attention_heads=12,
        # number of dimensions of intermediate vectors in FFN
        intermediate_size=3072,
        # dropout probabilitiy for all fully connected layers in the embeddings, encoder, and pooler
        hidden_dropout_prob=0.1,
        # dropout ratio for the attention probabilities
        attention_probs_dropout_prob=0.1,
        # maximum sequence length that this model might ever be used with
        max_position_embeddings=512,
        # vocabulary size of types, 2 in this case (for sentence pairs 0, 1
        # )
        type_vocab_size=2,
        # epsilon used by the layer normalization layers
        layer_norm_eps=1e-12,
        # The id of token [PAD] in the vocabulary
        pad_token_id=0,
        # The model may be used as a decoder in seq2seq tasks
        is_decoder=False,

        add_cross_attention=False,
        **kwargs
    ):

        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.hidden_act = hidden_act
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.max_position_embeddings = max_position_embeddings
        self.type_vocab_size = type_vocab_size
        self.layer_norm_eps = layer_norm_eps
        self.pad_token_id = pad_token_id
        self.is_decoder = is_decoder
        self.add_cross_attention = add_cross_attention



class BertEmbeddings(nn.Module):
    r'''
    Construct the embeddings from word, position and segment() embeddings
    '''
    def __init__(self, config):
        super(BertEmbeddings, self).__init__()
        # construct the word embeddings
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        # construct the position embeddings
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        # construct the segment embeddings
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)
        # layer normalization and dropout
        self.LayerNorm = nn.LayerNorm(config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        # position_ids (1, len position emb) is contiguous in memory and exported when serialized
        self.register_buffer("position_ids", torch.arange(config.max_position_embeddings).expand((1, -1)))

    def forward(self, input_ids=None, token_type_ids=None, position_ids=None, inputs_embeds=None):
        r'''
        input_ids: [batch_size X max_position_embeddings]
        token_type_ids: [batch_size X max_position_embeddings]
        position_ids: [batch_size X max_position_embeddings]
        inputs_embeds: [batch_size X max_position_embeddings X hidden_size]
        '''
        # handle inputs
        if input_ids is not None:
            # [batch_size X max_position_embeddings]
            input_shape = input_ids.size()
        else:
            # [batch_size X max_position_embeddings]
            input_shape = inputs_embeds.size()[:-1]

        seq_length = input_shape[1]

        # we truncate the embeddings if not having position_ids
        if position_ids is None:
            #print(self.position_ids)
            position_ids = self.position_ids[:, :seq_length]

        # we only have one sentence per example in this case
        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=self.position_ids.device)

        # construct the embeddings 
        if inputs_embeds is None:
            word_embeddings = self.word_embeddings(input_ids)
        else: 
            word_embeddings = inputs_embeds
        position_embeddings = self.position_embeddings(position_ids)
        segment_embeddings = self.token_type_embeddings(token_type_ids)

        # compute embeddings by summing all embeddings
        embeddings = word_embeddings + position_embeddings + segment_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class BertSelfAttention(nn.Module):
    def __init__(self, config):
        super(BertSelfAttention, self).__init__()
        # num_attention_heads must divides hidden_size
        assert config.hidden_size % config.num_attention_heads == 0

        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = config.hidden_size // config.num_attention_heads
        # Exactly hidden_size in our usage
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        # structure
        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)
        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(
        self, 
        # hidden_states from embeddings or last layer
        hidden_states,
        # mask to avoid performing attention on the padding token indices of the encoder input
        attention_mask=None,
        # encoder hidden states if we apply this structure in a decoder
        encoder_hidden_states=None,
        # mask to avoid performing attention on the padding token indices of the encoder input
        encoder_attention_mask=None,
    ):
        mixed_query_layer = self.query(hidden_states)

        # Not self attention, decoder
        if encoder_hidden_states is not None:
            mixed_key_layer = self.key(encoder_hidden_states)
            mixed_value_layer = self.value(encoder_hidden_states)
            attention_mask = encoder_attention_mask
        # Self attention, encoder
        else:
            mixed_key_layer = self.key(hidden_states)
            mixed_value_layer = self.value(hidden_states)

        # Shape: [batch_size X num_attention_heads X max_position_embeddings X hidden_size]
        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        # print(query_layer.size())
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / np.sqrt(self.attention_head_size)
        # pad masking
        if attention_mask is not None:
            # print(attention_scores.size(), attention_mask.size())
            # adjust the attention mask's size to attention score's size
            attention_mask = attention_mask.unsqueeze(1).expand(-1, self.num_attention_heads, -1, -1)
            attention_scores = attention_scores.masked_fill(attention_mask == 0, -1e9)
        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        # From the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)

        # Shape: [batch_size X max_position_embeddings X all_head_size(hidden_size)]
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)

        outputs = context_layer
        return outputs


class BertSelfOutput(nn.Module):
    def __init__(self, config):
        super(BertSelfOutput, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(
        self, 
        # hidden_states after attention computing
        hidden_states, 
        # hidden_states from embeddings or last layer
        input_states
    ):

        hidden_states = self.dropout(self.dense(hidden_states))
        # add and normalization
        hidden_states = self.LayerNorm(hidden_states + input_states)
        return hidden_states


class BertAttention(nn.Module):
    r'''
    Multi-head attention + Add & Norm.
    '''
    def __init__(self, config):
        super(BertAttention, self).__init__()
        self.self = BertSelfAttention(config)
        self.output = BertSelfOutput(config)

    def forward(
        self, 
        # hidden_states from embeddings or last layer
        hidden_states,
        # mask to avoid performing attention on the padding token indices of the encoder input
        attention_mask=None,
        # encoder hidden states if we apply this structure in a decoder
        encoder_hidden_states=None,
        # mask to avoid performing attention on the padding token indices of the encoder input
        encoder_attention_mask=None,
    ):
        attention_output = self.self(
            hidden_states,
            attention_mask,
            encoder_hidden_states,
            encoder_attention_mask,
        )
        # res net
        attention_output = self.output(attention_output, hidden_states)
        output = attention_output
        return output


class BertIntermediate(nn.Module):
    r'''
    Feed forward network layer 1.
    '''
    def __init__(self, config):
        super(BertIntermediate, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        self.act_fn = nn.GELU()

    def forward(self, hidden_states):
        hidden_states = self.act_fn(self.dense(hidden_states))
        return hidden_states


class BertOutput(nn.Module):
    r'''
    Feed forward network layer 2 + Add & Norm
    '''
    def __init__(self, config):
        super(BertOutput, self).__init__()
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(
        self, 
        # hidden_states after attention computing
        hidden_states, 
        # hidden_states from embeddings or last layer
        input_states
    ):
        hidden_states = self.dropout(self.dense(hidden_states))
        hidden_states = self.LayerNorm(hidden_states + input_states)
        return hidden_states
        


class BertLayer(nn.Module):
    def __init__(self, config):
        super(BertLayer, self).__init__()
        # Attention + Add & Norm + Feed Forward Network + Add & Norm
        self.attention = BertAttention(config)
        self.intermediate = BertIntermediate(config)
        self.output = BertOutput(config)
        
        
        self.is_decoder = config.is_decoder
        self.add_cross_attention = config.add_cross_attention
        if self.add_cross_attention:
            assert self.is_decoder
            self.crossattention = BertAttention(config)
        

    def forward(
        self, 
        # hidden_states from embeddings or last layer
        hidden_states,
        # mask to avoid performing attention on the padding token indices of the encoder input
        attention_mask=None,
        # encoder hidden states if we apply this structure in a decoder
        encoder_hidden_states=None,
        # mask to avoid performing attention on the padding token indices of the encoder input
        encoder_attention_mask=None,
    ):
        self_attention_output = self.attention(
            hidden_states,
            attention_mask,
        )
        attention_output = self_attention_output

        # Need to polish
        if self.is_decoder and encoder_hidden_states is not None:
            cross_attention_output = self.crossattention(
                attention_output,
                attention_mask,
                encoder_hidden_states,
                encoder_attention_mask,
            )
            attention_output = cross_attention_output

        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        # just return layer output: [batch_size X max_position_embeddings X hidden_size]
        return layer_output
        


class BertEncoder(nn.Module):
    def __init__(self, config):
        super(BertEncoder, self).__init__()
        self.config = config
        # many bert layers
        self.layer = nn.ModuleList([BertLayer(config) for i in range(self.config.num_hidden_layers)]) 
    def forward(
        self,
        # hidden_states from embeddings or last layer
        hidden_states,
        # mask to avoid performing attention on the padding token indices of the encoder input
        attention_mask=None,
        # encoder hidden states if we apply this structure in a decoder
        encoder_hidden_states=None,
        # mask to avoid performing attention on the padding token indices of the encoder input
        encoder_attention_mask=None, 
    ):
        all_hidden_states = [hidden_states]
        for i, layer_module in enumerate(self.layer):
            if encoder_hidden_states is not None:
                layer_output = layer_module(
                    hidden_states,
                    attention_mask,
                    encoder_hidden_states[i],
                    encoder_attention_mask,
                )
            else:
                layer_output = layer_module(
                    hidden_states,
                    attention_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                )
            hidden_states = layer_output
            all_hidden_states.append(hidden_states)
        
        last_hidden_states = hidden_states
        return (last_hidden_states, all_hidden_states)
        



class BertPooler(nn.Module):
    r'''
    BERT pooler to handle the first token.
    According to the BERT paper, we just take the hidden state correspondind 
    to the first token ([CLS]). The pooler may be applied in several tasks, 
    like sequence classification.
    '''
    def __init__(self, config):
        super(BertPooler, self).__init__()
        # dense + Tanh
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states):
        # extract the first token's embedding ([CLS]) 
        first_token = hidden_states[:, 0]
        pooled_output = self.activation(self.dense(first_token))
        return pooled_output



class BertModel(nn.Module):
    r'''
    The whole BERT model like the original paper
    '''
    def __init__(self, config, add_pooling_layer=True):
        # embeddings + encoder + pooler (optional)
        super(BertModel, self).__init__()
        self.config = config
        self.embeddings = BertEmbeddings(config)
        self.encoder = BertEncoder(config)
        self.pooler = BertPooler(config) if add_pooling_layer else None

    def get_input_embeddings(self):
        # retrieve the word embeddings
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        # add new embeddings to model
        self.embeddings.word_embeddings = value

    def forward(
        self,
        # input indexes after tokenization
        input_ids=None,
        # mask to avoid performing attention on the padding token indices of the encoder input
        attention_mask=None,
        # token type ids, if None then it would be handled as zero vector
        token_type_ids=None,
        # position ids
        position_ids=None,
        # input embeds
        inputs_embeds=None,
        # encoder hidden states if we apply this structure in a decoder
        encoder_hidden_states=None,
        # mask to avoid performing attention on the padding token indices of the encoder input
        encoder_attention_mask=None,
    ):

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        if attention_mask is None:
            attention_mask = torch.ones(input_shape).to(DEVICE)
        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long).to(DEVICE)
        
        # Computing
        embedding_output = self.embeddings(
            input_ids=input_ids,
            position_ids=position_ids,
            token_type_ids=token_type_ids,
            inputs_embeds=inputs_embeds,
        )
        encoder_output, all_hidden_states = self.encoder(
            embedding_output,
            attention_mask,
            encoder_hidden_states,
            encoder_attention_mask,
        )

        sequence_output = encoder_output
        pooled_output = self.pooler(encoder_output) if self.pooler is not None else None

        return (sequence_output, pooled_output, all_hidden_states)



class BertForSequenceClassification(nn.Module):
    r'''
    BERT model for sequence classification.
    '''
    def __init__(self, config, num_labels=2):
        super(BertForSequenceClassification, self).__init__()
        self.num_labels = num_labels

        # BERT model + dropout + dense
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, num_labels)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        inputs_embeds=None,
        labels=None,
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for computing the sequence classification/regression loss.
            Indices should be in :obj:`[0, ..., num_labels - 1]`.
            If :obj:`config.num_labels == 1` a regression loss is computed (Mean-Square loss),
            If :obj:`config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """

        outputs = self.bert(
            input_ids = input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
        )

        all_hidden_states = outputs[2]
        pooled_output = self.dropout(outputs[1])
        logits = self.classifier(pooled_output)
        probs = nn.Softmax(dim=-1)(logits)

        # Computing loss
        loss = None
        if labels is not None:
            if self.num_labels == 1:
                loss_fct = MSELoss()
                loss = loss_fct(probs.view(-1), labels.view(-1))
            else:
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(probs.view(-1, self.num_labels), labels.view(-1))
        
        # Return loss and logits
        return (loss, probs, all_hidden_states)


'''
Part for seq2seq or text generation
'''

class BertGenerationEncoder(nn.Module):
    def __init__(self, config):
        super(BertGenerationEncoder, self).__init__()

        self.embeddings = BertEmbeddings(config)
        self.bert = BertEncoder(config)

    def forward(
        self,
        # input indexes after tokenization
        input_ids=None,
        # mask to avoid performing attention on the padding token indices of the encoder input
        attention_mask=None,
        # token type ids, if None then it would be handled as zero vector
        token_type_ids=None,
        # position ids
        position_ids=None,
        # input embeds
        inputs_embeds=None,
    ):
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        if attention_mask is None:
            attention_mask = torch.ones(input_shape).to(DEVICE)
        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long).to(DEVICE)
        
        # Computing
        embedding_output = self.embeddings(
            input_ids=input_ids,
            position_ids=position_ids,
            token_type_ids=token_type_ids,
            inputs_embeds=inputs_embeds,
        )
        encoder_output, all_hidden_states = self.bert(
            hidden_states=embedding_output,
            attention_mask=attention_mask,
        )

        sequence_output = encoder_output

        return (sequence_output, all_hidden_states)



class BertGenerationDecoder(nn.Module):
    def __init__(self, config):
        super(BertGenerationDecoder, self).__init__()

        if not config.is_decoder:
            raise ValueError("The config.is_decoder in BertGenerationDecoder must be true.")
        self.embeddings = BertEmbeddings(config)
        self.bert = BertEncoder(config)
        self.predictor = nn.Linear(config.hidden_size, config.vocab_size)

    def forward(
        self,
        # input indexes after tokenization
        input_ids=None,
        # mask to avoid performing attention on the padding token indices of the encoder input
        attention_mask=None,
        # token type ids, if None then it would be handled as zero vector
        token_type_ids=None,
        # position ids
        position_ids=None,
        # input embeds
        inputs_embeds=None,
        # encoder hidden states if we apply this structure in a decoder
        encoder_hidden_states=None,
        # mask to avoid performing attention on the padding token indices of the encoder input
        encoder_attention_mask=None,
        # labels to compute logits and loss
        labels=None,
    ):
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        if attention_mask is None:
            attention_mask = torch.ones(input_shape).to(DEVICE)
        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long).to(DEVICE)
        
        # Computing
        embedding_output = self.embeddings(
            input_ids=input_ids,
            position_ids=position_ids,
            token_type_ids=token_type_ids,
            inputs_embeds=inputs_embeds,
        )
        decoder_output, all_hidden_states = self.bert(
            embedding_output=embedding_output,
            attention_mask=attention_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
        )

        sequence_output = decoder_output
        logits = self.predictor(sequence_output)

        ### loss and labels?
        loss = 0
        '''
        if labels is not None:
            # we are doing next-token prediction; shift prediction scores and input ids by one
            shifted_prediction_scores = logits[:, :-1, :].contiguous()
            labels = labels[:, 1:].contiguous()
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(shifted_prediction_scores.view(-1, self.config.vocab_size), labels.view(-1))
        '''

        return (loss, logits, sequence_output, all_hidden_states)



class BertGenerationEncoderDecoder(nn.Module):
    def __init__(
        self,
        encoder,
        decoder,
    ):
        super(BertGenerationEncoderDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(
        self,
        # input indexes after tokenization
        input_ids=None,
        # mask to avoid performing attention on the padding token indices of the encoder input
        attention_mask=None,
        # input indexes after tokenization for decoder
        decoder_input_ids=None,
        # mask to avoid performing attention on the padding token indices of the decoder input
        decoder_attention_mask=None,
        # input embeds
        inputs_embeds=None,
        # input embeds for decoder
        decoder_input_embeds=None,
        # The labels for computing loss
        labels=None,
    ):
        _, encoder_all_states = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
        )

        outputs = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            inputs_embeds=decoder_input_embeds,
            encoder_hidden_states=encoder_all_states,
            encoder_attention_mask=attention_mask,
            labels=labels,
        )

        
        return outputs



'''
class ScaledDotProductAttention(nn.Module):
    def __init__(self, config):
        super(ScaledDotProductAttention, self).__init__()
        self.config = config

    def forward(self, Q, K, V, attn_mask=None):
        d_k = Q.size(-1)
        print(d_k)
        scores = torch.matmul(Q, K.transpose(-1, -2))/np.sqrt(d_k)
        if attn_mask is not None:
            scores.masked_fill_(attn_mask, 1e-9)
        attn = nn.Softmax(dim=-1)(scores)
        attn = nn.Dropout(p=self.config.attention_probs_dropout_prob)(attn)
        context = torch.matmul(attn, V)
        return context, attn
'''

'''
class MultiHeadAttention(nn.Module):
    def __init__(self, config):
        super(MultiHeadAttention, self).__init__()
        self.config = config
        assert self.config.hidden_size % self.config.num_attention_heads == 0
        self.d_k = self.config.hidden_size // self.config.num_attention_heads
        self.linears = nn.ModuleList([nn.Linear(self.config.hidden_size, self.config.hidden_size) for i in range(4)])
        self.attn = ScaledDotProductAttention(self.config)

    def forward(self, Q, K, V, attn_mask=None):
        if attn_mask is not None:
            attn_mask = attn_mask.unsqueeze(1)
        n_batch = Q.size(0)

        # 1) Do all the linear projections in batch from d_model => h x d_k 
        Q, K, V = \
            [l(x).view(n_batch, -1, self.config.num_attention_heads, self.d_k).transpose(1, 2)
             for l, x in zip(self.linears, (Q, K, V))]
        
        # 2) Apply attention on all the projected vectors in batch. 
        x, attn = self.attn(Q, K, V, attn_mask=attn_mask)
        
        # 3) "Concat" using a view and apply a final linear. 
        x = x.transpose(1, 2).contiguous() \
             .view(n_batch, -1, self.config.hidden_size)
        return self.linears[-1](x)
'''

'''
class FeedForwardNetworks(nn.Module):
    def __init__(self, config):
        super(FeedForwardNetworks, self).__init__()
        self.config = config
        self.net_1 = nn.Linear(self.config.hidden_size, self.config.intermediate_size)
        self.net_2 = nn.Linear(self.config.intermediate_size, self.config.hidden_size)
        self.act = nn.GELU()

    def forward(self, x):
        return self.net_2(self.act(self.net_1(x)))
'''

'''
class BertLayer(nn.Module):

    def __init__(self, config):
        super(BertLayer, self).__init__()
        self.config = config
        self.attn = MultiHeadAttention(config)
        self.ff = FeedForwardNetworks(config)
        self.lm1 = nn.LayerNorm(self.config.hidden_size)
        self.lm2 = nn.LayerNorm(self.config.hidden_size)

    def forward(self, x, attn_mask=None):
        residual1 = x
        x = self.attn(x, x, x, attn_mask=attn_mask)
        x = self.lm1(x+residual1)
        residual2 = x
        x = self.ff(x)
        x = self.lm2(x+residual2)
        return x
'''

'''
class BertModel(nn.Module):
    def __init__(self, config):
        super(BertModel, self).__init__()
        self.embedding = BertEmbeddings(config)
        self.layers = nn.ModuleList([BertLayer(config) for i in range(self.config.num_hidden_layers)]) 
        self  

    def forward(self, input_ids, token_type_ids=None, attention_mask=None):
        return
'''