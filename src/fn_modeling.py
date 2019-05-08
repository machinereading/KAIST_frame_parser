from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import copy
import json
import math
import logging
import tarfile
import tempfile
import shutil

import torch
from torch import nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss

import sys
sys.path.insert(0,'../')
sys.path.insert(0,'../../')
from KAIST_frame_parser.src import dataio
from torch.nn.parameter import Parameter

from pytorch_pretrained_bert.modeling import *
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
n_gpu = torch.cuda.device_count()
# torch.cuda.get_device_name(0)

def add_vocab_to_model(model, added_vocab=False):
    if added_vocab == False:
        added_vocab = ['<tgt>', '</tgt>']
    ori_vocab_size = model.config.vocab_size
    vocab_size = ori_vocab_size + len(added_vocab)
    model.config.vocab_size = vocab_size
    ori_embedding = model.embeddings.word_embeddings
    new_embedding = ori_embedding
    new_embedding.num_embeddings = vocab_size
    new_weight = []
    for w in torch.tensor(ori_embedding.weight):
        new_weight.append(w)
    for i in range(len(added_vocab)):
        init_weight = torch.randn(768)
        new_weight.append(init_weight)
    new_weight_tensor = torch.stack(new_weight)
    new_weight_param = Parameter(new_weight_tensor)

    new_embedding.weight = new_weight_param
    model.embeddings.word_embeddings = new_embedding
    return model

class BertForFrameIdentification(BertPreTrainedModel):
    def __init__(self, config, num_labels=2, num_lus=2, ludim=64, lufrmap=None):
        super(BertForFrameIdentification, self).__init__(config)
        self.num_labels = num_labels # total number of all senses
        self.lu_embeddings = nn.Embedding(num_lus, ludim) # embeddings for lexical unit. (e.g. eat.v)
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size+ludim, num_labels)
        self.apply(self.init_bert_weights)
        self.lufrmap = lufrmap # mapping table for lu to its sense candidates

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, tgt_idxs=0, lus=None, frames=None,):
        sequence_output, _ = self.bert(input_ids, token_type_ids, attention_mask, output_all_encoded_layers=False)        
        sequence_output = self.dropout(sequence_output)
        tgt_vec = []
        for i in range(len(sequence_output)):
            tgt_vec.append(sequence_output[i][tgt_idxs[i]])
        tgt_vec = torch.stack(tgt_vec)
        lu_vec = self.lu_embeddings(lus)
        
        tgt_embs = torch.cat((tgt_vec, lu_vec), -1)
        logits = self.classifier(tgt_embs)
        masks = dataio.get_masks(lus, self.lufrmap, num_label=self.num_labels).to(device)
        
        total_loss = 0
        if frames is not None:
            for i in range(len(logits)):
                logit = logits[i]
                mask = masks[i]
                frame = frames[i]
                loss_fct = CrossEntropyLoss(weight = mask)
                loss_per_seq = loss_fct(logit.view(-1, self.num_labels), frame.view(-1))
                total_loss += loss_per_seq
            loss = total_loss / len(logits)
            return loss
        else:
            return logits
        
class BertForFrameId(BertPreTrainedModel):
    def __init__(self, config, num_labels=2, lufrmap=None):
        super(BertForFrameId, self).__init__(config)
        self.num_labels = num_labels # total number of all senses
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, num_labels)
        self.apply(self.init_bert_weights)
        self.lufrmap = lufrmap # mapping table for lu to its sense candidates    

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, lus=None, frames=None,):
        _, pooled_output = self.bert(input_ids, token_type_ids, attention_mask, output_all_encoded_layers=False)        
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        masks = dataio.get_masks(lus, self.lufrmap, num_label=self.num_labels).to(device)
        
        total_loss = 0
        if frames is not None:
            for i in range(len(logits)):
                logit = logits[i]
                mask = masks[i]
                frame = frames[i]
                loss_fct = CrossEntropyLoss(weight = mask)
                loss_per_seq = loss_fct(logit.view(-1, self.num_labels), frame.view(-1))
                total_loss += loss_per_seq
            loss = total_loss / len(logits)
            return loss
        else:
            return logits        
        
class BertForArgClassification(BertPreTrainedModel):
    def __init__(self, config, num_labels=2, num_lus=2, num_frames=2, ludim=64, framedim=100, frargmap=None):
        super(BertForArgClassification, self).__init__(config)
        self.num_labels = num_labels # total number of all FEs
        self.lu_embeddings = nn.Embedding(num_lus, ludim) # embeddings for lexical unit. (e.g. eat.v)
        self.frame_embeddings = nn.Embedding(num_frames, framedim) # embeddings for frame (e.g. Ingesting)
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size+config.hidden_size+ludim+framedim, num_labels)
        self.apply(self.init_bert_weights)
        self.frargmap = frargmap # mapping table for lu to its sense candidates

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, tgt_idxs=0, lus=None, frames=None, arg_idxs=None, args=None):
        sequence_output, _ = self.bert(input_ids, token_type_ids, attention_mask, output_all_encoded_layers=False)        
        sequence_output = self.dropout(sequence_output)
        
        # target and arg vector
        tgt_vec, arg_vec = [],[]
        for i in range(len(sequence_output)):
            tgt_vec.append(sequence_output[i][tgt_idxs[i]])
            arg_vec.append(sequence_output[i][arg_idxs[i]])
        tgt_vec = torch.stack(tgt_vec)
        arg_vec = torch.stack(arg_vec)
        # LU vector
        lu_vec = self.lu_embeddings(lus)
        #frame vector
        frame_vec = self.frame_embeddings(frames)
        # arg_embs
        arg_embs = torch.cat((arg_vec, tgt_vec, lu_vec, frame_vec), -1)
        
        logits = self.classifier(arg_embs)
        masks = dataio.get_masks(frames, self.frargmap, num_label=self.num_labels).to(device)
        
        total_loss = 0
        if args is not None:
            for i in range(len(logits)):
                logit = logits[i]
                mask = masks[i]
                arg = args[i]
                loss_fct = CrossEntropyLoss(weight = mask)
                loss_per_seq = loss_fct(logit.view(-1, self.num_labels), arg.view(-1))
                total_loss += loss_per_seq
            loss = total_loss / len(logits)
            return loss
        else:
            return logits