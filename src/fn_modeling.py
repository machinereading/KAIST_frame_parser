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
from src import dataio
from torch.nn.parameter import Parameter

from pytorch_pretrained_bert.modeling import *
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
n_gpu = torch.cuda.device_count()
torch.cuda.get_device_name(0)

class BertForFrameIdentification(BertPreTrainedModel):
    def __init__(self, config, num_labels=2, num_lus=2, ludim=64, lusensemap=None):
        super(BertForFrameIdentification, self).__init__(config)
        self.num_labels = num_labels # total number of all senses
        self.lu_embeddings = nn.Embedding(num_lus, ludim) # embeddings for lexical unit. (e.g. eat.v)
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size+ludim, num_labels)
        self.apply(self.init_bert_weights)
        self.lusensemap = lusensemap # mapping table for lu to its sense candidates

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, tgt_idxs=0, lus=None, senses=None,):
        sequence_output, _ = self.bert(input_ids, token_type_ids, attention_mask, output_all_encoded_layers=False)        
        sequence_output = self.dropout(sequence_output)
        tgt_vec = []
        for i in range(len(sequence_output)):
            tgt_vec.append(sequence_output[i][tgt_idxs[i]])
        tgt_vec = torch.stack(tgt_vec)
        lu_vec = self.lu_embeddings(lus)
        
        tgt_embs = torch.cat((tgt_vec, lu_vec), -1)
        logits = self.classifier(tgt_embs)
        masks = dataio.get_masks(lus, self.lusensemap, num_label=self.num_labels).to(device)
        
        total_loss = 0
        if senses is not None:
            for i in range(len(logits)):
                logit = logits[i]
                mask = masks[i]
                sense = senses[i]
                loss_fct = CrossEntropyLoss(weight = mask)
                loss_per_seq = loss_fct(logit.view(-1, self.num_labels), sense.view(-1))
                total_loss += loss_per_seq
            loss = total_loss / len(logits)
            return loss
        else:
            return logits