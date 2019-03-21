
# coding: utf-8

# In[7]:


import torch
import json
import numpy as np
import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from pytorch_pretrained_bert import BertTokenizer, BertConfig, BertModel
from pytorch_pretrained_bert import BertForTokenClassification, BertAdam
from tqdm import tqdm, trange
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
n_gpu = torch.cuda.device_count()

import sys
sys.path.insert(0,'../')

from src import dataio
from src.fn_modeling import BertForFrameIdentification


# In[2]:


MAX_LEN = 256
batch_size = 8


# In[3]:


# load pretrained BERT tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased', do_lower_case=False)

# bert tokenizer
def bert_tokenizer(text):
    orig_tokens = text.split(' ')
    bert_tokens = []
    orig_to_tok_map = []
    bert_tokens.append("[CLS]")
    for orig_token in orig_tokens:
        orig_to_tok_map.append(len(bert_tokens))
        bert_tokens.extend(tokenizer.tokenize(orig_token))
    bert_tokens.append("[SEP]")
    
    return orig_tokens, bert_tokens, orig_to_tok_map


# In[4]:


def frame_identifier(tgt_inputs):
    bert_inputs = dataio.convert_to_bert_input_frameid(tgt_inputs)
    
    data_inputs, data_tgt_idx, data_lus, data_senses, data_masks = bert_inputs[0],bert_inputs[1],bert_inputs[2],bert_inputs[3],bert_inputs[4]
    input_data = TensorDataset(data_inputs, data_tgt_idx, data_lus, data_senses, data_masks)
#     trn_sampler = RandomSampler(trn_data)
    trn_dataloader = DataLoader(trn_data, sampler=None, batch_size=batch_size)
    return trn_dataloader

