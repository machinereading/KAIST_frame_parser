import torch
import sys
import json
sys.path.insert(0,'../')
from src import models
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from keras.preprocessing.sequence import pad_sequences
from pytorch_pretrained_bert import BertTokenizer, BertConfig, BertModel
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
n_gpu = torch.cuda.device_count()

import os
dir_path = os.path.dirname( os.path.abspath( __file__ ))

MAX_LEN = 256
batch_size = 8

def get_masks(datas, mapdata, num_label=2):
    masks = []
    for idx in datas:
        mask = torch.zeros(num_label)
        try:
            candis = mapdata[str(int(idx[0]))]
        except KeyboardInterrupt:
            raise
        except:
            candis = mapdata[int(idx[0])]
        for candi_idx in candis:
            mask[candi_idx] = 1
        masks.append(mask)
    masks = torch.stack(masks)
    return masks

def text2json(text):
    result = {}
    result['text'] = text
    return result

def preprocessor(input_data):
    if type(input_data) == str:
        data = text2json(input_data)
    else:
        data = input_data
    tokens = data['text'].split(' ')
    idxs = []
    for i in range(len(tokens)):
        idxs.append(str(i))
    result = []
    result.append(idxs)
    result.append(tokens)
    return result


class for_BERT():
    
    def __init__(self, mode='training', version=1.0):
        version = str(version)
        self.mode = mode
        data_path = dir_path+'/../koreanframenet/resource/info/kfn'+version+'_'
        with open(data_path+'lu2idx.json','r') as f:
            self.lu2idx = json.load(f)
        with open(dir_path+'/../koreanframenet/resource/info/fn1.7_frame2idx.json','r') as f:
            self.sense2idx = json.load(f)      
        with open(data_path+'lufrmap.json','r') as f:
            self.lusensemap = json.load(f)

        self.idx2sense = dict(zip(self.sense2idx.values(),self.sense2idx.keys()))
        self.idx2lu = dict(zip(self.lu2idx.values(),self.lu2idx.keys()))

        # load pretrained BERT tokenizer
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased', do_lower_case=False)

    def idx2tag(self, predictions):
        pred_tags = [self.idx2sense[p_i] for p in predictions for p_i in p]
        return pred_tags
    
    def get_masks(self, datas, model='frameid'):
        if model == 'frameid':
            mapdata = self.lusensemap
            num_label = len(self.sense2idx)
        masks = []
        for idx in datas:
            mask = torch.zeros(num_label)
            try:
                candis = mapdata[str(int(idx[0]))]
            except KeyboardInterrupt:
                raise
            except:
                candis = mapdata[int(idx[0])]
            for candi_idx in candis:
                mask[candi_idx] = 1
            masks.append(mask)
        masks = torch.stack(masks)
        return masks
    
    
    # bert tokenizer
    def bert_tokenizer(self, text):
        orig_tokens = text.split(' ')
        bert_tokens = []
        orig_to_tok_map = []
        bert_tokens.append("[CLS]")
        for orig_token in orig_tokens:
            orig_to_tok_map.append(len(bert_tokens))
            bert_tokens.extend(self.tokenizer.tokenize(orig_token))
        bert_tokens.append("[SEP]")

        return orig_tokens, bert_tokens, orig_to_tok_map

    def convert_to_bert_input_frameid(self, input_data):
        tokenized_texts, lus, senses = [],[],[]

        for i in range(len(input_data)):    
            data = input_data[i]
            text = ' '.join(data[0])
            orig_tokens, bert_tokens, orig_to_tok_map = self.bert_tokenizer(text)
            tokenized_texts.append(bert_tokens)

            ori_lus = data[1]    
            lu_sequence = []
            for i in range(len(bert_tokens)):
                if i in orig_to_tok_map:
                    idx = orig_to_tok_map.index(i)
                    l = ori_lus[idx]
                    lu_sequence.append(l)
                else:
                    lu_sequence.append('_')
            lus.append(lu_sequence)        

            if self.mode == 'training':
                ori_senses = data[2]    
                sense_sequence = []
                for i in range(len(bert_tokens)):
                    if i in orig_to_tok_map:
                        idx = orig_to_tok_map.index(i)
                        l = ori_senses[idx]
                        sense_sequence.append(l)
                    else:
                        sense_sequence.append('_')
                senses.append(sense_sequence)

        input_ids = pad_sequences([self.tokenizer.convert_tokens_to_ids(txt) for txt in tokenized_texts],
                              maxlen=MAX_LEN, dtype="long", truncating="post", padding="post")

        tgt_seq, lu_seq, sense_seq = [],[],[]
        for sent_idx in range(len(lus)):
            lu_items = lus[sent_idx]
            tgt,lu = [],[]
            for idx in range(len(lu_items)):
                if lu_items[idx] != '_':
                    if len(tgt) == 0:
                        tgt.append(idx)
                        lu.append(self.lu2idx[lu_items[idx]])
            tgt_seq.append(tgt)
            lu_seq.append(lu)
            
            if self.mode == 'training':
                sense_items = senses[sent_idx]
                sense = []
                for idx in range(len(sense_items)):
                    if sense_items[idx] != '_':
                        if len(sense) == 0:
                            sense.append(self.sense2idx[sense_items[idx]])
                sense_seq.append(sense)
            
            

        attention_masks = [[float(i>0) for i in ii] for ii in input_ids]    
        data_inputs = torch.tensor(input_ids)
        data_tgt_idx = torch.tensor(tgt_seq)
        data_lus = torch.tensor(lu_seq)
        data_senses = torch.tensor(sense_seq)
        data_masks = torch.tensor(attention_masks)
        
        if self.mode == 'training':
            bert_inputs = TensorDataset(data_inputs, data_tgt_idx, data_lus, data_senses, data_masks)
        else:
            bert_inputs = TensorDataset(data_inputs, data_tgt_idx, data_lus,data_masks)

        return bert_inputs