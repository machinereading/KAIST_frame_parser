import torch
import sys
import glob
import json
sys.path.insert(0,'../')
sys.path.insert(0,'../../')

import numpy as np
# from src import models
import models
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from keras.preprocessing.sequence import pad_sequences
from pytorch_pretrained_bert import BertTokenizer, BertConfig, BertModel
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
n_gpu = torch.cuda.device_count()

import os
dir_path = os.path.dirname( os.path.abspath( __file__ ))

MAX_LEN = 256
batch_size = 8

def load_fn_data(fn_dir):
    fnames = [f for f in glob.glob(fn_dir + "/*.conll")]
    trn, dev, tst = [],[],[]
    for fname in fnames:
        with open(fname, 'r') as f:
            lines = f.readlines()
        tsv, sent = [],[]
        for line in lines:
            line = line.strip()
            if line != '':
                token = line.split('\t')
                sent.append(token)
            else:
                tsv.append(sent)
                sent = []
        data = []
        for sent in tsv:     
            tok_str, tok_lu, tok_frame, tok_fe= [],[],[],[]
            for token in sent:
                tok_str.append(token[1])
                tok_lu.append(token[12])
                tok_frame.append(token[13])
                
                if 'B-' in token[14]:
                    old_fe = token[14].split('B-')[-1]
                    if '-' in old_fe:
                        new_fe = old_fe.replace('-','_')
                    else:
                        new_fe = old_fe
                    arg = 'B-'+new_fe
                    
                elif 'I-' in token[14]:
                    old_fe = token[14].split('I-')[-1]
                    if '-' in old_fe:
                        new_fe = old_fe.replace('-','_')
                    else:
                        new_fe = old_fe
                    arg = 'I-'+new_fe
                else:
                    arg = token[14]
                                      
                tok_fe.append(arg)
            sent_list = []
            sent_list.append(tok_str)
            sent_list.append(tok_lu)
            sent_list.append(tok_frame)
            sent_list.append(tok_fe)
            data.append(sent_list)
        if 'train' in fname:
            trn = data
        elif 'dev' in fname:
            dev = data
        elif 'test' in fname:
            tst = data
    return trn, dev, tst
        

def get_masks(datas, mapdata, num_label=2):
    masks = []
    for idx in datas:
        mask = torch.zeros(num_label)
#         mask[mask==0] = np.NINF
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
    
    def __init__(self, mode='training', language='ko', version=1.0):
        version = str(version)
        self.mode = mode
        if language == 'en':
            data_path = dir_path+'/../koreanframenet/resource/info/fn'+version+'_'
        else:
            data_path = dir_path+'/../koreanframenet/resource/info/kfn'+version+'_'
        with open(data_path+'lu2idx.json','r') as f:
            self.lu2idx = json.load(f)
        with open(dir_path+'/../koreanframenet/resource/info/fn1.7_frame2idx.json','r') as f:
            #self.sense2idx = json.load(f)
            self.frame2idx = json.load(f)
        with open(data_path+'lufrmap.json','r') as f:
            #self.lusensemap = json.load(f)
            self.lufrmap = json.load(f)
        with open(dir_path+'/../koreanframenet/resource/info/fn1.7_fe2idx.json','r') as f:
            self.arg2idx = json.load(f)
        with open(dir_path+'/../koreanframenet/resource/info/fn1.7_frargmap.json','r') as f:
            self.frargmap = json.load(f)
        with open(dir_path+'/../koreanframenet/resource/info/fn1.7_bio_fe2idx.json','r') as f:
            self.bio_arg2idx = json.load(f)
        with open(dir_path+'/../koreanframenet/resource/info/fn1.7_bio_frargmap.json','r') as f:
            self.bio_frargmap = json.load(f)

        self.idx2frame = dict(zip(self.frame2idx.values(),self.frame2idx.keys()))
        self.idx2lu = dict(zip(self.lu2idx.values(),self.lu2idx.keys()))
        self.idx2arg = dict(zip(self.arg2idx.values(),self.arg2idx.keys()))
        self.idx2bio_arg = dict(zip(self.bio_arg2idx.values(),self.bio_arg2idx.keys()))

        # load pretrained BERT tokenizer
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased', do_lower_case=False)

    def idx2tag(self, predictions, model='frameid'):
        if model == 'frameid':
            pred_tags = [self.idx2frame[p_i] for p in predictions for p_i in p]
        elif model == 'argclassification':
            pred_tags = [self.idx2arg[p_i] for p in predictions for p_i in p]
        elif model == 'argid':
            pred_tags = [self.idx2bio_arg[p_i] for p in predictions for p_i in p]
        return pred_tags
    
    def get_masks(self, datas, model='frameid'):
        if model == 'frameid':
            mapdata = self.lufrmap
            num_label = len(self.frame2idx)
        elif model == 'argclassification':
            mapdata = self.frargmap
            num_label = len(self.arg2idx)
        elif model == 'argid':
            mapdata = self.bio_frargmap
            num_label = len(self.bio_arg2idx)
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
    
    
    # bert tokenizer and assign to the first token
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
    
    # bert tokenizer and assign to the last token
    def bert_tokenizer_assign_to_last_token(self, text):
        orig_tokens = text.split(' ')
        bert_tokens = []
        orig_to_tok_map = []
        bert_tokens.append("[CLS]")
        for orig_token in orig_tokens:
            bert_tokens.extend(self.tokenizer.tokenize(orig_token))
            orig_to_tok_map.append(len(bert_tokens)-1)
        bert_tokens.append("[SEP]")

        return orig_tokens, bert_tokens, orig_to_tok_map

    def convert_to_bert_input_frameid(self, input_data):
        tokenized_texts, lus, frames = [],[],[]

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
                ori_frames = data[2]    
                frame_sequence = []
                for i in range(len(bert_tokens)):
                    if i in orig_to_tok_map:
                        idx = orig_to_tok_map.index(i)
                        l = ori_frames[idx]
                        frame_sequence.append(l)
                    else:
                        frame_sequence.append('_')
                frames.append(frame_sequence)

        input_ids = pad_sequences([self.tokenizer.convert_tokens_to_ids(txt) for txt in tokenized_texts],
                              maxlen=MAX_LEN, dtype="long", truncating="post", padding="post")

        tgt_seq, lu_seq, frame_seq = [],[],[]
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
                frame_items = frames[sent_idx]
                frame = []
                for idx in range(len(frame_items)):
                    if frame_items[idx] != '_':
                        if len(frame) == 0:
                            frame.append(self.frame2idx[frame_items[idx]])
                frame_seq.append(frame)
            
            

        attention_masks = [[float(i>0) for i in ii] for ii in input_ids]    
        data_inputs = torch.tensor(input_ids)
        data_tgt_idx = torch.tensor(tgt_seq)
        data_lus = torch.tensor(lu_seq)
        data_frames = torch.tensor(frame_seq)
        data_masks = torch.tensor(attention_masks)
        
        if self.mode == 'training':
            bert_inputs = TensorDataset(data_inputs, data_tgt_idx, data_lus, data_frames, data_masks)
        else:
            bert_inputs = TensorDataset(data_inputs, data_tgt_idx, data_lus,data_masks)
        return bert_inputs
    
    
    def convert_to_bert_input_arg_classifier(self, input_data):
        tokenized_texts, lus, frames, args = [],[],[],[]

        for i in range(len(input_data)):    
            data = input_data[i]
            text = ' '.join(data[0])
            orig_tokens, bert_tokens, orig_to_tok_map = self.bert_tokenizer_assign_to_last_token(text)
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

            ori_frames = data[2]    
            frame_sequence = []
            for i in range(len(bert_tokens)):
                if i in orig_to_tok_map:
                    idx = orig_to_tok_map.index(i)
                    l = ori_frames[idx]
                    frame_sequence.append(l)
                else:
                    frame_sequence.append('_')
            frames.append(frame_sequence)
            
            if self.mode == 'training':
                ori_args = data[3]    
                arg_sequence = []
                for i in range(len(bert_tokens)):
                    if i in orig_to_tok_map:
                        idx = orig_to_tok_map.index(i)
                        l = ori_args[idx]
                        arg_sequence.append(l)
                    else:
                        arg_sequence.append('O')
                args.append(arg_sequence)

        input_ids = pad_sequences([self.tokenizer.convert_tokens_to_ids(txt) for txt in tokenized_texts],
                              maxlen=MAX_LEN, dtype="long", truncating="post", padding="post")

        tgt_seq, lu_seq, frame_seq, arg_idx_seq, arg_seq = [],[],[],[],[]
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

            frame_items = frames[sent_idx]
            frame = []
            for idx in range(len(frame_items)):
                if frame_items[idx] != '_':
                    if len(frame) == 0:
                        frame.append(self.frame2idx[frame_items[idx]])
            frame_seq.append(frame)
                
            if self.mode == 'training':
                arg_items = args[sent_idx]
                arg, arg_idx = [],[]
                for idx in range(len(arg_items)):
                    if arg_items[idx] != 'O':
                        if len(arg) == 0:
                            try:
                                arg.append(self.arg2idx[arg_items[idx]])
                                arg_idx.append(idx)
                            except KeyboardInterrupt:
                                raise
                            except:
                                print(arg_items[idx])
                arg_seq.append(arg)
                arg_idx_seq.append(arg_idx)
            
            

        attention_masks = [[float(i>0) for i in ii] for ii in input_ids]    
        data_inputs = torch.tensor(input_ids)
        data_tgt_idx = torch.tensor(tgt_seq)
        data_lus = torch.tensor(lu_seq)
        data_frames = torch.tensor(frame_seq)
        data_arg_idxs = torch.tensor(arg_idx_seq)
        data_args = torch.tensor(arg_seq)
        data_masks = torch.tensor(attention_masks)
        
        if self.mode == 'training':
            bert_inputs = TensorDataset(data_inputs, data_tgt_idx, data_lus, data_frames, data_arg_idxs, data_args, data_masks)
        else:
            bert_inputs = TensorDataset(data_inputs, data_tgt_idx, data_lus, data_frames, data_masks)
        return bert_inputs
    
    
    def convert_to_bert_input_argid(self, input_data):
        tokenized_texts, lus, frames, args = [],[],[],[]

        for i in range(len(input_data)):    
            data = input_data[i]
            text = ' '.join(data[0])
            orig_tokens, bert_tokens, orig_to_tok_map = self.bert_tokenizer_assign_to_last_token(text)
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

            ori_frames = data[2]    
            frame_sequence = []
            for i in range(len(bert_tokens)):
                if i in orig_to_tok_map:
                    idx = orig_to_tok_map.index(i)
                    l = ori_frames[idx]
                    frame_sequence.append(l)
                else:
                    frame_sequence.append('_')
            frames.append(frame_sequence)
            
            if self.mode == 'training':
                ori_args = data[3]    
                arg_sequence = []
                for i in range(len(bert_tokens)):
                    if i in orig_to_tok_map:
                        idx = orig_to_tok_map.index(i)
                        l = ori_args[idx]
                        arg_sequence.append(l)
                    else:
                        arg_sequence.append('O')
                args.append(arg_sequence)

        input_ids = pad_sequences([self.tokenizer.convert_tokens_to_ids(txt) for txt in tokenized_texts],
                              maxlen=MAX_LEN, dtype="long", truncating="post", padding="post")

        tgt_seq, lu_seq, frame_seq, arg_idx_seq, arg_seq = [],[],[],[],[]
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

            frame_items = frames[sent_idx]
            frame = []
            for idx in range(len(frame_items)):
                if frame_items[idx] != '_':
                    if len(frame) == 0:
                        frame.append(self.frame2idx[frame_items[idx]])
            frame_seq.append(frame)
                
            if self.mode == 'training':
                arg_items = args[sent_idx]
                arg, arg_idx = [],[]
                for idx in range(len(arg_items)):
                    if arg_items[idx] != 'O':
                        if len(arg) == 0:
                            try:
                                arg.append(self.arg2idx[arg_items[idx]])
                                arg_idx.append(idx)
                            except KeyboardInterrupt:
                                raise
                            except:
                                print(arg_items[idx])
                arg_seq.append(arg)
                arg_idx_seq.append(arg_idx)
            
            

        attention_masks = [[float(i>0) for i in ii] for ii in input_ids]    
        data_inputs = torch.tensor(input_ids)
        data_tgt_idx = torch.tensor(tgt_seq)
        data_lus = torch.tensor(lu_seq)
        data_frames = torch.tensor(frame_seq)
        data_arg_idxs = torch.tensor(arg_idx_seq)
        data_args = torch.tensor(arg_seq)
        data_masks = torch.tensor(attention_masks)
        
        if self.mode == 'training':
            bert_inputs = TensorDataset(data_inputs, data_tgt_idx, data_lus, data_frames, data_arg_idxs, data_args, data_masks)
        else:
            bert_inputs = TensorDataset(data_inputs, data_tgt_idx, data_lus, data_frames, data_masks)
        return bert_inputs