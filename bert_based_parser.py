
# coding: utf-8

# In[1]:


import json
import sys
sys.path.append('../')

import numpy as np
from KAIST_frame_parser.src import dataio, etri
from KAIST_frame_parser.src import targetid
import torch
from torch import nn
import os
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
n_gpu = torch.cuda.device_count()

from pytorch_pretrained_bert import BertTokenizer, BertConfig, BertModel
from keras.preprocessing.sequence import pad_sequences

from KAIST_frame_parser.koreanframenet.src import conll2textae
from KAIST_frame_parser.koreanframenet import koreanframenet

from KAIST_frame_parser.src.fn_modeling import BertForJointFrameParsing

from konlpy.tag import Kkma
from pprint import pprint


# In[2]:


MAX_LEN = 256
batch_size = 6

try:
    dir_path = os.path.dirname(os.path.abspath( __file__ ))
except:
    dir_path = '.'


# In[3]:


def data2tgt_data(input_data, mode='training'):
    result = []
    for item in input_data:
        
        if mode == 'training':
            ori_tokens, ori_lus, ori_frames, ori_args = item[0],item[1],item[2],item[3]
        else:
            ori_tokens, ori_lus = item[0],item[1]
            
            
        for idx in range(len(ori_lus)):
            lu = ori_lus[idx]
            if lu != '_':
                if idx == 0:
                    begin = idx
                elif ori_lus[idx-1] == '_':
                    begin = idx
                end = idx
                
        if mode == 'training':
            tokens, lus, frames, args = [],[],[],[]
        else:
            tokens, lus = [],[]
            
        for idx in range(len(ori_lus)):
            token = ori_tokens[idx]
            lu = ori_lus[idx]
            if mode == 'training':
                frame = ori_frames[idx]
                arg = ori_args[idx]
                
            if idx == begin:
                tokens.append('<tgt>')
                lus.append('_')
                if mode == 'training':
                    frames.append('_')
                    args.append('O')
                
            tokens.append(token)
            lus.append(lu)
            if mode == 'training':
                frames.append(frame)
                args.append(arg)
            
            if idx == end:
                tokens.append('</tgt>')
                lus.append('_')
                if mode == 'training':
                    frames.append('_')
                    args.append('O')
        sent = []
        sent.append(tokens)
        sent.append(lus)
        
        if mode == 'training':
            sent.append(frames)
            sent.append(args)
            
        result.append(sent)
    return result 


# In[4]:


class for_BERT():
    
    def __init__(self, mode='training', language='ko', version=1.0):
        version = str(version)
        self.mode = mode
        if language == 'en':
            data_path = dir_path+'/koreanframenet/resource/info/fn'+version+'_'
        else:
            data_path = dir_path+'/koreanframenet/resource/info/kfn'+version+'_'
        with open(data_path+'lu2idx.json','r') as f:
            self.lu2idx = json.load(f)
        if version == '1.5':
            fname = dir_path+'/koreanframenet/resource/info/fn1.5_frame2idx.json'
        else:
            fname = dir_path+'/koreanframenet/resource/info/fn1.7_frame2idx.json'
        with open(fname,'r') as f:
            #self.sense2idx = json.load(f)
            self.frame2idx = json.load(f)
        with open(data_path+'lufrmap.json','r') as f:
            #self.lusensemap = json.load(f)
            self.lufrmap = json.load(f)
        with open(dir_path+'/koreanframenet/resource/info/fn1.7_fe2idx.json','r') as f:
            self.arg2idx = json.load(f)
        with open(dir_path+'/koreanframenet/resource/info/fn1.7_frargmap.json','r') as f:
            self.frargmap = json.load(f)
        with open(dir_path+'/koreanframenet/resource/info/fn1.7_bio_fe2idx.json','r') as f:
            self.bio_arg2idx = json.load(f)
        with open(dir_path+'/koreanframenet/resource/info/fn1.7_bio_frargmap.json','r') as f:
            self.bio_frargmap = json.load(f)

        self.idx2frame = dict(zip(self.frame2idx.values(),self.frame2idx.keys()))
        self.idx2lu = dict(zip(self.lu2idx.values(),self.lu2idx.keys()))
        self.idx2arg = dict(zip(self.arg2idx.values(),self.arg2idx.keys()))
        self.idx2bio_arg = dict(zip(self.bio_arg2idx.values(),self.bio_arg2idx.keys()))

        # load pretrained BERT tokenizer
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased', do_lower_case=False)
        
        # load BERT tokenizer with untokenizing frames
        never_split_tuple = ("[UNK]", "[SEP]", "[PAD]", "[CLS]", "[MASK]")
        added_never_split = []
        added_never_split.append('<tgt>')
        added_never_split.append('</tgt>')
        added_never_split_tuple = tuple(added_never_split)
        never_split_tuple += added_never_split_tuple
        vocab_file_path = dir_path+'/data/bert-multilingual-cased-dict-add-frames'
        self.tokenizer_with_frame = BertTokenizer(vocab_file_path, do_lower_case=False, max_len=256, never_split=never_split_tuple)

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
            bert_tokens.extend(self.tokenizer_with_frame.tokenize(orig_token))
        bert_tokens.append("[SEP]")

        return orig_tokens, bert_tokens, orig_to_tok_map
    
    def convert_to_bert_input_JointFrameParsing(self, input_data):
        tokenized_texts, lus, frames, args = [],[],[],[]
        orig_tok_to_maps = []
        for i in range(len(input_data)):    
            data = input_data[i]
            text = ' '.join(data[0])
            orig_tokens, bert_tokens, orig_to_tok_map = self.bert_tokenizer(text)
            orig_tok_to_maps.append(orig_to_tok_map)
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
                ori_frames, ori_args = data[2], data[3]
                frame_sequence, arg_sequence = [],[]
                for i in range(len(bert_tokens)):
                    if i in orig_to_tok_map:
                        idx = orig_to_tok_map.index(i)
                        fr = ori_frames[idx]
                        frame_sequence.append(fr)
                        ar = ori_args[idx]
                        arg_sequence.append(ar)
                    else:
                        frame_sequence.append('_')
                        arg_sequence.append('X')
                frames.append(frame_sequence)
                args.append(arg_sequence)

        input_ids = pad_sequences([self.tokenizer.convert_tokens_to_ids(txt) for txt in tokenized_texts],
                              maxlen=MAX_LEN, dtype="long", truncating="post", padding="post")
        orig_tok_to_maps = pad_sequences(orig_tok_to_maps, maxlen=MAX_LEN, dtype="long", truncating="post", padding="post", value=-1)
        
        if self.mode =='training':
            arg_ids = pad_sequences([[self.bio_arg2idx.get(ar) for ar in arg] for arg in args],
                                    maxlen=MAX_LEN, value=self.bio_arg2idx["X"], padding="post",
                                    dtype="long", truncating="post")

        lu_seq, frame_seq = [],[]
        for sent_idx in range(len(lus)):
            lu_items = lus[sent_idx]
            lu = []
            for idx in range(len(lu_items)):
                if lu_items[idx] != '_':
                    if len(lu) == 0:
                        lu.append(self.lu2idx[lu_items[idx]])
            lu_seq.append(lu)
            
            if self.mode == 'training':
                frame_items, arg_items = frames[sent_idx], args[sent_idx]
                frame= []
                for idx in range(len(frame_items)):
                    if frame_items[idx] != '_':
                        if len(frame) == 0:
                            frame.append(self.frame2idx[frame_items[idx]])
                frame_seq.append(frame)

        attention_masks = [[float(i>0) for i in ii] for ii in input_ids]    
        data_inputs = torch.tensor(input_ids)
        data_orig_tok_to_maps = torch.tensor(orig_tok_to_maps)
        data_lus = torch.tensor(lu_seq)
        data_masks = torch.tensor(attention_masks)
        
        if self.mode == 'training':
            data_frames = torch.tensor(frame_seq)
            data_args = torch.tensor(arg_ids)
            bert_inputs = TensorDataset(data_inputs, data_orig_tok_to_maps, data_lus, data_frames, data_args, data_masks)
        else:
            bert_inputs = TensorDataset(data_inputs, data_orig_tok_to_maps, data_lus, data_masks)
        return bert_inputs


# In[30]:


def logit2label(logit, mask):
    masking = np.multiply(logit, mask)
    masking[masking==0] = np.NINF
    sm = nn.Softmax()
    pred_logits = sm(masking).view(1,-1)
    score, label = pred_logits.max(1)
    score = float(score)
    return label, score


# In[78]:


class BERTbasedParser():
    def __init__(self, version=1.1, language='ko', model_dir=False):
        self.version = version
        self.language = language
        
        #load model
        self.model_dir = model_dir        
        self.model = torch.load(model_dir)
        self.model.eval()
        
        self.bert_io = for_BERT(mode='parse', language=language, version=version)
        
    def joint_parser(self, text):
        conll_data = dataio.preprocessor(text)
        
        # target ID
        tid_data = targetid.baseline(conll_data)
        
        # add <tgt> and </tgt> to target word
        tgt_data = data2tgt_data(tid_data, mode='parse')
        
        
        result = []
        if tgt_data:
        
            # convert conll to bert inputs
            bert_inputs = self.bert_io.convert_to_bert_input_JointFrameParsing(tgt_data)
            dataloader = DataLoader(bert_inputs, sampler=None, batch_size=1)

            pred_frames, pred_args = [],[]
            for batch in dataloader:
                batch = tuple(t.to(device) for t in batch)
                b_input_ids, b_orig_tok_to_maps, b_lus, b_masks = batch

                with torch.no_grad():
                    frame_logits, arg_logits = self.model(b_input_ids, token_type_ids=None, 
                                    lus=b_lus, attention_mask=b_masks)

                frame_logits = frame_logits.detach().cpu().numpy()
                arg_logits = arg_logits.detach().cpu().numpy()
                input_ids = b_input_ids.to('cpu').numpy()
                lufr_masks = dataio.get_masks(b_lus, self.bert_io.lufrmap, num_label=len(self.bert_io.frame2idx)).to(device)


                for b_idx in range(len(frame_logits)):
                    input_id = input_ids[b_idx]
                    frame_logit = frame_logits[b_idx]
                    arg_logit = arg_logits[b_idx]
                    lufr_mask = lufr_masks[b_idx]
                    orig_tok_to_map = b_orig_tok_to_maps[b_idx]

                    pred_frame, frame_score = logit2label(frame_logit, lufr_mask)
                    frarg_mask = dataio.get_masks([pred_frame], self.bert_io.bio_frargmap, num_label=len(self.bert_io.bio_arg2idx)).to(device)[0]

                    pred_arg_bert = []
                    for logit in arg_logit:
                        label, score = logit2label(logit, frarg_mask)
                        pred_arg_bert.append(int(label))

                    #infer
                    pred_arg = []
                    for idx in orig_tok_to_map:
                        if idx != -1:
                            tok_id = int(input_id[idx])
                            if tok_id == 1:
                                pass
                            elif tok_id == 2:
                                pass
                            else:
                                pred_arg.append(pred_arg_bert[idx])
                    pred_frames.append([int(pred_frame)])
                    pred_args.append(pred_arg)

            pred_frame_tags = [self.bert_io.idx2frame[p_i] for p in pred_frames for p_i in p]
            pred_arg_tags = [[self.bert_io.idx2bio_arg[p_i] for p_i in p] for p in pred_args]

            for i in range(len(pred_arg_tags)):                
                conll = tid_data[i]
                frame_seq = ['_' for i in range(len(conll[0]))]
                for idx in range(len(conll[1])):
                    if conll[1][idx] != '_':
                        frame_seq[idx] = pred_frame_tags[i]
                conll.append(frame_seq)
                conll.append(pred_arg_tags[i])
                result.append(conll)
                
        return result


# In[79]:


# model_dir = '/disk_4/resource/models/kfn1.1/joint/epoch-19-joint.pt'
# parser = BERTbasedParser(model_dir=model_dir)


# In[95]:


def test_parser():
#     model_dir = '/disk_4/resource/models/kfn1.1/epoch-9-frameid.pt'
    
    text  = '헤밍웨이는 1899년 7월 21일 미국 일리노이에서 태어났고, 62세에 자살로 사망했다.'
    text = '그는 그녀와 사랑에 빠졌다.'
    text = '얼룩이 옷에서 빠졌다.'
    d = parser.joint_parser(text)
    for i in d:
        print(i)
        print('')
        
    textae = conll2textae.get_textae(d)
    print(textae)
    
# test_parser()

