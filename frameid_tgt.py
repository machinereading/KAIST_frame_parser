
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
from torch.optim import Adam
import glob
import os
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from keras.preprocessing.sequence import pad_sequences
from pytorch_pretrained_bert import BertTokenizer, BertConfig, BertModel
from tqdm import tqdm, trange
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
n_gpu = torch.cuda.device_count()

from KAIST_frame_parser.koreanframenet.src import conll2textae
from KAIST_frame_parser.koreanframenet import koreanframenet

from KAIST_frame_parser.src.fn_modeling import BertForFrameId

from sklearn.metrics import accuracy_score

from pprint import pprint


# In[2]:


MAX_LEN = 256
batch_size = 6


# In[3]:


# language = 'en'
# version = 1.7
# version = 1.5

language = 'ko'
version = 0.8
if language == 'en':
    framenet = 'fn'+str(version)
    fn_dir = '/disk_4/resource/fn'+str(version)
    trn_d, dev_d, tst_d = dataio.load_fn_data(fn_dir)
elif language == 'ko':
    framenet = 'kfn'+str(version)
    kfn = koreanframenet.interface(version=version)
    trn_d, dev_d, tst_d = kfn.load_data()
    
try:
    dir_path = os.path.dirname(os.path.abspath( __file__ ))
except:
    dir_path = '.'
    
# save your model to
model_dir = '/disk_4/resource/models/'

print(len(trn_d))
print(len(dev_d))
print(len(tst_d))


# In[4]:


def data2tgt_data(input_data):
    result = []
    for item in input_data:
        ori_tokens, ori_lus, ori_frames, ori_args = item[0],item[1],item[2],item[3]
        for idx in range(len(ori_lus)):
            lu = ori_lus[idx]
            if lu != '_':
                if idx == 0:
                    begin = idx
                elif ori_lus[idx-1] == '_':
                    begin = idx
                end = idx
        tokens, lus, frames, args = [],[],[],[]
        for idx in range(len(ori_lus)):
            token = ori_tokens[idx]
            lu = ori_lus[idx]
            frame = ori_frames[idx]
            arg = ori_args[idx]
            if idx == begin:
                tokens.append('<tgt>')
                lus.append('_')
                frames.append('_')
                args.append('X')
                
            tokens.append(token)
            lus.append(lu)
            frames.append(frame)
            args.append(arg)
            
            if idx == end:
                tokens.append('</tgt>')
                lus.append('_')
                frames.append('_')
                args.append('X')
        sent = []
        sent.append(tokens)
        sent.append(lus)
        sent.append(frames)
        sent.append(args)
        result.append(sent)
    return result 
    
trn = data2tgt_data(trn_d)
dev = data2tgt_data(dev_d)
tst = data2tgt_data(tst_d)


# In[5]:


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
#         for frame in self.frame2idx:
#             added_never_split.append('['+frame+']')
        added_never_split_tuple = tuple(added_never_split)
        never_split_tuple += added_never_split_tuple
        vocab_file_path = dir_path+'/data/bert-multilingual-cased-dict-add-frames'
        self.tokenizer_with_frame = BertTokenizer(vocab_file_path, do_lower_case=False, max_len=512, never_split=never_split_tuple)

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


# In[6]:


bert_io = for_BERT(mode='training', language=language, version=version)


# In[7]:


def train():
    model_path = model_dir+framenet+'/'
    print('your model would be saved at', model_path)
    
    model = BertForFrameId.from_pretrained("bert-base-multilingual-cased", num_labels = len(bert_io.frame2idx), lufrmap=bert_io.lufrmap)
    model.cuda();
    
    trn_data = bert_io.convert_to_bert_input_frameid(trn)
    sampler = RandomSampler(trn)
    trn_dataloader = DataLoader(trn_data, sampler=sampler, batch_size=batch_size)
    
    # load optimizer
    FULL_FINETUNING = True
    if FULL_FINETUNING:
        param_optimizer = list(model.named_parameters())
        no_decay = ['bias', 'gamma', 'beta']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
             'weight_decay_rate': 0.01},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
             'weight_decay_rate': 0.0}
        ]
    else:
        param_optimizer = list(model.classifier.named_parameters()) 
        optimizer_grouped_parameters = [{"params": [p for n, p in param_optimizer]}]
    optimizer = Adam(optimizer_grouped_parameters, lr=3e-5)
    
    
    # train 
    epochs = 10
    max_grad_norm = 1.0
    num_of_epoch = 0
    accuracy_result = []
    for _ in trange(epochs, desc="Epoch"):
        # TRAIN loop
        model.train()
        tr_loss = 0
        nb_tr_examples, nb_tr_steps = 0, 0
        for step, batch in enumerate(trn_dataloader):
            # add batch to gpu
            batch = tuple(t.to(device) for t in batch)
            b_input_ids, b_input_tgt_idxs, b_input_lus, b_input_frames, b_input_masks = batch            
            # forward pass
            loss = model(b_input_ids, token_type_ids=None, lus=b_input_lus, frames=b_input_frames, attention_mask=b_input_masks)
            # backward pass
            loss.backward()
            # track train loss
            tr_loss += loss.item()
            nb_tr_examples += b_input_ids.size(0)
            nb_tr_steps += 1
            # gradient clipping
            torch.nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=max_grad_norm)
            # update parameters
            optimizer.step()
            model.zero_grad()
#             break

        # print train loss per epoch
        print("Train loss: {}".format(tr_loss/nb_tr_steps))
        model_saved_path = model_path+'epoch-'+str(num_of_epoch)+'-frameid.pt'        
        torch.save(model, model_saved_path)
        num_of_epoch += 1
    print('...training is done')


# In[8]:


def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=2).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)

def test():
    model_path = model_dir+framenet+'/'
    models = glob.glob(model_path+'*.pt')
    results = []
    for m in models:
        print('model:', m)
        model = torch.load(m)
        model.eval()

        tst_data = bert_io.convert_to_bert_input_frameid(tst)
        sampler = RandomSampler(tst)
        tst_dataloader = DataLoader(tst_data, sampler=sampler, batch_size=batch_size)

        eval_loss, eval_accuracy = 0, 0
        nb_eval_steps, nb_eval_examples = 0, 0
        predictions , true_labels, scores, candis, all_lus = [], [], [], [], []

        for batch in tst_dataloader:
            batch = tuple(t.to(device) for t in batch)
            b_input_ids, b_tgt_idxs, b_lus, b_frames, b_masks = batch

            with torch.no_grad():
                tmp_eval_loss = model(b_input_ids, token_type_ids=None, 
                                     lus=b_lus, frames=b_frames, attention_mask=b_masks)
                logits = model(b_input_ids, token_type_ids=None, 
                                lus=b_lus, attention_mask=b_masks)
            logits = logits.detach().cpu().numpy()
            label_ids = b_frames.to('cpu').numpy()          
            masks = dataio.get_masks(b_lus, bert_io.lufrmap, num_label=len(bert_io.frame2idx)).to(device)
            for lu in b_lus:
                candi_idx = bert_io.lufrmap[str(int(lu))]
                candi = [bert_io.idx2frame[c] for c in candi_idx]
                candi_txt = ','.join(candi)
                candi_txt = str(len(candi))+'\t'+candi_txt
                candis.append(candi_txt)
                all_lus.append(bert_io.idx2lu[int(lu)])

            for b_idx in range(len(logits)):
                logit = logits[b_idx]
                mask = masks[b_idx]
                b_pred_idxs, b_pred_logits = [],[]
                for fr_idx in range(len(mask)):
                    if mask[fr_idx] > 0:
                        b_pred_idxs.append(fr_idx)
                        b_pred_logits.append(logit[fr_idx].item())
                b_pred_idxs = torch.tensor(b_pred_idxs)
                b_pred_logits = torch.tensor(b_pred_logits)
                sm = nn.Softmax()
                b_pred_logits = sm(b_pred_logits).view(1, -1)
                score, indice = b_pred_logits.max(1)                
                prediction = b_pred_idxs[indice]
                predictions.append([int(prediction)])
                score = float(score)
                scores.append(score)
            true_labels.append(label_ids)
#             break

        pred_tags = [bert_io.idx2frame[p_i] for p in predictions for p_i in p]
        valid_tags = [bert_io.idx2frame[l_ii] for l in true_labels for l_i in l for l_ii in l_i]

        acc = accuracy_score(pred_tags, valid_tags)
        print("Accuracy: {}".format(accuracy_score(pred_tags, valid_tags)))
        
        result = m+'\t'+str(acc)+'\n'
        results.append(result)
        
        epoch = m.split('-')[1]
        fname = model_path+str(epoch)+'-result.txt'
        with open(fname, 'w') as f:
            line = 'accuracy: '+str(acc) +'\n\n'
            f.write(line)
            line = 'gold' + '\t' + 'prediction' + '\t' + 'score' + '\t' + 'lu' + '\t' + 'candis' + '\n'
            f.write(line)
            for r in range(len(pred_tags)):
                line = valid_tags[r] + '\t' + pred_tags[r] + '\t' + str(scores[r]) + '\t' + all_lus[r] + '\t' + candis[r] + '\n'
                f.write(line)
    fname = model_path+'accuracy.txt'
    with open(fname, 'w') as f:
        for r in results:
            f.write(r)

    print('result is written to', fname)


# In[9]:


train()
test()

