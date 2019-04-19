
# coding: utf-8

# In[1]:


import torch
import os
import sys
sys.path.append('../')
sys.path.append('../../')

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
torch.cuda.get_device_name(0)

from KAIST_frame_parser.src import dataio
from KAIST_frame_parser.src.fn_modeling import BertForFrameIdentification
from KAIST_frame_parser.src.fn_modeling import BertForArgClassification
from KAIST_frame_parser.koreanframenet import koreanframenet
from sklearn.metrics import accuracy_score

from datetime import datetime
start_time = datetime.now()


# In[ ]:


class arg_classifier():
    def __init__(self, language='ko', version=1.1):
        self.language = language
        self.version = version
        if self.language == 'en':
            self.framenet = 'fn'+str(version)
        elif self.language == 'ko':
            self.framenet = 'kfn'+str(version)            
        print('### SETINGS')
        print('\t# FrameNet:', self.framenet)        
        if self.language == 'ko':
            kfn = koreanframenet.interface(version=version)
            self.trn, self.dev, self.tst = kfn.load_data()            
        try:
            target_dir = os.path.dirname(os.path.abspath( __file__ ))
        except:
            target_dir = '.'
        data_path = target_dir+'/../koreanframenet/resource/info/'
        with open(data_path+self.framenet+'_lu2idx.json','r') as f:
            self.lu2idx = json.load(f)
        with open(data_path+'fn1.7_frame2idx.json','r') as f:
            self.frame2idx = json.load(f)      
        with open(data_path+self.framenet+'_lufrmap.json','r') as f:
            self.lufrmap = json.load(f)
        with open(data_path+'fn1.7_fe2idx.json','r') as f:
            self.arg2idx = json.load(f)
        self.idx2frame = dict(zip(self.frame2idx.values(),self.frame2idx.keys()))
        self.idx2lu = dict(zip(self.lu2idx.values(),self.lu2idx.keys()))
        self.idx2arg = dict(zip(self.arg2idx.values(),self.arg2idx.keys()))
        
        with open(data_path+'fn1.7_frargmap.json','r') as f:
            self.frargmap = json.load(f)
            
    def flat_accuracy(self, preds, labels):
        pred_flat = np.argmax(preds, axis=2).flatten()
        labels_flat = labels.flatten()
        return np.sum(pred_flat == labels_flat) / len(labels_flat)
            
    def data2argdata(self, data):
        result = []
        for i in data:
            tokens, lus, frames, args = i[0],i[1],i[2],i[3]
            for idx in range(len(args)):
                arg_tag = args[idx]
                if arg_tag.startswith('B'):
                    new_args = ['O' for i in range(len(tokens))]                
                    fe_tag = arg_tag.split('-')[1]
                    next_idx = idx + 1
                    while next_idx < len(args) and args[next_idx] == 'I-'+fe_tag:
                        next_idx +=1
                    new_args[next_idx-1] = fe_tag
                    new_sent = []
                    new_sent.append(tokens)
                    new_sent.append(lus)
                    new_sent.append(frames)
                    new_sent.append(new_args)
                    result.append(new_sent)
        return result
    
    def gen_bert_input_representation(self, fn_data, MAX_LEN=256, batch_size=8):
        bert_io = dataio.for_BERT(mode='training', version=self.version)
        data = bert_io.convert_to_bert_input_arg_classifier(fn_data)
        sampler = RandomSampler(fn_data)
        dataloader = DataLoader(data, sampler=sampler, batch_size=batch_size)        
        return data, sampler, dataloader
        
    def train(self, model_dir='.', trn=False, dev=False, MAX_LEN = 256, batch_size = 8, epoch=4):
        model_path = model_dir+'/'+self.framenet+'-arg_classifier.pt'
        print('your model would be saved at', model_path)
        
        # load BERT model for arg-classification
        model = BertForArgClassification.from_pretrained("bert-base-multilingual-cased", num_labels = len(self.arg2idx), num_lus = len(self.lu2idx), num_frames = len(self.frame2idx), ludim = 64, framedim = 100, frargmap=self.frargmap)
        model.cuda();
        
        # trn to arg-granularity data
        trn = arg_classifier.data2argdata(self, trn)
        
        # gen BERT input representations            
        trn_data, trn_sampler, trn_dataloader = arg_classifier.gen_bert_input_representation(self, trn, MAX_LEN=256, batch_size=8)
        
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
        epochs = epoch
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
                b_input_ids, b_input_tgt_idxs, b_input_lus, b_input_frames, b_input_arg_idxs, b_input_args, b_input_masks = batch            
                # forward pass
                loss = model(b_input_ids, token_type_ids=None, tgt_idxs=b_input_tgt_idxs, 
                             lus=b_input_lus, frames=b_input_frames, arg_idxs=b_input_arg_idxs, args=b_input_args, attention_mask=b_input_masks)
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

            # print train loss per epoch
            print("Train loss: {}".format(tr_loss/nb_tr_steps))
            torch.save(model, model_path)
            num_of_epoch += 1
        print('...training is done')
        end_time = datetime.now()
        running_time = 'running_ttime:'+str(end_time - start_time)
        print('running_time:', running_time)          
        
    def test(self, tst=False, model_dir='.', MAX_LEN = 256, batch_size = 8):
        model_path = model_dir+'/'+self.framenet+'-arg_classifier.pt'
        print('your model is', model_path)
        model = torch.load(model_path)
        
        model.eval()
        
        # trn to arg-granularity data
        tst = arg_classifier.data2argdata(self, tst)
        
        # gen BERT input representations            
        tst_data, tst_sampler, tst_dataloader = arg_classifier.gen_bert_input_representation(self, tst, MAX_LEN=256, batch_size=8)
        
        
        eval_loss, eval_accuracy = 0, 0
        nb_eval_steps, nb_eval_examples = 0, 0
        predictions , true_labels, scores, candis, all_frames = [], [], [], [], []
        for batch in tst_dataloader:
            batch = tuple(t.to(device) for t in batch)
            b_input_ids, b_tgt_idxs, b_lus, b_frames, b_arg_idxs, b_args, b_masks = batch

            with torch.no_grad():
                tmp_eval_loss = model(b_input_ids, token_type_ids=None, tgt_idxs=b_tgt_idxs, 
                         lus=b_lus, frames=b_frames, arg_idxs=b_arg_idxs, attention_mask=b_masks)
                logits = model(b_input_ids, token_type_ids=None, tgt_idxs=b_tgt_idxs, 
                         lus=b_lus, frames=b_frames, arg_idxs=b_arg_idxs, attention_mask=b_masks)
            logits = logits.detach().cpu().numpy()
            label_ids = b_args.to('cpu').numpy()          
            masks = dataio.get_masks(b_frames, self.frargmap, num_label=len(self.arg2idx)).to(device)
            for frame in b_frames:
                candi_idx = self.frargmap[str(int(frame))]
                candi = [self.idx2arg[c] for c in candi_idx]
                candi_txt = ','.join(candi)
                candi_txt = str(len(candi))+'\t'+candi_txt
                candis.append(candi_txt)
                all_frames.append(self.idx2frame[int(frame)])
            
            for b_idx in range(len(logits)):
                logit = logits[b_idx]
                mask = masks[b_idx]
                b_pred_idxs, b_pred_logits = [],[]
                for fe_idx in range(len(mask)):
                    if mask[fe_idx] > 0:
                        b_pred_idxs.append(fe_idx)
                        b_pred_logits.append(logit[0][fe_idx].item())
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
            tmp_eval_accuracy = arg_classifier.flat_accuracy(self, logits, label_ids)
            eval_loss += tmp_eval_loss.mean().item()
            eval_accuracy += tmp_eval_accuracy
            nb_eval_examples += b_input_ids.size(0)
            nb_eval_steps += 1
            
        pred_tags = [self.idx2arg[p_i] for p in predictions for p_i in p]
        valid_tags = [self.idx2arg[l_ii] for l in true_labels for l_i in l for l_ii in l_i]        
        
        acc = accuracy_score(pred_tags, valid_tags)
        print("Accuracy: {}".format(accuracy_score(pred_tags, valid_tags)))
        
        return acc

