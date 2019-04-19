
# coding: utf-8

# In[17]:


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
from KAIST_frame_parser.koreanframenet import koreanframenet
from sklearn.metrics import accuracy_score

from datetime import datetime
start_time = datetime.now()


# In[16]:


class frameid():
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
        self.idx2frame = dict(zip(self.frame2idx.values(),self.frame2idx.keys()))
        self.idx2lu = dict(zip(self.lu2idx.values(),self.lu2idx.keys()))
        
    def flat_accuracy(self, preds, labels):
        pred_flat = np.argmax(preds, axis=2).flatten()
        labels_flat = labels.flatten()
        return np.sum(pred_flat == labels_flat) / len(labels_flat)
        
    def gen_bert_input_representation(self, fn_data, MAX_LEN=256, batch_size=8):
        bert_io = dataio.for_BERT(mode='training', version=self.version)
        data = bert_io.convert_to_bert_input_frameid(fn_data)
        sampler = RandomSampler(fn_data)
        dataloader = DataLoader(data, sampler=sampler, batch_size=batch_size)        
        return data, sampler, dataloader
    
    def train(self, model_dir='.', trn=False, dev=False, MAX_LEN = 256, batch_size = 8, epoch=4):
        model_path = model_dir+'/'+self.framenet+'-frameid-'+str(self.version)+'.pt'
        print('your model would be saved at', model_path)
        # load BERT model for frameid
        model = BertForFrameIdentification.from_pretrained("bert-base-multilingual-cased", num_labels = len(self.frame2idx), num_lus = len(self.lu2idx), ludim = 64, lufrmap=self.lufrmap)
        model.cuda();
            
        # gen BERT input representations            
        trn_data, trn_sampler, trn_dataloader = frameid.gen_bert_input_representation(self, trn, MAX_LEN=256, batch_size=8)
        #dev_data, dev_sampler, dev_data_loader = frameid.gen_bert_input_representation(dev)
            
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
                b_input_ids, b_input_tgt_idxs, b_input_lus, b_input_frames, b_input_masks = batch            
                # forward pass
                loss = model(b_input_ids, token_type_ids=None, tgt_idxs=b_input_tgt_idxs, 
                             lus=b_input_lus, frames=b_input_frames, attention_mask=b_input_masks)
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
        
    def test(self, tst=False, model_dir='.', MAX_LEN = 256, batch_size = 8):
        model_path = model_dir+'/'+self.framenet+'-frameid-'+str(self.version)+'.pt'
        print('your model is', model_path)
        model = torch.load(model_path)
        
        model.eval()
        tst_data, tst_sampler, tst_dataloader = frameid.gen_bert_input_representation(self, tst, MAX_LEN=256, batch_size=8)
        
        eval_loss, eval_accuracy = 0, 0
        nb_eval_steps, nb_eval_examples = 0, 0
        predictions , true_labels, scores, candis, all_lus = [], [], [], [], []
        for batch in tst_dataloader:
            batch = tuple(t.to(device) for t in batch)
            b_input_ids, b_tgt_idxs, b_lus, b_frames, b_masks = batch

            with torch.no_grad():
                tmp_eval_loss = model(b_input_ids, token_type_ids=None, tgt_idxs=b_tgt_idxs, 
                                     lus=b_lus, frames=b_frames, attention_mask=b_masks)
                logits = model(b_input_ids, token_type_ids=None, tgt_idxs=b_tgt_idxs, 
                                lus=b_lus, attention_mask=b_masks)
            logits = logits.detach().cpu().numpy()
            label_ids = b_frames.to('cpu').numpy()          
            masks = dataio.get_masks(b_lus, self.lufrmap, num_label=len(self.frame2idx)).to(device)
            for lu in b_lus:
                candi_idx = self.lufrmap[str(int(lu))]
                candi = [self.idx2frame[c] for c in candi_idx]
                candi_txt = ','.join(candi)
                candi_txt = str(len(candi))+'\t'+candi_txt
                candis.append(candi_txt)
                all_lus.append(self.idx2lu[int(lu)])
            
            for b_idx in range(len(logits)):
                logit = logits[b_idx]
                mask = masks[b_idx]
                b_pred_idxs, b_pred_logits = [],[]
                for fr_idx in range(len(mask)):
                    if mask[fr_idx] > 0:
                        b_pred_idxs.append(fr_idx)
                        b_pred_logits.append(logit[0][fr_idx].item())
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
            tmp_eval_accuracy = frameid.flat_accuracy(self, logits, label_ids)
            eval_loss += tmp_eval_loss.mean().item()
            eval_accuracy += tmp_eval_accuracy
            nb_eval_examples += b_input_ids.size(0)
            nb_eval_steps += 1
        pred_tags = [self.idx2frame[p_i] for p in predictions for p_i in p]
        valid_tags = [self.idx2frame[l_ii] for l in true_labels for l_i in l for l_ii in l_i]
        acc = accuracy_score(pred_tags, valid_tags)
        print("Accuracy: {}".format(accuracy_score(pred_tags, valid_tags)))
        
        return acc


# In[ ]:


# # evaluation for each epoch
#         model.eval()
#         eval_loss, eval_accuracy = 0, 0
#         nb_eval_steps, nb_eval_examples = 0, 0
#         predictions , true_labels, scores, candis, all_lus = [], [], [], [], []
#         for batch in tst_dataloader:
#             batch = tuple(t.to(device) for t in batch)
#             b_input_ids, b_tgt_idxs, b_lus, b_senses, b_masks = batch

#             with torch.no_grad():
#                 tmp_eval_loss = model(b_input_ids, token_type_ids=None, tgt_idxs=b_tgt_idxs, 
#                                      lus=b_lus, senses=b_senses, attention_mask=b_masks)
#                 logits = model(b_input_ids, token_type_ids=None, tgt_idxs=b_tgt_idxs, 
#                                 lus=b_lus, attention_mask=b_masks)
#             logits = logits.detach().cpu().numpy()
#             label_ids = b_senses.to('cpu').numpy()          
#             masks = dataio.get_masks(b_lus, lusensemap, num_label=len(sense2idx)).to(device)
#             for lu in b_lus:
#                 candi_idx = lusensemap[str(int(lu))]
#                 candi = [idx2sense[c] for c in candi_idx]
#                 candi_txt = ','.join(candi)
#                 candi_txt = str(len(candi))+'\t'+candi_txt
#                 candis.append(candi_txt)
#                 all_lus.append(idx2lu[int(lu)])
            
#             for b_idx in range(len(logits)):
#                 logit = logits[b_idx]
#                 mask = masks[b_idx]
#                 b_pred_idxs, b_pred_logits = [],[]
#                 for fr_idx in range(len(mask)):
#                     if mask[fr_idx] > 0:
#                         b_pred_idxs.append(fr_idx)
#                         b_pred_logits.append(logit[0][fr_idx].item())
#                 b_pred_idxs = torch.tensor(b_pred_idxs)
#                 b_pred_logits = torch.tensor(b_pred_logits)
#                 sm = nn.Softmax()
#                 b_pred_logits = sm(b_pred_logits).view(1, -1)
#                 score, indice = b_pred_logits.max(1)                
#                 prediction = b_pred_idxs[indice]
#                 predictions.append([int(prediction)])
#                 score = float(score)
#                 scores.append(score)
#             true_labels.append(label_ids)
#             tmp_eval_accuracy = flat_accuracy(logits, label_ids)
#             eval_loss += tmp_eval_loss.mean().item()
#             eval_accuracy += tmp_eval_accuracy
#             nb_eval_examples += b_input_ids.size(0)
#             nb_eval_steps += 1
            
#         eval_loss = eval_loss/nb_eval_steps
#         print("Validation loss: {}".format(eval_loss))
#         print("Validation Accuracy: {}".format(eval_accuracy/nb_eval_steps))
#         pred_tags = [idx2sense[p_i] for p in predictions for p_i in p]
#         valid_tags = [idx2sense[l_ii] for l in true_labels for l_i in l for l_ii in l_i]
#         acc = accuracy_score(pred_tags, valid_tags)
#         accuracy_result.append(acc)
#         print("Accuracy: {}".format(accuracy_score(pred_tags, valid_tags)))
        
#         result_path = result_dir+str(version)+'.frameid-'+str(num_of_epoch)+'.tsv'
#         with open(result_path,'w') as f:
#             line = 'gold' + '\t' + 'prediction' + '\t' + 'score' + '\t' + 'input_lu' + '\t' + 'sense_candidates'
#             f.write(line+'\n')
#             for item in range(len(pred_tags)):
#                 line = valid_tags[item] + '\t' + pred_tags[item] + '\t' + str(scores[item]) +'\t'+ all_lus[item]+'\t' + candis[item]
#                 f.write(line+'\n')
#     accuracy_result_path = result_dir+str(version)+'.frameid.accuracy'
#     with open(accuracy_result_path,'w') as f:
#         n = 0
#         for acc in accuracy_result:
#             f.write('epoch:'+str(n)+'\t' + 'accuracy: '+str(acc)+'\n')
#             n +=1

