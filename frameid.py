
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
torch.cuda.get_device_name(0)

from KAIST_frame_parser.src import dataio
from KAIST_frame_parser.src.fn_modeling import BertForFrameIdentification


# # BASIC SETTINGS

# In[8]:


MAX_LEN = 256
batch_size = 8
language = 'ko'
version = 1.1

if language == 'en':
    framenet = 'fn'
    framenet_data = 'English FrameNet '+str(version)
elif language == 'ko':
    framenet = 'kfn'
    framenet_data = 'Korean FrameNet '+str(version)

# save your model to
model_dir = './models/'+framenet+'/'
result_dir = './result/'

print('### SETINGS')
print('\t# FrameNet:', framenet_data)
print('\t# model will be saved to', model_dir)
print('\t# result will be saved to', result_dir)


# # LOAD DATA

# In[9]:


from koreanframenet import koreanframenet
if language == 'ko':
    kfn = koreanframenet.interface(version=version)
    trn, dev, tst = kfn.load_data()
    
# print('\nan example of dataset')
# print(trn[0])


# In[10]:


data_path = './koreanframenet/resource/info/'

with open(data_path+framenet+str(version)+'_lu2idx.json','r') as f:
    lu2idx = json.load(f)
with open(data_path+'fn1.7_frame2idx.json','r') as f:
    sense2idx = json.load(f)      
with open(data_path+framenet+str(version)+'_lufrmap.json','r') as f:
    lusensemap = json.load(f)
    
idx2sense = dict(zip(sense2idx.values(),sense2idx.keys()))
idx2lu = dict(zip(lu2idx.values(),lu2idx.keys()))
        
print('\nData Statistics...')
print('\t# of lu:', len(lu2idx))


# # LOAD BERT TOKENIZER

# In[11]:


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


# # GENERATE BERT input representations

# In[12]:


def gen_data(input_data):
    tokenized_texts, lus, senses = [],[],[]

    for i in range(len(input_data)):    
        data = input_data[i]
        text = ' '.join(data[0])
        orig_tokens, bert_tokens, orig_to_tok_map = bert_tokenizer(text)
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

    input_ids = pad_sequences([tokenizer.convert_tokens_to_ids(txt) for txt in tokenized_texts],
                          maxlen=MAX_LEN, dtype="long", truncating="post", padding="post")
    
    tgt_seq, lu_seq, sense_seq = [],[],[]
    for sent_idx in range(len(lus)):
        lu_items = lus[sent_idx]
        sense_items = senses[sent_idx]
        tgt,lu, sense = [],[],[]
        for idx in range(len(lu_items)):
            if lu_items[idx] != '_':
                if len(tgt) == 0:
                    tgt.append(idx)
                    lu.append(lu2idx[lu_items[idx]])
        for idx in range(len(sense_items)):
            if sense_items[idx] != '_':
                if len(sense) == 0:
                    sense.append(sense2idx[sense_items[idx]])
        tgt_seq.append(tgt)
        lu_seq.append(lu)
        sense_seq.append(sense)
        
    attention_masks = [[float(i>0) for i in ii] for ii in input_ids]    
    data_inputs = torch.tensor(input_ids)
    data_tgt_idx = torch.tensor(tgt_seq)
    data_lus = torch.tensor(lu_seq)
    data_senses = torch.tensor(sense_seq)
    data_masks = torch.tensor(attention_masks)
    
    return data_inputs, data_tgt_idx, data_lus, data_senses, data_masks

trn_inputs, trn_tgt_idxs, trn_lus, trn_senses, trn_masks = gen_data(trn)
dev_inputs, dev_tgt_idxs, dev_lus, dev_senses, dev_masks = gen_data(dev)
tst_inputs, tst_tgt_idxs, tst_lus, tst_senses, tst_masks = gen_data(tst)


# In[15]:


print(trn[0])
print('')
print(trn_inputs[0])
print('')
print(trn_tgt_idxs[0])
print('')
print(trn_lus[0])
print('')
print(trn_senses[0])
print('')


# In[ ]:


trn_data = TensorDataset(trn_inputs, trn_tgt_idxs, trn_lus, trn_senses, trn_masks)
trn_sampler = RandomSampler(trn_data)
trn_dataloader = DataLoader(trn_data, sampler=trn_sampler, batch_size=batch_size)

dev_data = TensorDataset(dev_inputs, dev_tgt_idxs, dev_lus, dev_senses, dev_masks)
dev_sampler = RandomSampler(dev_data)
dev_dataloader = DataLoader(dev_data, sampler=dev_sampler, batch_size=batch_size)

tst_data = TensorDataset(tst_inputs, tst_tgt_idxs, tst_lus, tst_senses, tst_masks)
tst_sampler = RandomSampler(tst_data)
tst_dataloader = DataLoader(tst_data, sampler=tst_sampler, batch_size=batch_size)


# # LOAD BERT framenet model

# In[ ]:


model = BertForFrameIdentification.from_pretrained("bert-base-multilingual-cased", num_labels = len(sense2idx), num_lus = len(lu2idx), ludim = 64, lusensemap=lusensemap)
model.cuda();


# In[ ]:


# optimizer
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


# In[ ]:


# Evaluation
from sklearn.metrics import accuracy_score

def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=2).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)


# # TRAINING the pretrained BERT language model

# In[ ]:


def training():    
    epochs = 5
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
            b_input_ids, b_input_tgt_idxs, b_input_lus, b_input_senses, b_input_masks = batch            
            # forward pass
            loss = model(b_input_ids, token_type_ids=None, tgt_idxs=b_input_tgt_idxs, 
                         lus=b_input_lus, senses=b_input_senses, attention_mask=b_input_masks)
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
        num_of_epoch += 1
        model_path = model_dir+'frame_identifier-epoch-'+str(num_of_epoch)+'.pt'
        torch.save(model, model_path)        

        # evaluation for each epoch
        model.eval()
        eval_loss, eval_accuracy = 0, 0
        nb_eval_steps, nb_eval_examples = 0, 0
        predictions , true_labels, scores, candis, all_lus = [], [], [], [], []
        for batch in tst_dataloader:
            batch = tuple(t.to(device) for t in batch)
            b_input_ids, b_tgt_idxs, b_lus, b_senses, b_masks = batch

            with torch.no_grad():
                tmp_eval_loss = model(b_input_ids, token_type_ids=None, tgt_idxs=b_tgt_idxs, 
                                     lus=b_lus, senses=b_senses, attention_mask=b_masks)
                logits = model(b_input_ids, token_type_ids=None, tgt_idxs=b_tgt_idxs, 
                                lus=b_lus, attention_mask=b_masks)
            logits = logits.detach().cpu().numpy()
            label_ids = b_senses.to('cpu').numpy()          
            masks = dataio.get_masks(b_lus, lusensemap, num_label=len(sense2idx)).to(device)
            for lu in b_lus:
                candi_idx = lusensemap[str(int(lu))]
                candi = [idx2sense[c] for c in candi_idx]
                candi_txt = ','.join(candi)
                candi_txt = str(len(candi))+'\t'+candi_txt
                candis.append(candi_txt)
                all_lus.append(idx2lu[int(lu)])
            
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
            tmp_eval_accuracy = flat_accuracy(logits, label_ids)
            eval_loss += tmp_eval_loss.mean().item()
            eval_accuracy += tmp_eval_accuracy
            nb_eval_examples += b_input_ids.size(0)
            nb_eval_steps += 1
            
        eval_loss = eval_loss/nb_eval_steps
        print("Validation loss: {}".format(eval_loss))
        print("Validation Accuracy: {}".format(eval_accuracy/nb_eval_steps))
        pred_tags = [idx2sense[p_i] for p in predictions for p_i in p]
        valid_tags = [idx2sense[l_ii] for l in true_labels for l_i in l for l_ii in l_i]
        acc = accuracy_score(pred_tags, valid_tags)
        accuracy_result.append(acc)
        print("Accuracy: {}".format(accuracy_score(pred_tags, valid_tags)))
        
        result_path = result_dir+str(version)+'.frameid-'+str(num_of_epoch)+'.tsv'
        with open(result_path,'w') as f:
            line = 'gold' + '\t' + 'prediction' + '\t' + 'score' + '\t' + 'input_lu' + '\t' + 'sense_candidates'
            f.write(line+'\n')
            for item in range(len(pred_tags)):
                line = valid_tags[item] + '\t' + pred_tags[item] + '\t' + str(scores[item]) +'\t'+ all_lus[item]+'\t' + candis[item]
                f.write(line+'\n')
    accuracy_result_path = result_dir+str(version)+'.frameid.accuracy'
    with open(accuracy_result_path,'w') as f:
        n = 0
        for acc in accuracy_result:
            f.write('epoch:'+str(n)+'\t' + 'accuracy: '+str(acc)+'\n')
            n +=1

training()


# In[ ]:


bert_io = dataio.for_BERT()

def frame_identifier(bert_inputs):
    data_inputs, data_tgt_idx, data_lus, data_senses, data_masks = bert_inputs[0],bert_inputs[1],bert_inputs[2],bert_inputs[3],bert_inputs[4]
    input_data = TensorDataset(data_inputs, data_tgt_idx, data_lus, data_senses, data_masks)
#     trn_sampler = RandomSampler(trn_data)
    trn_dataloader = DataLoader(trn_data, sampler=None, batch_size=batch_size)
    return trn_dataloader

