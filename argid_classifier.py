
# coding: utf-8

# In[1]:


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
from KAIST_frame_parser.src.fn_modeling import BertForArgClassification

from datetime import datetime
start_time = datetime.now()


# In[2]:


MAX_LEN = 256
batch_size = 8
language = 'ko'
version = 1.1

global_frargmap = False

if global_frargmap == True:
    frarg_type = 'global-frargmap-'
else:
    frarg_type = 'local-frargmap-'

framenet = 'kfn'
framenet_data = 'Korean FrameNet '+str(version)

# save your model to
model_dir = './models/'+framenet+'/'
result_dir = './result/'

print('### SETINGS')
print('\t# FrameNet:', framenet_data)
print('\t# model will be saved to', model_dir)
print('\t# result will be saved to', result_dir)


# # Load Data

# In[3]:


from koreanframenet import koreanframenet
if language == 'ko':
    kfn = koreanframenet.interface(version=version)
    trn, dev, tst = kfn.load_data()


# # gen arg-granularity data

# In[4]:


def data2argdata(data):
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
    
trn = data2argdata(trn)
dev = data2argdata(dev)
tst = data2argdata(tst)
print('# of examples in trn:', len(trn))
print('# of examples in dev:', len(dev))
print('# of examples in tst:', len(tst))


# In[5]:


print('\tdata example')
print('\ttrn[0]')
print(trn[0])
print('\n\ttrn[0]')
print(trn[1])


# In[6]:


data_path = './koreanframenet/resource/info/'

with open(data_path+framenet+str(version)+'_lu2idx.json','r') as f:
    lu2idx = json.load(f)
with open(data_path+'fn1.7_frame2idx.json','r') as f:
    frame2idx = json.load(f)
with open(data_path+'fn1.7_fe2idx.json','r') as f:
    arg2idx = json.load(f)
with open(data_path+framenet+str(version)+'_lufrmap.json','r') as f:
    lufrmap = json.load(f)
    
if global_frargmap == True:
    with open(data_path+'fn1.7_frargmap.json','r') as f:
        frargmap = json.load(f)
else:
    with open(data_path+'kfn1.1_frargmap.json','r') as f:
        frargmap = json.load(f)
    
idx2frame = dict(zip(frame2idx.values(),frame2idx.keys()))
idx2lu = dict(zip(lu2idx.values(),lu2idx.keys()))
idx2arg = dict(zip(arg2idx.values(),arg2idx.keys()))
        
print('\nData Statistics...')
print('\t# of lu:', len(lu2idx))


# # Load BERT tokenizer

# In[7]:


# # load pretrained BERT tokenizer
# tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased', do_lower_case=False)

# # bert tokenizer
# def bert_tokenizer(text):
#     orig_tokens = text.split(' ')
#     bert_tokens = []
#     orig_to_tok_map = []
#     bert_tokens.append("[CLS]")
#     for orig_token in orig_tokens:
#         orig_to_tok_map.append(len(bert_tokens))
#         bert_tokens.extend(tokenizer.tokenize(orig_token))
#     bert_tokens.append("[SEP]")
    
#     return orig_tokens, bert_tokens, orig_to_tok_map


# # Generate BERT input representation

# In[8]:


print('generate BERT input representation ...')
bert_io = dataio.for_BERT(mode='training', version=version)

trn_data = bert_io.convert_to_bert_input_arg_classifier(trn)
trn_sampler = RandomSampler(trn_data)
trn_dataloader = DataLoader(trn_data, sampler=trn_sampler, batch_size=batch_size)

dev_data = bert_io.convert_to_bert_input_arg_classifier(dev)
dev_sampler = RandomSampler(dev_data)
dev_dataloader = DataLoader(dev_data, sampler=dev_sampler, batch_size=batch_size)

tst_data = bert_io.convert_to_bert_input_arg_classifier(tst)
tst_sampler = RandomSampler(tst_data)
tst_dataloader = DataLoader(tst_data, sampler=tst_sampler, batch_size=batch_size)
print('... is done')


# In[9]:


# # print(len(trn_data[0]))
# idx = 3
# trn_inputs, trn_tgt_idxs, trn_lus, trn_frames, trn_arg_idxs, trn_args, trn_masks = trn_data[idx]


# print(trn[idx])
# print('\ntrn_inputs')
# print(trn_inputs)
# print('\ntgt_idxs')
# print(trn_tgt_idxs)
# print('\nlus')
# print(trn_lus)
# print('\nframes')
# print(trn_frames)
# print(idx2frame[int(trn_frames[0])])
# print('\narg_idxs')
# print(trn_arg_idxs)
# print('\nargs')
# print(trn_args)
# print(idx2arg[int(trn_args[0])])
# print('')


# # Load BERT arg-classifier model

# In[10]:


model = BertForArgClassification.from_pretrained("bert-base-multilingual-cased", num_labels = len(arg2idx), num_lus = len(lu2idx), num_frames = len(frame2idx), ludim = 64, framedim = 100, frargmap=frargmap)
model.cuda();


# In[11]:


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


# In[12]:


# Evaluation
from sklearn.metrics import accuracy_score

def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=2).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)


# # TRAINING the pretrained BERT LM

# In[13]:


def training():    
    epochs = 7
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
            
#             break

        # print train loss per epoch
        print("Train loss: {}".format(tr_loss/nb_tr_steps))
        num_of_epoch += 1
        model_path = model_dir+frarg_type+'arg_classifier-epoch-'+str(num_of_epoch)+'.pt'
        torch.save(model, model_path)

        # evaluation for each epoch
        model.eval()
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
            masks = dataio.get_masks(b_frames, frargmap, num_label=len(arg2idx)).to(device)
            for frame in b_frames:
                candi_idx = frargmap[str(int(frame))]
                candi = [idx2arg[c] for c in candi_idx]
                candi_txt = ','.join(candi)
                candi_txt = str(len(candi))+'\t'+candi_txt
                candis.append(candi_txt)
                all_frames.append(idx2frame[int(frame)])
            
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
            tmp_eval_accuracy = flat_accuracy(logits, label_ids)
            eval_loss += tmp_eval_loss.mean().item()
            eval_accuracy += tmp_eval_accuracy
            nb_eval_examples += b_input_ids.size(0)
            nb_eval_steps += 1
            
#             break
            
        eval_loss = eval_loss/nb_eval_steps
        print("Validation loss: {}".format(eval_loss))
        print("Validation Accuracy: {}".format(eval_accuracy/nb_eval_steps))
        pred_tags = [idx2arg[p_i] for p in predictions for p_i in p]
        valid_tags = [idx2arg[l_ii] for l in true_labels for l_i in l for l_ii in l_i]        
        
        acc = accuracy_score(pred_tags, valid_tags)
        accuracy_result.append(acc)
        print("Accuracy: {}".format(accuracy_score(pred_tags, valid_tags)))
        
        result_path = result_dir+frarg_type+str(version)+'.arg-classifier-'+str(num_of_epoch)+'.tsv'
        with open(result_path,'w') as f:
            line = 'gold' + '\t' + 'prediction' + '\t' + 'score' + '\t' + 'input_frame' + '\t' + 'sense_candidates'
            f.write(line+'\n')
            for item in range(len(pred_tags)):
                line = valid_tags[item] + '\t' + pred_tags[item] + '\t' + str(scores[item]) +'\t'+ all_frames[item]+'\t' + candis[item]
                f.write(line+'\n')
        
    accuracy_result_path = result_dir+frarg_type+str(version)+'.arg-classifier.accuracy'
    with open(accuracy_result_path,'w') as f:
        end_time = datetime.now()
        running_time = 'running_ttime:'+str(end_time - start_time)
        f.write(running_time+'\n')
        n = 0
        for acc in accuracy_result:
            f.write('epoch:'+str(n)+'\t' + 'accuracy: '+str(acc)+'\n')
            n +=1

training()

