
# coding: utf-8

# In[1]:


import json
from src import dataio, etri
import targetid
import torch
from torch import nn
import os
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
n_gpu = torch.cuda.device_count()

from koreanframenet.src import conll2textae

from konlpy.tag import Kkma
from pprint import pprint


# # SETTINGS

# In[2]:


try:
    dir_path = os.path.dirname( os.path.abspath( __file__ ))
except:
#     dir_path = '.'
    dir_path = '/disk_4/resource'

version = 1.1
frameid_model_path = dir_path+'/models/kfn/frameid-'+str(version)+'.pt'
arg_classifier_model_path = dir_path+'/models/kfn/arg_classifier-'+str(version)+'.pt'
framenet = 'kfn'
framenet_data = 'Korean FrameNet '+str(version)


# In[3]:


data_path = './koreanframenet/resource/info/'

with open(data_path+framenet+str(version)+'_lu2idx.json','r') as f:
    lu2idx = json.load(f)
with open(data_path+'fn1.7_frame2idx.json','r') as f:
    frame2idx = json.load(f)
with open(data_path+'fn1.7_fe2idx.json','r') as f:
    arg2idx = json.load(f)
with open(data_path+framenet+str(version)+'_lufrmap.json','r') as f:
    lufrmap = json.load(f)
    
idx2frame = dict(zip(frame2idx.values(),frame2idx.keys()))
idx2lu = dict(zip(lu2idx.values(),lu2idx.keys()))
idx2arg = dict(zip(arg2idx.values(),arg2idx.keys()))


# In[4]:


bert_io = dataio.for_BERT(mode='parser', version=version)


def convert_word_idx_2_token_idx_for_arg(bert_token_tuples, word_idx):
    pass


# In[5]:


class models():
    def __init__(self, mode='parser', version=1.0):
        self.version = version
        self.mode = mode
        self.bert_io = dataio.for_BERT(mode=self.mode, version=self.version)
        self.frameid_model = torch.load(frameid_model_path)
        self.arg_classifier_model = torch.load(arg_classifier_model_path)
        
    def frame_identifier(self, tgt_data):
        bert_inputs = self.bert_io.convert_to_bert_input_frameid(tgt_data)
        dataloader = DataLoader(bert_inputs, sampler=None, batch_size=1)
        
        predictions, scores, candis = [], [], []
        for batch in dataloader:
            batch = tuple(t.to(device) for t in batch)
            b_input_ids, b_tgt_idxs, b_lus, b_masks = batch
            with torch.no_grad():
                logits = self.frameid_model(b_input_ids, token_type_ids=None, tgt_idxs=b_tgt_idxs, 
                                lus=b_lus, attention_mask=b_masks)
            logits = logits.detach().cpu().numpy()                
            masks = self.bert_io.get_masks(b_lus, model='frameid').to(device)
            
            for lu in b_lus:
                candi_idx = lufrmap[str(int(lu))]
                candi = [idx2frame[c] for c in candi_idx]
                candi_txt = ','.join(candi)
                candi_txt = str(len(candi))+'\t'+candi_txt
                candis.append(candi_txt)
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
        pred_tags = self.bert_io.idx2tag(predictions, model='frameid')       
        conll, tuples = [],[]
        for i in range(len(tgt_data)):
            instance = tgt_data[i]
            tokens, targets = instance[0], instance[1]
            frames = ['_' for i in range(len(targets))]
            for t in range(len(targets)):
                if targets[t] != '_':
                    frames[t] = pred_tags[i]
            instance.append(frames)
            conll.append(instance)
            
            tup = (pred_tags[i], scores[i])
            tuples.append(tup)
            
        return conll, tuples
    
    def arg_classifier(self, fr_data, arg_tokens):
        
        result = []
        
        text = ' '.join(fr_data[0][0])
        orig_tokens, bert_tokens, orig_to_tok_map = bert_io.bert_tokenizer_assign_to_last_token(text)     
                
        bert_inputs = self.bert_io.convert_to_bert_input_arg_classifier(fr_data)
        dataloader = DataLoader(bert_inputs, sampler=None, batch_size=1)
        predictions, scores = [],[]
        for batch in dataloader:
            batch = tuple(t.to(device) for t in batch)
            b_input_ids, b_tgt_idxs, b_lus, b_frames, b_masks = batch
            
            arg_word_idx = arg_tokens[-1]
            arg_idx = [[orig_to_tok_map[arg_word_idx]]]
            b_arg_idxs = torch.tensor(arg_idx)
            
            with torch.no_grad():
                logits = self.arg_classifier_model(b_input_ids, token_type_ids=None, tgt_idxs=b_tgt_idxs, 
                         lus=b_lus, frames=b_frames, arg_idxs=b_arg_idxs, attention_mask=b_masks)
            logits = logits.detach().cpu().numpy()      
            masks = self.bert_io.get_masks(b_frames, model='argid').to(device)
            
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
            pred_tags = self.bert_io.idx2tag(predictions, model='argid')
            result.append(pred_tags)
        return result


# In[6]:


fn_models = models(mode='parser', version=version)


# In[7]:


kkma = Kkma()
def doc2sents(text):
    result = []
    n = 0
    sents = text.split('. ')
    for sent in sents:
        if len(sent) >0:
            sent = sent+'.'
            stringuri = 'test_0_'+str(n)
            tup = (sent, stringuri)
            result.append(tup)
            n +=1
    return result


# In[8]:


def result2triples(text, conll, tuples, stringuri):
    triples = []
    triple = (str(stringuri), 'nif:isString', text)
    triples.append(triple)
    # for target_id
    if len(conll) > 0:
        for i in range(len(conll)):
            instance = conll[i]
            tokens, targets, frames, args = instance[0], instance[1], instance[2], instance[3]
            for tok in range(len(tokens)):
                if frames[tok] != '_':
                    frame = 'frame:'+frames[tok]
                    lu = targets[tok]
            triple = (frame, 'frdf:provinence', str(stringuri))
            triples.append(triple)
            triple = (frame, 'frdf:lu', lu)
            triples.append(triple)
            triple = (frame, 'frdf:score', str(tuples[i][1]))
            triples.append(triple)
            
            #args to triples
            for idx in range(len(args)):
                arg_tag = args[idx]
                arg_tokens = []
                if arg_tag.startswith('B'):
                    fe_tag = arg_tag.split('-')[1]
                    arg_tokens.append(tokens[idx])
                    next_idx = idx + 1
                    while next_idx < len(args) and args[next_idx] == 'I-'+fe_tag:
                        arg_tokens.append(tokens[next_idx])
                        next_idx +=1
                            
                    arg_text = ' '.join(arg_tokens)
                    triple = (frame, 'arg:'+fe_tag, arg_text)
                    triples.append(triple)
                        
    return triples


# In[9]:


def sent2pa(text):
    nlp = etri.getETRI(text)    
    conll_2009 = etri.getETRI_CoNLL2009(nlp)
    predicate_argument = etri.phrase_parser(conll_2009, nlp)
    return predicate_argument


# In[10]:


def arg_identifier(conll):
    text = ' '.join(conll[0][0])
    pas = sent2pa(text)
    
    result = []
    for anno in conll:
        args = ['O' for i in range(len(anno[1]))]
        for idx in range(len(anno[1])):
            if anno[1][idx] != '_':
                lu = anno[1][idx]
                frame = anno[2][idx]
                target_idx = idx
        for pa in pas:
            if target_idx == pa['predicate']['id']:                
                for argument in pa['arguments']:
                    arg_tokens = argument['tokens']
                    
                    pred = fn_models.arg_classifier([anno], arg_tokens)[0][0]
                    for arg_token_idx in range(len(arg_tokens)):
                        arg_token = arg_tokens[arg_token_idx]
                        if arg_token_idx == 0:
                            fe = 'B-'+pred
                        else:
                            fe = 'I-'+pred
                        args[arg_token] = fe                
                    
        new_anno = anno
        new_anno.append(args)
        result.append(new_anno)
    return result


# In[11]:


def main(data, sentence_id):
    input_data = dataio.preprocessor(data)
    text = data
    
    tgt_data = targetid.baseline(input_data)
    fid_data, fid_result  = fn_models.frame_identifier(tgt_data)
    argid_data = arg_identifier(fid_data)   
    
    framegraph = result2triples(text, argid_data, fid_result, sentence_id)
    textae = conll2textae.get_textae(argid_data)
        
    result = {}
    result['graph'] = framegraph
    result['textae'] = textae
    return result


# In[12]:


# text = '헤밍웨이는 1899년 7월 21일 미국 일리노이에서 태어났고, 62세에 자살로 사망했다.'
# stringuri = 'test:offset_0_53'
# parsed = main(text, stringuri)
# pprint(parsed['graph'])


# In[13]:


# text = '미중 무역전쟁을 벌이는 중국이 이번에는 브라질산 설탕 수입을 제한하면서 브라질과 갈등을 빚고 있다.'
# text = '바둑 인공지능(AI) 알파고에 도전장을 던진 중국랭킹 1위 커제 9단이 초반부터 극단적인 실리작전을 들고 나왔다.'
# text = '마이크로소프트의 공동 창업자로 억만장자 반열에 오른 빌 게이츠가 20대 젊은이들에게 추천하는 유망 분야로 인공지능(AI)과 에너지, 생명공학을 꼽았다.'
# text = '반면 안철수 바른미래당 대표가 만든 안랩(053800)은 전날 대비 0.88% 하락한 7만8400원에 거래 중이다.'
# text = '헤밍웨이는 1899년 7월 21일 일리노이주에서 태어났고, 62세에 자살로 사망했다.'
# text = '태풍 Hugo가 남긴 피해들과 회사 내 몇몇 주요 부서들의 저조한 실적들을 반영하여 Aetna Life and Caualty Co.의 3분기 순이익이 182.6 백만 달러 또는 주당 1.63 달러로 22 % 하락하였다.'
# stringuri = 'test:offset_0_53'
# parsed = main(text, stringuri)
# pprint(parsed['graph'])


# In[15]:


# text = '어니스트 헤밍웨이는 미국의 소설가이자 저널리스트이다. 1854년 노벨 문학상을 수상하였다. 헤밍웨이는 1899년 7월 21일 일리노이주에서 태어났다. 헤밍웨이는 풀린 파이퍼와 이혼한 뒤 마사 겔혼과 재혼하였다. 헤밍웨이는 1961년 아이다호 주에서 62세의 나이에 자살했다.'
# sents = doc2sents(text)
# framegraphs = []
# for text, stringuri in sents:
#     parsed = main(text, stringuri)
#     pprint(parsed['graph'])
#     print('')
# #     framegraphs+=(parsed['framegraph'])
# pprint(framegraphs)

