
# coding: utf-8

# In[1]:


import json
import sys
sys.path.append('../')

from KAIST_frame_parser.src import dataio, etri
from KAIST_frame_parser.src import targetid
import torch
from torch import nn
import os
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
n_gpu = torch.cuda.device_count()

from KAIST_frame_parser.koreanframenet.src import conll2textae

from konlpy.tag import Kkma
from pprint import pprint


# In[2]:


class models():
    def __init__(self, mode='parser', version=1.1, language='ko', model_dir=False):
        self.version = version
        self.mode = mode
        if language == 'en':
            self.framenet = 'fn'+str(version)
        elif language == 'ko':
            self.framenet = 'kfn'+str(version)
            
        if model_dir.endswith('/'):
            pass
        else:
            model_dir += '/'
        self.frameid_model_path = model_dir+self.framenet+'-frameid.pt'
        self.arg_classifier_model_path = model_dir+self.framenet+'-arg_classifier.pt'
        
        self.bert_io = dataio.for_BERT(mode=self.mode, version=self.version)
        self.frameid_model = torch.load(self.frameid_model_path, map_location=device)
        self.arg_classifier_model = torch.load(self.arg_classifier_model_path, map_location=device)
        
        
        
        try:
            target_dir = os.path.dirname(os.path.abspath( __file__ ))
        except:
            target_dir = '.'
        data_path = target_dir+'/koreanframenet/resource/info/'
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
                candi_idx = self.lufrmap[str(int(lu))]
                candi = [self.idx2frame[c] for c in candi_idx]
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
        orig_tokens, bert_tokens, orig_to_tok_map = self.bert_io.bert_tokenizer_assign_to_last_token(text)     
                
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
            masks = self.bert_io.get_masks(b_frames, model='argclassification').to(device)
            
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
            pred_tags = self.bert_io.idx2tag(predictions, model='argclassification')
            result.append(pred_tags)
        return result


# In[3]:


class SRLbasedParser():
    def __init__(self, version=1.1, language='ko', model_dir=False):
        try:
            config_dir = os.path.dirname( os.path.abspath( __file__ ))
        except:
            config_dir = '.'
        config_file = config_dir+'/config.json'
        with open(config_file, 'r') as f:
            config = json.load(f)
            
        self.version=version
        self.language=language
        self.model_dir=model_dir
                
        self.nlp_service = etri.etri(serviceType=config['nlp']['serviceType'], url=config['nlp']['url'], port=config['nlp']['port'])     
        self.fn_models = models(mode='parser', version=self.version, language=self.language, model_dir=self.model_dir)
        self.kkma = Kkma()

        print('### SETTINGS')
        print('\t# FrameNet:', self.fn_models.framenet)
        print('\t# PARSER:', config['parser'])
        print('\t# MODEL_PATH:', model_dir)
        print('\t# (your frameid model should be:', self.fn_models.frameid_model_path)
        print('\t# (your argid model should be:', self.fn_models.arg_classifier_model_path)
        
    def doc2sents(self, text):
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
    
    def result2triples(self, text, conll, tuples, stringuri):
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
    
    def arg_identifier(self, conll):
        text = ' '.join(conll[0][0])
        pas = SRLbasedParser.sent2pa(self, text)

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

                        pred = self.fn_models.arg_classifier([anno], arg_tokens)[0][0]
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
    
    def sent2pa(self, text):
        nlp = self.nlp_service.getETRI(text)    
        conll_2009 = self.nlp_service.getETRI_CoNLL2009(nlp)
        predicate_argument = self.nlp_service.phrase_parser(conll_2009, nlp)
        return predicate_argument
    
    def parser(self, data, sentence_id='stringurl:null'):
        input_data = dataio.preprocessor(data)
        text = data

        tgt_data = targetid.baseline(input_data)
        fid_data, fid_result  = self.fn_models.frame_identifier(tgt_data)
        if len(fid_data) > 0:
            argid_data = SRLbasedParser.arg_identifier(self, fid_data)
        else:
            argid_data = []

        framegraph = SRLbasedParser.result2triples(self, text, argid_data, fid_result, str(sentence_id))
        textae = conll2textae.get_textae(argid_data)

        result = {}
        result['graph'] = framegraph
        result['textae'] = textae
        result['conll'] = argid_data
        return result    


# In[4]:


def load_parser():
    language = 'ko'
    version = 1.1
    model_dir = '/disk_4/resource/models/'
    parser = SRLbasedParser(language='ko', version=version, model_dir=model_dir)
    return parser


# In[9]:


def test():
    parser = load_parser()
    text = '헤밍웨이는 1899년 7월 21일 미국 일리노이에서 태어났고, 62세에 자살로 사망했다.'
    text = '나는 밥을 먹었다.'
    stringuri = 'test:offset_0_53'
    parsed = parser.parser(text, sentence_id=stringuri)
    pprint(parsed['graph'])
    
# test()


# In[6]:


# word = '나는 밥을 먹었다'
# kkma = Kkma()
# morps = kkma.pos(word)
# print(morps)

