
# coding: utf-8

# In[1]:


import os
import json
from KAIST_frame_parser.src import dataio
from KAIST_frame_parser.src import etri
from collections import Counter


# # Basic settings

# In[2]:


language = 'ko'
version = 1.1


# In[3]:


from KAIST_frame_parser.koreanframenet import koreanframenet
kfn = koreanframenet.interface(version)


# In[46]:


from konlpy.tag import Kkma
kkma = Kkma()
def targetize(word):
    result = []
    morps = kkma.pos(word)
    v = False
    for m,p in morps:
        if p == 'XSV' or p == 'VV':
            v = True    
    
    if v:
        for i in range(len(morps)):
            m,p = morps[i]
            if p == 'VA' or 'VV':
                if m[0] == word[0] and len(m) > 1:
                    result.append(m)
            if i > 0 and p == 'XSV':
                if m[0] == word[0] and len(m) > 1:
                    result.append(m)
                r = morps[i-1][0]+m
                if r[0] == word[0]:
                    result.append(r)
    else:
        pos_list = []
        for m,p in morps:
            if p.startswith('J'):
                pos_list.append(m)
            elif p == 'VCP' or p == 'EFN':
                pos_list.append(m)
        for m, p in morps:
            if p == 'NNG' or p == 'NNP':
                if len(pos_list) == 0:
                    if m == word:
                        result.append(m)
                else:
                    if m[0] == word[0]:
                        result.append(m)
    return result

try:
    target_dir = os.path.dirname( os.path.abspath( __file__ ))
except:
    target_dir = '.'

with open(target_dir+'/data/targetdic-'+str(version)+'.json','r') as f:
    targetdic = json.load(f)
def get_lu_by_token(token):
    target_candis = targetize(token)
    lu_candis = []
    for target_candi in target_candis:
        for lu in targetdic:
            if target_candi in targetdic[lu]:
                lu_candis.append(lu)
    common = Counter(lu_candis).most_common()
    if len(common) > 0:
        result = common[0][0]
    else:
        result = False
    return result


# In[49]:


# input: text or json
def baseline(data):
    result = []
    idxs, tokens = data[0], data[1]
    for idx in range(len(tokens)):
        token = tokens[idx]
        lu = get_lu_by_token(token)
        lus = ['_' for i in range(len(tokens))]
        if lu:
            lus[idx] = lu
            instance = []            
#             instance.append(idxs)
            instance.append(tokens)
            instance.append(lus)
            result.append(instance)
    return result
        
# text = '애플은 스티브 잡스와 스티브 워즈니악과 론 웨인이 1976년에 설립한 컴퓨터 회사이다.'
# text = '헤밍웨이는 1961년 아이다호 주에서 62세의 나이에 자살했다.'
# text = '헤밍웨이는 풀린 파이퍼와 이혼한 뒤 마사 겔혼과 재혼하였다'
# text = '애플은 스티브 잡스와 스티브 워즈니악과 론 웨인이 1976년에 설립한 회사이다.'
# text = '1854년 노벨 문학상을 수상하였다'
# text = '잡스는 미국에서 태어났다.'
# text = '헤밍웨이는 태어났고 마사 겔혼과 이혼하였다.'
# text = '헤밍웨이는 미국에서 태어났다.'
# text = '헤밍웨이는 풀린 파이퍼와 이혼한 뒤 마사 겔혼과 재혼하였다'


# tl = text.split(' ')
# for t in tl:
#     target = targetize(t)
#     morps = kkma.pos(t)
#     print('pos:',morps)
#     print('targets:',target)
# input_data = dataio.preprocessor(text)
# d = baseline(input_data)
# print('result')
# print(d)

