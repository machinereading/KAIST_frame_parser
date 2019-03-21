
# coding: utf-8

# # This is a IO adapter for ETRI NLP service
# 
# * HOW TO USE: set SETTINGS for ETRI NLP SERVICE URL

# In[1]:


# SETTINGS
etri_rest_url = 'http://143.248.135.20:31235/etri_parser'
etri_socket_url = '143.248.135.60'
etri_socket_port = 33222

# service = 'REST'
serviceType = 'SOCKET'


# In[2]:


import urllib.request
from urllib.parse import urlencode
import json
import pprint
import socket
import struct


# In[12]:


def getETRI_rest(text):
    url = etri_rest_url
    contents = {}
    contents['text'] = text
    contents = json.dumps(contents).encode('utf-8')
    u = urllib.request.Request(url, contents)
    response = urllib.request.urlopen(u)
    result = response.read().decode('utf-8')
    result = json.loads(result)
    return result


# In[4]:


def read_blob(sock, size):
    buf = ''
    while len(buf) != size:
        ret = sock.recv(size - len(buf))
        if not ret:
            raise Exception("Socket closed")
        ret += buf
    return buf
def read_long(sock):
    size = struct.calcsize("L")
    data = readblob(sock, size)
    return struct.unpack("L", data)

def getETRI(text):    
    host = etri_socket_url
    port = etri_socket_port
    ADDR = (host, port)
    clientSocket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        clientSocket.connect(ADDR)
    except Exception as e:
        return None
    try:
        clientSocket.sendall(str.encode(text))
        #clientSocket.sendall(text.encode('unicode-escape'))
        #clientSocket.sendall(text.encode('utf-8'))
        buffer = bytearray()
        while True:
            data = clientSocket.recv(1024)
            if not data:
                break
            buffer.extend(data)
        result = json.loads(buffer.decode(encoding='utf-8'))
        return result['sentence']
    except Exception as e:
        return None


# In[5]:


def lemmatizer(word, pos):
    etri = getETRI(word)
    lemmas = etri[0]['WSD']
    lemma = word
    for i in lemmas:
        p = i['type']
        if pos == 'v' or pos == 'VV':
            if p == 'VV':
                lemma = i['text']
                break
        elif pos == 'n' or pos == 'NN' or pos == 'NNG' or pos == 'NNP' or pos =='NNB' or pos =='NR' or pos == 'NP':
            if 'NN' in p:
                lemma = i['text']
                break
        elif pos == 'adj' or pos == 'VA':
            if p == 'VA':
                lemma = i['text']
                break
        else:
            pass
    return lemma

def getPOS(word):
    etri = getETRI(word)
    pos = etri[0]['WSD'][0]['type']
    if pos.startswith('N'):
        pos = 'n'
    elif pos == 'VV':
        pos = 'v'
    elif pos == 'VA':
        pos = 'adj'
    else:
        pos == 'n'
    return pos

def getMorpEval(tid, nlp):
    result = '_'
    for i in nlp[0]['morp_eval']:
        if i['id'] == tid:
            morp = i['result']
            morps = morp.split('+')
            pos_sequence = []
            for m in morps:
                if '/' not in m:
                    pass
                else:
                    p = m.split('/')[1]
                    pos_sequence.append(p)
            pos = '+'.join(pos_sequence)
            result = pos
        else:
            pass
    return result

def getMorhWithWord(tid, nlp):
    result = '_'
    for i in nlp[0]['morp_eval']:
        if i['id'] == tid:
            morp = i['result']
            break
    return morp


# In[6]:


def getETRI_CoNLL2006(text):
    nlp = getETRI(text)
    result = []
    for i in nlp[0]['dependency']:
        tid = i['id']
        token = i['text']
        third = getMorhWithWord(tid, nlp)
        pos = getMorpEval(tid, nlp)
        five = '_'
        arc = i['head']
        pt = i['label']
        eight = '_'
        nine = '_'
        line = [tid, token, third, pos, five, arc, pt, eight, nine]
        result.append(line)
    return result


# In[48]:


def getETRI_CoNLL2009(nlp):
    result = []
    
    if nlp:
        for i in nlp[0]['dependency']:
            tid = i['id']
            token = i['text']
            third = getMorhWithWord(tid, nlp)
            plemma = token
            pos = getMorpEval(tid, nlp)
            ppos = pos
            feat = '_'
            pfeat = '_'
            head = i['head']
            phead = head
            deprel = i['label']
            pdeprel = i['label']
            line = [tid, token, third, plemma, pos, ppos, feat, pfeat, head, phead, deprel, pdeprel]
            result.append(line)
    else:
        pass
    return result


# In[8]:


def get_verb_ids(conll):
    verbs = []
    for token in conll:
        m = token[2].split('+')[0]
        w,p = m.split('/')[0], m.split('/')[1]
        last_p = token[2].split('+')[-1].split('/')[1]
        if last_p == 'ETM':
            vtype = 'a'
        else:
            vtype = 'v'
        if p == 'VV':
            pos = 'v'
            v = w+'.'+pos
            verb = (int(token[0]), v, vtype)
            verbs.append(verb)
    return verbs

def get_arg_ids(verb_id, verb_type, nlp):    
    phrase_dp = nlp[0]['phrase_dependency']
    dp = nlp[0]['dependency']
    verb_phrase_id = -1
    for arg in phrase_dp:
        b,e = arg['begin'], arg['end']
        if b <= verb_id <= e:
            verb_phrase_id = arg['id']
            break
    arg_ids = []
    if verb_phrase_id > 0:
        tokens = phrase_dp[verb_phrase_id]['text'].split(' ')
        in_sbj = False
        if '@SBJ' in phrase_dp[verb_phrase_id]['text']:
            in_sbj = True
        
        
        for token in tokens:
            if '@' in token:
                arg_id = int(token.split('@')[0].split('#')[1])
                arg_type = token.split('@')[0].split('#')[0]
                arg_label = token.split('@')[1][:3]
                if arg_type != 'S':
                    dp_label = phrase_dp[arg_id]['label']
                    seperate = False
                    arg = (arg_id, dp_label, seperate)
                    arg_ids.append(arg)
                elif arg_label == 'CMP':
                    dp_label = phrase_dp[arg_id]['label']
                    if 'SBJ' in phrase_dp[verb_phrase_id]['text']:
                        seperate = False
                        arg = (arg_id, dp_label, seperate)
                        arg_ids.append(arg)
                    else:
                        seperate = True
                        arg = (arg_id, dp_label, seperate)
                        arg_ids.append(arg)
                        
                else:
                    if in_sbj == True:
                        pass
                    else:
                        origin = phrase_dp[arg_id]['text']
                        ori_tokens = origin.split(' ')
                        for t in ori_tokens:
                            if '@SBJ' in t:
                                sbj_id = int(t.split('@')[0].split('#')[1])
                                dp_label = phrase_dp[sbj_id]['label']
                                seperate = False
                                arg = (sbj_id, dp_label, seperate)
                                arg_ids.append(arg)
        if verb_type == 'v':
            pass
        elif verb_type == 'a':
            head_id = phrase_dp[verb_phrase_id]['head_phrase']
            dp_label = phrase_dp[head_id]['label']
            seperate = False
            arg = (head_id, dp_label, seperate)
            arg_ids.append(arg)
            
            
                    
    args = []
    for arg_id, dp_label, seperate in arg_ids:
        if seperate == False:
            if arg_id < verb_phrase_id:
                begin = phrase_dp[arg_id]['begin']
                end = phrase_dp[arg_id]['end']
                span = []
                span.append(begin)
                n = begin
                while n < end:
                    n = n+1
                    span.append(n)
                arg = {}
                arg['tokens'] = span
                arg['dp_label'] = dp_label
                args.append(arg)
            else:
                if verb_phrase_id in phrase_dp[arg_id]['sub_phrase']:
                    begin = phrase_dp[verb_phrase_id]['end'] +1
                    end = phrase_dp[arg_id]['end']
                    span = []
                    span.append(begin)
                    n = begin
                    while n < end:
                        n = n+1
                        span.append(n)
                    arg = {}
                    arg['tokens'] = span
                    arg['dp_label'] = dp_label
                    args.append(arg)
        else:
            sbj_end = -1
            add_sbj = True
            for token in tokens:
                if 'SBJ' in token:
                    add_sbj = False
            if add_sbj:
                for token in tokens:
                    if token.startswith('S'):
                        subphrase_id = int(token.split('@')[0].split('#')[-1])
                        subphrase = phrase_dp[subphrase_id]
                        sub_p_toks = subphrase['text'].split(' ')
                        for sub_p_tok in sub_p_toks:
                            if 'SBJ' in sub_p_tok:
                                sbj_id = int(sub_p_tok.split('@')[0].split('#')[-1])
                        
                        
                                sbj_begin = phrase_dp[sbj_id]['begin']
                                sbj_end = phrase_dp[sbj_id]['end']
                                span = []
                                span.append(sbj_begin)
                                n = sbj_begin
                                while n < sbj_end:
                                    n = n+1
                                    span.append(n)
                                arg = {}
                                arg['tokens'] = span
                                arg['dp_label'] = phrase_dp[sbj_id]['label']
                                args.append(arg)
                    
            if arg_id < verb_phrase_id:
#                 begin = phrase_dp[arg_id]['begin']
                begin = sbj_end +1
                if sbj_end >0:
                    begin = sbj_end +1
                else:
                    begin = phrase_dp[arg_id]['begin']
                end = phrase_dp[arg_id]['end']
                
                
        
                span = []
                span.append(begin)
                n = begin
                while n < end:
                    n = n+1
                    span.append(n)
                arg = {}
                arg['tokens'] = span
                arg['dp_label'] = dp_label
                args.append(arg)
            else:
                if verb_phrase_id in phrase_dp[arg_id]['sub_phrase']:
                    begin = phrase_dp[verb_phrase_id]['end'] +1
                    end = phrase_dp[arg_id]['end']
                    span = []
                    span.append(begin)
                    n = begin
                    while n < end:
                        n = n+1
                        span.append(n)
                    arg = {}
                    arg['tokens'] = span
                    arg['dp_label'] = dp_label
                    args.append(arg)
            tokens = phrase_dp[arg_id]['text'].split(' ')
            
                
            
    return args

def get_arg_text(arg_ids, conll):
    arg = []
    for arg_id in arg_ids:
        token = conll[arg_id][1]
        arg.append(token)
    arg_text = ' '.join(arg)
    return arg_text

def get_josa(conll, token_id):
    josa = {}
    josa['pos'] = {}
    josa['josa'] = {}
    josa['josa+pos'] = {}
    if token_id >= len(conll):
        token_id = -1
    morphemes = conll[token_id][2].split('+')
    for m in morphemes:
        word = m.split('/')[0]
        pos = m.split('/')[-1]
        if pos.startswith('J') or pos == 'EC':
            josa = {}
            josa['pos'] = pos
            josa['josa'] = word
            josa['josa+pos'] = m
    return josa

def get_args(verb_id, verb_type, nlp, conll):
    arguments = []
    arg_ids = get_arg_ids(verb_id, verb_type, nlp)
    sent_lenth = len(nlp[0]['dependency'])
    for arg_item in arg_ids:
        tokens = arg_item['tokens']
        arg_text = get_arg_text(tokens, conll)
        arg = {}
        arg['text'] = arg_text
        arg['tokens'] = tokens
        arg['dp_label'] = arg_item['dp_label']
        
        span = {}
        begin, end = tokens[0], tokens[-1]+1
        if end > sent_lenth:
            end = sent_lenth
        span['begin'] = begin
        span['end'] = end        
        arg['span'] = span
        
        josa = get_josa(conll, end-1)
        arg['josa'] = josa
        arguments.append(arg)
    return arguments


# In[33]:

def stemmer(text):
    nlp = getETRI(text)
    result = []
    for i in nlp[0]['morp_eval']:
        result.append(i['result'].split('+')[0])
    return result



def phrase_parser(conll_2009, nlp):
    conll = conll_2009
    result = []
    if conll:
        verb_ids = get_verb_ids(conll)
        for verb_id, verb, verb_type in verb_ids:
            d = {}
            pred = {}
            pred['text'] = verb
            pred['id'] = verb_id
            d['predicate'] = pred
            arguments = get_args(verb_id, verb_type, nlp, conll)
            d['arguments'] = arguments
            result.append(d)
    else:
        pass
    return result


# In[50]:


def test_conll():
    text = '안녕하세요'
    #conll = getETRI_CoNLL2006(text)
    nlp = getETRI(text)
    conll= getETRI_CoNLL2009(nlp)
    for token in conll:
        print(token)
# test_conll()


# In[43]:


def test():
    example_1 = '센터장은 안전 절차의 철저한 검토를 약속하였다.'
    example_2 = '안토니우 구테흐스는 포르투칼의 총리를 지냈고, 2016년 10월 13일에 유엔 사무총장으로 선출되었다.'
    example_3 = '포르투칼의 총리를 지낸 안토니우 구테흐스는, 2016년 10월 13일에 유엔 사무총장으로 선출되었다.'
    example_4 = '클래리베이트 애널리틱스는 올해 노벨상 수상자 예측 명단에 울산과학기술원(UNIST) 소속 연구자가 포함됐다고 오늘 발표했다.'
    example_5 = '올해 노벨상 수상자 예측 명단에 울산과학기술원(UNIST) 소속 연구자가 포함됐다고 2018년 9월 20일 클래리베이트 애널리틱스는 발표했다.'
    example_6 = '함영균의 직업은 학생이다'
    example_7 = '해고당한 노동자가 자기를 선처해 달라고 고용주에게 사정사정했다.'
    example_8 = '어머니는 고기를 많이 주물럭대야지 고기가 부드러워진다고 하셨다.'
    example_9 = '아이의 엄마는 선생님께 아이를 잘 봐달라고 말했다.'
    example_10 = '카다피는 이날 인민통치체제 성립 23 주년을 기념해 리비아 남부 세브하시에서 열린 군중집회에서 "나는 사회인민본부 총조정관이 공식적인 국가원수가 되는 헌법조항이 마련돼야 한다고 생각한다"면서 "국가원수는 전쟁이나 대재앙 등의 문제가 생길 때를 대비해 필요하다"고 말했다.'
    
    text = example_10
    
    nlp = getETRI(text)    
    conll_2009 = getETRI_CoNLL2009(nlp)
    predicate_argument = phrase_parser(conll_2009, nlp)
    print('\n###################\nINPUT TEXT:',text,'\n')
    pprint.pprint(predicate_argument)
# test()

