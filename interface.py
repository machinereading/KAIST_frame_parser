
# coding: utf-8

# In[7]:


import json
import os
from pprint import pprint
from KAIST_frame_parser import srl_based_parser


# In[2]:


try:
    config_dir = os.path.dirname( os.path.abspath( __file__ ))
except:
    config_dir = '.'    
config_file = config_dir+'/config.json'
with open(config_file, 'r') as f:
    config = json.load(f)


# In[3]:


class parser():
    def __init__(self):
        if config['parser'] == 'srl-based':
            self.frame_parser = srl_based_parser.SRLbasedParser
        self.fn_parser = self.frame_parser()
            
    def parsing(self, text, stringuri='stringurl:Null'):
        parsed = self.fn_parser.parser(text, stringuri)
        return parsed


# In[4]:


parser = parser()


# In[5]:


text = '헤밍웨이는 1899년 7월 21일 미국 일리노이에서 태어났고, 62세에 자살로 사망했다.'
parsed = parser.parsing(text)
pprint(parsed['graph'])

