# KAIST_frame_parser
**\*\*\*\*\* \[Update\] July, 2019  \*\*\*\*\***

KAIST_frame_parser is available for [Korean FrameNet](https://github.com/machinereading/koreanframenet).
KAIST_frame_parser consists of two parsers. (1) BERT-based (the newest version), and (2) Phrase Dependency-based (older version, called SRL-based). 

## About
KAIST-frame-parser is a semantic parser to understand the meaning of texts in terms of [FrameNet](https://framenet.icsi.berkeley.edu/fndrupal/). 

**frame** (frame semantics) is a schematic representation of a situation or an event. 
For an example sentence, '헤밍웨이는 1899년 7월 21일 일리노이에서 태어났고, 62세에 자살로 사망했다.', KAIST-frame-parser identifies several frames such as `Being_born` and `Death` for Korean lexical units (e.g. 태어나다.v and 사망하다.v)

<img src="./images/framenet.png" width="60%" height="60%">

Our model is based on the BERT with fine-tuning. The model predict Frames and their arguments jointly.

<img src="./images/framenet-bert.png" width="60%" height="60%">

## prerequisite
* `python 3`
* `pytorch-pretrained-BERT` ([Link](https://github.com/huggingface/pytorch-pretrained-BERT))
* `Korean FrameNet` ([Link](https://github.com/machinereading/koreanframenet))

## How to use
**Install**

First, install `pytorch-pretrained-BERT`, `KAIST_frame_parser`, and Korean FrameNet.
```
pip3 install pytorch-pretrained-bert
git clone https://github.com/machinereading/KAIST_frame_parser.git
git clone https://github.com/machinereading/koreanframenet.git
```
Second, copy a file `'./data/bert-multilingual-cased-dict-add-frames'` to your pytorch-pretrained-bert tokenizer's vocabulary.
Please follow this:
* (1) may your vocabulary is in `.pytorch-pretrained-bert`folder under your home. 
```
cd ~/.pytorch-pretrained-bert
```
* (2) make sure what file is a vocabulary file for `bert-base-multilingual-cased`. 
For example, if the url `https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-multilingual-cased-vocab.txt` is in the file `9626...252.json` (file name would be different), another file with same name `9626...252` is the vocabulary file for `bert-base-multilingual-cased`.
* (3) copy the file `'./data/bert-multilingual-cased-dict-add-frames'` to that folder.
```
cp ./data/bert-multilingual-cased-dict-add-frames ~/.pytorch-pretrained-bert/9626...252
```
* (don't forget to make a back-up file for `9626...252`)

### How to use BERT-based frame parser (the newest version)

**Download the pretrained model**

Download two pretrained model files to `{your_model_dir}` (e.g. `/home/model/bert_ko_srl_model.pt`). 
* **Download:** ([click](https://drive.google.com/open?id=1lmyFhrr77oNYZlo0sYsTJFz8stXscoEr))

**Import bert_based_parser (in your python code)**
(make sure that your code is in a parent folder of BERT_for_Korean_SRL)
```
from KAIST_frame_parser import bert_based_parser

model_dir = {your_model_dir} # absolute_path (e.g. /home/model/bert_ko_frame_model.pt)
parser = bert_based_parser.BERTbasedParser(model_dir=model_dir)
```

**Parse the input text**
```
text = '헤밍웨이는 1899년 7월 21일 미국 일리노이에서 태어났고 62세에 자살로 사망했다.'
parsed = parser.joint_parser(text)
```

**Result**
The result is a list, which consists of multiple Frame-Semantic structures. Each SRL structure is in a list, which consists of four lists: (1) tokens, (2) lexical units, (3) its frames, and (4) its arguments. For example, for the given input text, the output is in the following format:

```
[
    [
        ['헤밍웨이는', '1899년', '7월', '21일', '미국', '일리노이에서', '태어났고,', '62세에', '자살로', '사망했다.'], 
        ['_', '_', '_', '_', '미국.n', '_', '_', '_', '_', '_'], 
        ['_', '_', '_', '_', 'Origin', '_', '_', '_', '_', '_'], 
        ['O', 'O', 'O', 'O', 'O', 'B-Entity', 'O', 'O', 'O', 'O']
    ], 
    [
        ['헤밍웨이는', '1899년', '7월', '21일', '미국', '일리노이에서', '태어났고,', '62세에', '자살로', '사망했다.'],
        ['_', '_', '_', '_', '_', '_', '태어나다.v', '_', '_', '_'], 
        ['_', '_', '_', '_', '_', '_', 'Being_born', '_', '_', '_'], 
        ['B-Child', 'B-Time', 'I-Time', 'I-Time', 'B-Place', 'I-Place', 'O', 'O', 'O', 'O']
    ], 
    [
        ['헤밍웨이는', '1899년', '7월', '21일', '미국', '일리노이에서', '태어났고,', '62세에', '자살로', '사망했다.'], 
        ['_', '_', '_', '_', '_', '_', '_', '_', '자살.n', '_'], 
        ['_', '_', '_', '_', '_', '_', '_', '_', 'Killing', '_'], 
        ['B-Victim', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O']
    ],
    [
        ['헤밍웨이는', '1899년', '7월', '21일', '미국', '일리노이에서', '태어났고,', '62세에', '자살로', '사망했다.'], 
        ['_', '_', '_', '_', '_', '_', '_', '_', '_', '사망.n'], 
        ['_', '_', '_', '_', '_', '_', '_', '_', '_', 'Death'], 
        ['B-Protagonist', 'O', 'O', 'O', 'O', 'O', 'O', 'B-Time', 'B-Manner', 'O']
    ]
]
```

Another example sentence is '그는 그녀와 사랑에 빠졌다.'.
```
[
    [
        ['그는', '그녀와', '사랑에', '빠졌다.'], 
        ['_', '_', '사랑.n', '_'], 
        ['_', '_', 'Personal_relationship', '_'], 
        ['B-Partner_1', 'B-Partner_2', 'O', 'O']
    ],
    [
        ['그는', '그녀와', '사랑에', '빠졌다.'], 
        ['_', '_', '_', '빠지다.v'], 
        ['_', '_', '_', 'Experiencer_focus'], 
        ['B-Experiencer', 'B-Topic', 'I-Topic', 'O']
    ]
]
```
The word '빠지다' would be have different meaning in its usage in the context. 

An example is '검은 얼룩이 흰 옷에서 빠졌다.'.
```
[
    [
        ['검은', '얼룩이', '흰', '옷에서', '빠졌다.'], 
        ['_', '_', '_', '옷.n', '_'], 
        ['_', '_', '_', 'Clothing', '_'], 
        ['O', 'O', 'B-Descriptor', 'O', 'O']
    ],
    [
        ['검은', '얼룩이', '흰', '옷에서', '빠졌다.'], 
        ['_', '_', '_', '_', '빠지다.v'], 
        ['_', '_', '_', '_', 'Emptying'], 
        ['B-Theme', 'I-Theme', 'B-Source', 'I-Source', 'O']
    ]
]
```

### (Optional) How to use SRL-based frame parser (the older version)
**prerequisite**

SRL-based frame parser is working only for Korean. It requires NLP modules as a preprocessing. In this library, we use Korean NLP service [wiseNLU](http://aiopen.etri.re.kr/guide_wiseNLU.php). Please get API code and edit the config file first. 

**Download the pretrained model**

Download two pretrained model files to `{your_model_dir}` (e.g. `/home/models`). Do not change the model file names.
* **kfn1.1-frameid.pt** ([download](https://drive.google.com/open?id=1fgmUU9trekwP-fBc7pz62n0lgJH9P4eJ))
* **kfn1.1-arg_classifier.pt** ([download](https://drive.google.com/open?id=1jZEvrmQEvRwDDS3wDZ4pqHbyhqoJ99Wy))

**Import srl_based_parser (in your python code)**
```
from KAIST_frame_parser import srl_based_parser
language = 'ko' # default
version = 1.1 # default
model_dir = {your_model_dir} # absolute_path (e.g. /home/models)
parser = srl_based_parser.SRLbasedParser(language=language, version=version, model_dir=model_dir)
```

**parse the input text**
```
text = '헤밍웨이는 1899년 7월 21일 미국 일리노이에서 태어났고, 62세에 자살로 사망했다.'
sentence_id = 'input_sentence' # (optional) you can assign the input text to its ID.
parsed = parser.parser(text, sentence_id=sentence_id)
```

**result**

The result consits of following three parts: (1) triple format, (2) conll format,  and (3) [pubannotation format](https://textae.pubannotation.org/). 

* (1) triple format (`parsed['graph']`)
```
[
    ('input_sentence', 'nif:isString', '헤밍웨이는 1899년 7월 21일 미국 일리노이에서 태어났고, 62세에 자살로 사망했다.'),
    ('frame:Origin', 'frdf:provinence', 'input_sentence'),
    ('frame:Origin', 'frdf:lu', '미국.n'),
    ('frame:Origin', 'frdf:score', '1.0'),
    ('frame:Being_born', 'frdf:provinence', 'input_sentence'),
    ('frame:Being_born', 'frdf:lu', '태어나다.v'),
    ('frame:Being_born', 'frdf:score', '1.0'),
    ('frame:Being_born', 'arg:Child', '헤밍웨이는'),
    ('frame:Being_born', 'arg:Time', '1899년 7월 21일'),
    ('frame:Being_born', 'arg:Place', '미국 일리노이에서'),
    ('frame:Killing', 'frdf:provinence', 'input_sentence'),
    ('frame:Killing', 'frdf:lu', '자살.n'),
    ('frame:Killing', 'frdf:score', '1.0'),
    ('frame:Death', 'frdf:provinence', 'input_sentence'),
    ('frame:Death', 'frdf:lu', '사망.n'),
    ('frame:Death', 'frdf:score', '1.0'),
    ('frame:Death', 'arg:Protagonist', '헤밍웨이는'),
    ('frame:Death', 'arg:Time', '62세에'),
    ('frame:Death', 'arg:Manner', '자살로')
]
```
* (2) conll format (`parsed['conll']`)
The result is a list of (multiple) FrameNet annotations for a given sentence. 
Each annotation consits of 4 lists:  tokens, target, frame, and its arguments


## Licenses
* `CC BY-NC-SA` [Attribution-NonCommercial-ShareAlike](https://creativecommons.org/licenses/by-nc-sa/2.0/)
* If you want to commercialize this resource, [please contact to us](http://mrlab.kaist.ac.kr/contact)

## Publisher
[Machine Reading Lab](http://mrlab.kaist.ac.kr/) @ KAIST

## Contact
Younggyun Hahm. `hahmyg@kaist.ac.kr`, `hahmyg@gmail.com`

## Acknowledgement
This work was supported by Institute for Information & communications Technology Promotion(IITP) grant funded by the Korea government(MSIT) (2013-0-00109, WiseKB: Big data based self-evolving knowledge base and reasoning platform)
