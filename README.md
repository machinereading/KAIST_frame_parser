# KAIST_frame_parser

## About
KAIST_frame_parser is based on the [Korean FrameNet](https://github.com/machinereading/koreanframenet).
It consists of two parsers. (1) SRL-based, and (2) BERT-based. Now (April, 2019) only SRL-based is available.

## prerequisite
* `python 3`
* `pytorch-pretrained-BERT` ([Link](https://github.com/huggingface/pytorch-pretrained-BERT))
* `Korean FrameNet` ([Link](https://github.com/machinereading/koreanframenet))

## How to use
**Install**

First, install `pytorch-pretrained-BERT`.
```
pip3 install pytorch-pretrained-bert
```
Second, install KAIST_frame_parser at your workspace directory.
```
git clone https://github.com/machinereading/KAIST_frame_parser.git
```
Last, install Korean FrameNet at the KAIST_frame_parser directory.
```
cd ./KAIST_frame_parser
git clone https://github.com/machinereading/koreanframenet.git
```

### (1) How to use SRL-based frame parser
**prerequisite**

SRL-based frame parser requires NLP modules as a preprocessing. In this library, we use wiseNLU(http://aiopen.etri.re.kr/guide_wiseNLU.php). Please get API code and edit the config file first. 

**Download the pretrained model**

Download two pretrained model files to `{your_model_dir}`. Do not change the model file names.
* kfn11-frameid.pt ([link])
* kfn1.1-arg_classifier.pt ([link])

**Import srl_based_parser (in your python code)**
```
from KAIST_frame_parser import srl_based_parser
language = 'ko' _# default_
version = 1.1 _# default_
model_dir = {your_model_dir}
parser = srl_based_parser.SRLbasedParser(language=language, version=version, model_dir=model_dir)
```

**parse the input text**
```
text = '헤밍웨이는 1899년 7월 21일 미국 일리노이에서 태어났고, 62세에 자살로 사망했다.'
sentence_id = 'input_sentence' _# (optional) you can assign the input text to its ID._
parsed = parser.parser(text, sentence_id=sentence_id)
```

**result**

The result consits of following three parts: (1) conll format, (2) triple format, and (3) [pubannotation format](https://textae.pubannotation.org/). 

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
