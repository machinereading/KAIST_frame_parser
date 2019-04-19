# KAIST_frame_parser

## About
KAIST_frame_parser is based on the [Korean FrameNet](https://github.com/machinereading/koreanframenet).
It consists of two parsers. (1) SRL-based, and (2) BERT-based. Now (April, 2019) only SRL-based is available.

## prerequisite
* `python 3`
* `pytorch-pretrained-BERT` [Link](https://github.com/huggingface/pytorch-pretrained-BERT)
* `Korean FrameNet` [Link](https://github.com/machinereading/koreanframenet)

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
Finally, install Korean FrameNet at the KAIST_frame_parser directory.
```
cd ./KAIST_frame_parser
git clone https://github.com/machinereading/koreanframenet.git
```

**Edit the `config.json` file**

NOTICE: TRAINED MODEL IS NOT AVAILABLE YET

## Licenses
* `CC BY-NC-SA` [Attribution-NonCommercial-ShareAlike](https://creativecommons.org/licenses/by-nc-sa/2.0/)
* If you want to commercialize this resource, [please contact to us](http://mrlab.kaist.ac.kr/contact)

## Publisher
[Machine Reading Lab](http://mrlab.kaist.ac.kr/) @ KAIST

## Contact
Younggyun Hahm. `hahmyg@kaist.ac.kr`, `hahmyg@gmail.com`

## Acknowledgement
This work was supported by Institute for Information & communications Technology Promotion(IITP) grant funded by the Korea government(MSIT) (2013-0-00109, WiseKB: Big data based self-evolving knowledge base and reasoning platform)
