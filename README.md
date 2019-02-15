# Sound-Seperation-and-Localization

A pytorch implementation of Cross-Modal Sound Seperation and Localization. 

## Introduction

The Sound Seperation part is an implementation of the model proposed in [the Sound of Pixels](https://arxiv.org/pdf/1804.03160.pdf). The Sound Localization part is composed of a detection module to first decompose the sound spacially and then seperate the sound grounding its visual correspondence.

## Getting Started

### Environment

Make sure you have Anaconda installed. To configure the virtual environment, run the following command in the root directory.

```python
conda env create -f std_env.yml
pip install chainercv
```
### Dataset

