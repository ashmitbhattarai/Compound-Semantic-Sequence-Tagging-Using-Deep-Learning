# Deep Learning for Intent Classification and Information Extraction

- Ashmit Bhattarai, Macquire University, 2017, Application of Data Science

## Introduction

- Text Classification
- Intent Classification
- NLP
- Information Extraction

## Related Works


## Methods

### Model
- Custom Architechture

## Tagging the dataset
- Overall Intent Since this is multi-class classifier
- To tag the phrases each word has following attribute
I) Intent - d,c,s,a,i,q,o,n
II) Sentiment Tag - p for positive, n negative effect in the phrase, neu for Neutral
III) Parts of Speech - rechecking the parts of Speech is infact valid (intuitive) || PENN Tree Bank to be followed
 - Examples to be added and discussed here
IV) Phrase chuck to be saved or not pr or npr
V) WH hooks - Part where WH question can be raised -> WH, Part Where A is gotten->A, No questions -> NQ

Examples:
d_p_VBZ_pr means the word belongs to a phrase chunk which contains demand/s from employee has positive sentiment and is a Verb word

### Dataset
- X sentences, split train ration and psuedo random generation
