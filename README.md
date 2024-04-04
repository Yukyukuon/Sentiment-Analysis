<!--<h3><b>DPR</b></h3>-->
# <b>Aspect-Term Polarity Classification in Sentiment Analysis</b>
## Introduction
This work implement a model that predicts opinion polarities (positive, negative or neutral) for given aspect terms in sentences. 
The model takes as input 3 elements: a sentence, a term occurring in the sentence, and its aspect category.For each input triple, it produces a polarity label: positive, negative or neutral. 
Note: the term can occur more than once in the same sentence, that is why its character start/end offsets are also provided.

![](https://cdn.jsdelivr.net/gh/Yukyukuon/CDN@latest/project/Sentiment_Analysis.jpg)

## Dataset
The dataset is in TSV format, one instance per line. As an example, here are 2 instances:  

| <!-- --> | <!-- --> | <!-- --> | <!-- --> | <!-- --> |
| :--- | :--- | :--- | --- | --- |  
| negative | SERVICE#GENERAL | Wait staff | 0:10 | Wait staff is blantently unappreciative of your business but its the best pie on the UWS! |  
| positive | FOOD#QUALITY | pie | 74:77 | Wait staff is blantently unappreciative of your business but its the best pie on the UWS! |  

Each line contains 5 tab-separated fields: the polarity of the opinion (the ground truth polarity label), the aspect category on which the opinion is expressed, a specific target term, the character offsets of the term (start:end), and the sentence in which the term occurs and the opinion is expressed.

There are 12 different aspects categories:
| <!-- --> | 
| :--- | 
| AMBIENCE#GENERAL |
| DRINKS#PRICES |
| DRINKS#QUALITY |
| DRINKS#STYLE_OPTIONS |
| FOOD#PRICES |
| FOOD#QUALITY |
| FOOD#STYLE_OPTIONS |
| LOCATION#GENERAL |
| RESTAURANT#GENERAL |
| RESTAURANT#MISCELLANEOUS |
| RESTAURANT#PRICES |
| SERVICE#GENERAL |

The training set (filename: traindata.csv) has this format (5 fields) and contains 1503 lines, i.e. 1503 opinions. The classifier should be learned only from this training set.  

A development dataset (filename: devdata.csv) is distributed to help you set up your classifier and estimate its performance. It has the same format as the training dataset. It has 376 lines, i.e. 376 opinions.  

## Dependencies
<p> pytorch = 2.1.x </p>
<p> transformers = 4.34.1 </p>
<P> tokenizers = 0.14.1 </p>
<p> datasets = 2.14.5 (just the huggingface library ‘datasets’, no labelled data) </p>
<p> scikit-learn = 1.2.1 </p>
<p> numpy = 1.26.0 </p>
<p> pandas = 2.1.1 </p>

## Train
I have prepared the training code in src document.  
You can use **tester.py** to train the model.  
```Python
python tester.py -n xxx -g xxxx
```
>'-n' '--n_runs' Number of runs  
>'-g' '--gpu' GPU device id on which to run the model
