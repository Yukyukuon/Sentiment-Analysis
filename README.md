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
