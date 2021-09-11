---
layout: post
title: "Image Caption Generation Using LSTM"
author: "Abhishek"
categories: journal
tags: [Project,ML]
image: forest.jpg
---
# Contents

1. [Objective](#objective)
2. [Pre-processing Data](#pre-processing)
3. [Model](#model)
4. [Processing in Batch](#processing-in-batch)
5. [Test](#test)
6. [Conclusion](#conclusion)

## Objective

Generating sequential text from image encodings to create caption suitable for the given image. Evaluated using BLEU Score


Caption generation is a challenging artificial intelligence problem where a textual description must be generated for a given photograph. It requires both methods from computer vision to understand the content of the image and a language model from the field of natural language processing to turn the understanding of the image into words in the right order.

## Pre-processing

### For Text data

We used [Flickr8k dataset](https://www.kaggle.com/ming666/flicker8k-dataset). It has two parts:-
* Flickr8k_Dataset: Contains 8092 photographs in JPEG format.
* Flickr8k_text: Contains a number of files containing different sources of descriptions for the photographs.

The text requires some minimal cleaning. we will clean the text in the following ways in order to reduce the size of the vocabulary of words we will need to work with:

* Convert all words to lowercase.
* Remove all punctuation.
* Remove all words that are one character or less in length (e.g. ‘a’).
* Remove all words with numbers in them.

### For Image data

To get Image features using pre-trained model is feasible.We'll use [resnet50](https://arxiv.org/abs/1512.03385) Model to get image encodings


## Model
#### Outline :
![alt](https://drive.google.com/uc?export=view&id=1WmhME7QXQ6FAnu6Y_WwHknSYvdqJXYp-)

```
____________________________________________________________________________________________________
Layer (type)                     Output Shape          Param #     Connected to
====================================================================================================
input_2 (InputLayer)             (None, 34)            0
____________________________________________________________________________________________________
input_1 (InputLayer)             (None, 4096)          0
____________________________________________________________________________________________________
embedding_1 (Embedding)          (None, 34, 256)       1940224     input_2[0][0]
____________________________________________________________________________________________________
dropout_1 (Dropout)              (None, 4096)          0           input_1[0][0]
____________________________________________________________________________________________________
dropout_2 (Dropout)              (None, 34, 256)       0           embedding_1[0][0]
____________________________________________________________________________________________________
dense_1 (Dense)                  (None, 256)           1048832     dropout_1[0][0]
____________________________________________________________________________________________________
lstm_1 (LSTM)                    (None, 256)           525312      dropout_2[0][0]
____________________________________________________________________________________________________
add_1 (Add)                      (None, 256)           0           dense_1[0][0]
                                                                   lstm_1[0][0]
____________________________________________________________________________________________________
dense_2 (Dense)                  (None, 256)           65792       add_1[0][0]
____________________________________________________________________________________________________
dense_3 (Dense)                  (None, 7579)          1947803     dense_2[0][0]
====================================================================================================
Total params: 5,527,963
Trainable params: 5,527,963
Non-trainable params: 0
____________________________________________________________________________________________________
```
## Processing in batch

The training of this model may need a lot of ram. A Video card GPU RAM may not be suuficient hence we use progressive loading. We basically only load data worth one batch in memory instead of the entire data.

```python

def data_generator(train_descriptions,encoding_train,word_to_idx,max_len,batch_size):
    X1,X2, y = [],[],[]
    
    n =0
    while True:
        for key,desc_list in train_descriptions.items():
            n += 1
            
            photo = encoding_train[key+".jpg"]
            for desc in desc_list:
                
                seq = [word_to_idx[word] for word in desc.split() if word in word_to_idx]
                for i in range(1,len(seq)):
                    xi = seq[0:i]
                    yi = seq[i]
                    
                    #0 denote padding word
                    xi = pad_sequences([xi],maxlen=max_len,value=0,padding='post')[0]
                    yi = to_categorcial([yi],num_classes=vocab_size)[0]
                    
                    X1.append(photo)
                    X2.append(xi)
                    y.append(yi)
                    
                if n==batch_size:
                    yield [[np.array(X1),np.array(X2)],np.array(y)]
                    X1,X2,y = [],[],[]
                    n = 0


``` 

## Test 

We decided to use BLEU for evaluation because it is simple to use and its already
implemented in NLTK library. BLEU stands for Bilingual Evaluation Understudy. It lies between [0,1]. Higher the score better the quality of caption.

![alt](https://drive.google.com/uc?export=view&id=1C75tc4PA2kI8mQRQVSYktEHozxDWOPUW)

![](https://drive.google.com/uc?export=view&id=1UWXq6kdjVS_HRyHzMswHDRqXwJoJbHI0)

## Conclusion 

We also conclude that an LSTM is more suitable for caption generation than other vanilla recurrent neural networks(RCNN) because of its Long short term memory. Result varies greatly between images but can be improved by parameter
tuning and using one of the recent evaluation metrics which can overcome BLEU score.

### Further Scope

* Using Alternate pre-trained CNN models can improve results
* Using smaller vocabulary
* Using pre-trained word vectors like Word2Vec 

This was capstone project for my internship at IIT Kanpur under [Prof. Vipul Arora](https://scholar.google.co.in/citations?user=SC9YYPAAAAAJ&hl=en)

The code can be found [here](https://www.kaggle.com/abhishekgawande/image-caption-eda)
