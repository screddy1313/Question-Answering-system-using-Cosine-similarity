# -*- coding: utf-8 -*-
"""
Created on Thu Oct 31 23:35:57 2019

@author: sarat
"""
import numpy as np
#import pandas as pd

from nltk import word_tokenize
from nltk.stem import PorterStemmer
ps = PorterStemmer()

import os
import math

from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))

import pickle

import query as q
#%%
def preprocess(doc,vocab):
    words = word_tokenize(doc)
    mod_words = []
    for word in words:
        word = word.lower()
        #word = lemmatizer.lemmatize(word)
        
        word = ps.stem(word) 
        symbols = ['.',',','?','!','@','#','$','%','&']
        if word[-1] in symbols:
            word = word[:-1]
        
        if len(word) > 1 and word not in stop_words:
                vocab.add(word)
                mod_words.append(word)
    return mod_words,vocab

def load_data(path,vocab):
    
    docs = []
    with open(path,'r',encoding="utf8") as f:
        doc = f.readline()
        while doc :
            doc,vocab = preprocess(doc,vocab)
            docs.append(doc)
            doc = f.readline()
    return docs,vocab

def pickle_data(data,name):
    
    path = os.path.join('pickle',name)
    out = open(path+'.pkl','wb')
    pickle.dump(data,out)

def load_pickle_data(path):
    
    path = os.path.join('pickle',path)
    out = open(path,'rb')
    data = pickle.load(out)
    return data


def compute_tf(doc):
    tf_dict = {}
    for key in doc:
        if key in tf_dict.keys():
            tf_dict[key] += 1
        else:
            tf_dict[key] = 1
            
    n = len(doc)
    for key in tf_dict.keys():
        tf_dict[key] = 1+ math.log10(tf_dict[key]/n)
    return tf_dict
    
def tf_wrapper(docs):
    docs_tf = []
    for doc in docs:
        tf = compute_tf(doc)
        docs_tf.append(tf)
    return docs_tf


    

def compute_idf(tf,vocab_freq):
    idf_dict = {}
    N = 1000
    for key in tf.keys():
        #count = vocab_freq[key]
        if key in vocab_freq.keys():
            count = vocab_freq[key]
        else:
            count = 0
        idf = math.log10(N/(1+count)) # New formula
         
        idf_dict[key] = idf
    return idf_dict

def idf_wrapper(docs_tf,vocab_freq):
    docs_idf = []
    for tf in docs_tf:
        idf = compute_idf(tf,vocab_freq)
        docs_idf.append(idf)
    return docs_idf

def count_docs(key,docs_tf):
    count = 0
    for tf in docs_tf:
        if key in tf.keys():
            count += 1
    return count

def calculate_vocab_freq(vocab,docs_tf):
    vocab_freq = {}
    for key in vocab:
        count = count_docs(key,docs_tf)
        vocab_freq[key] = count
    return vocab_freq


def compute_tfidf(tf,idf):
    tfidf = {}
    for key in tf.keys():
        val = tf[key] * idf[key]
        tfidf[key] = val
    return tfidf
    

def tfidf_wrapper(docs_tf,docs_idf):
    docs_tfidf = []
    for(tf,idf) in zip(docs_tf,docs_idf):
        tfidf = compute_tfidf(tf,idf)
        docs_tfidf.append(tfidf)
        
    return docs_tfidf
        
def create_vector(vocab,tfidf):
    vector = {}
    for key in vocab:
        if key in tfidf.keys():
            vector[key] = tfidf[key]
        else:
            vector[key] = 0.0
    return vector

def vectors_wrapper(vocab,docs_tfidf):
    vectors = []
    for tfidf in docs_tfidf:
        vector = create_vector(vocab,tfidf)
        vectors.append(vector)
    return vectors

def cosine(a,b):
    
    vec_a = []
    vec_b = []
    for key in a.keys():
        vec_a.append(a[key])
        vec_b.append(b[key])
    
    mod_a = np.sqrt(sum(np.square(vec_a)))
    mod_b = np.sqrt(sum(np.square(vec_b)))
    if mod_a == 0.0 or mod_b == 0.0 :
        score = 0
    else:
        score = np.dot(vec_a,vec_b)/(mod_a * mod_b)
    #print('score ',score)
    return score


def cosine_wrapper(vector,docs_vectors):
    dummy = []
    for vec in docs_vectors:
        dummy.append(cosine(vector,vec))
    return max(dummy)
    


def find_query_score(query,vocab_freq,docs_vectors):
    dummy_set = set()
    doc,_ = preprocess(query,dummy_set)
    tf = compute_tf(doc)
    idf = compute_idf(tf,vocab_freq)
    tfidf = compute_tfidf(tf,idf)

    vector = create_vector(vocab_freq.keys(),tfidf)
    score = cosine_wrapper(vector,docs_vectors)
    return score
    


def calculate_similarity(data,vocab_freq,docs_vectors):
    scores = []
    
    for i in range(len(data)):
        query_score = []
        ques = data.iloc[i]['question']
        for j in range(1,5):
            opt = data.iloc[i]['choice'+str(j)]
            query = ques + " " + opt
            score = find_query_score(query,vocab_freq,docs_vectors)
            query_score.append(score)
        scores.append(query_score)
    return scores

def find_max(score):
    max_val = max(score)
    ind = []
    for i in range(len(score)):
        c = ''
        if score[i] == max_val:
            if i == 0:
                c = 'A'
            elif i == 1:
                c = 'B'
            elif i == 2:
                c = 'C'
            else:
                c = 'D'
            ind.append(c)
    return ind


def find_labels(scores):
    labels = []
    for score in scores:
        label = find_max(score)
        labels.append(label)
    return labels

def find_accuracy(labels,answers):
    count = 0
    n = len(answers)
    for i in range(n):
        if answers[i] in labels[i]:
            l = len(labels[i])
            count += 1.0 / l
    acc = count / n
    return acc
#%%

#path = os.path.join(os.getcwd(),'NLP/asg5/data.txt')



vocab = set()
path = 'data.txt'
docs,vocab = load_data(path,vocab)

docs_tf = tf_wrapper(docs)

vocab_freq = calculate_vocab_freq(vocab,docs_tf)
docs_idf = idf_wrapper(docs_tf,vocab_freq)

docs_tfidf = tfidf_wrapper(docs_tf,docs_idf)

docs_vectors = vectors_wrapper(vocab,docs_tfidf)

#%%
pickle_data(docs,'docs')
pickle_data(vocab,'vocab')

pickle_data(docs_tf,'docs_tf')
pickle_data(docs_idf,'docs_idf')

pickle_data(docs_vectors,'vectors')


#%%



docs = load_pickle_data('docs.pkl')
docs_tf = load_pickle_data('docs_tf.pkl')
docs_idf = load_pickle_data('docs_idf.pkl')

vocab = load_pickle_data('vocab.pkl')
#docs_vectors = load_pickle_data('vectors.pkl')
docs_vectors = vectors_wrapper(vocab,docs_tfidf)


#%%


N = len(docs_vectors)
V = len(vocab)
print('Num of docs :',N)
print('Vocab Size : ',V)

#%% Load Queries
#path = os.path.join(os.getcwd(),'NLP/asg5/samp.jsonl')
path = 'Challenge.jsonl'
query_data = q.load_json(path)

#%%
#scores = calculate_similarity(query_data,vocab_freq,docs_vectors)
#pickle_data(scores,'scores')
#%%

scores = load_pickle_data('scores.pkl')

labels = find_labels(scores)

acc = find_accuracy(labels,query_data['answer'])
print('Accuracy : ',acc)