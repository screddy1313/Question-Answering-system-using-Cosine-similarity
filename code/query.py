# -*- coding: utf-8 -*-
"""
Created on Sat Nov  2 11:36:20 2019

@author: sarat
"""

import json
import os
import pandas as pd
#path = os.path.join(os.getcwd(),'NLP/asg5/samp.jsonl')
#path = 'test.jsonl'

def append_data(result):
    samp = []
    samp.append(result['question']['stem'])
    
    choices = result['question']['choices']
    
    for i in range(len(choices)):
        samp.append(choices[i]['text'])
    samp.append(result['answerKey']) 
    dummy = pd.DataFrame(data = samp).T
    return dummy

def load_json(path):
    
    with open(path, 'r') as json_file:
        json_list = list(json_file)
        data = pd.DataFrame()
    for json_str in json_list:
        
        result = json.loads(json_str)
        df = append_data(result)
        data = pd.concat([data,df],axis = 0)
    
    data = data.reset_index(drop = True)
    
    cols = ['question','choice1','choice2','choice3','choice4','answer']
    data.columns = cols
   
    samp = data['answer']
    samp.value_counts()
    samp[samp == '1'] = 'A'
    samp[samp == '2'] = 'B'
    samp[samp == '3'] = 'C'
    samp[samp == '4'] = 'D'

    return data

#data = load_json(path)



