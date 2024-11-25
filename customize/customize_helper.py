#!/usr/bin/env python
# coding: utf-8

# Copyright 2021, IBM Corporation. 
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Python helper function to customize json sentences locally.
"""

__author__ = "Vagner Santana, Melina Alberio, Cassia Sanctos and Tiago Machado"
__copyright__ = "IBM Corporation 2024"
__credits__ = ["Vagner Santana, Melina Alberio, Cassia Sanctos, Tiago Machado"]
__license__ = "Apache 2.0"
__version__ = "0.0.1"

import os
import json
import pandas as pd
import numpy as np
import math
from sentence_transformers import SentenceTransformer

# Requests embeddings for a given sentence
def query_model(texts, model_path):
    out = []
    model = SentenceTransformer(model_path)
    input_embedding = model.encode(texts)
    out.append(input_embedding)
    if( out != [] ):
        return out[0]
    else:
        return out

# Returns euclidean distance between two embeddings
def get_distance(embedding1, embedding2):
    total = 0    
    if( len(embedding1) != len(embedding2)):
        return math.inf
    
    for i, obj in enumerate(embedding1):
        total += math.pow(embedding2[0][i] - embedding1[0][i], 2)
    return(math.sqrt(total))

# Returns the centroid for a given value
def get_centroid(v, dimension = 384, k = 10):
    centroid = [0] * dimension
    count = 0            
    for p in v['prompts']:
        i = 0
        while i < len(p['embedding']):
            centroid[i] += p['embedding'][i]
            i += 1
        count += 1
    i = 0
    while i < len(centroid):
        centroid[i] /= count
        i += 1

    # Update centroid considering only the k-near elements
    if(len(v['prompts']) <= k):
        return centroid
    else:
        k_items = []
        k_prompts = []
        for p in v['prompts']:
            dist = get_distance(pd.DataFrame(centroid ), pd.DataFrame(p['embedding'])) 
            if(len(k_items) < k):
                k_items.append(p['embedding'])
                k_prompts.append(p['text'])
            else:
                i = 0
                while(i < len(k_items)):
                    if( dist < get_distance( pd.DataFrame(centroid), pd.DataFrame(k_items[i]))):
                        k_items[i] = p['embedding']
                        k_prompts[i] = p['text']
                        break
                    i += 1

        # Computing centroid only for the k-near elements
        for i, embedding  in enumerate(k_items):
            for j, dimension in enumerate(k_items[i]):
                centroid[j] += k_items[i][j]
        while i < len( centroid ):
            centroid[i] /= k
            i += 1
    return centroid

def populate_embeddings(prompt_json, model_path):
    errors, successess = 0, 0
    for v in prompt_json['positive_values']:
        for p in v['prompts']:
                if( p['text'] != '' and p['embedding'] == []): # only considering missing embeddings
                    embedding = query_model(p['text'], model_path)
                    if( 'error' in embedding ):
                        p['embedding'] = []
                        errors += 1
                    else: 
                        p['embedding'] = embedding.tolist()
                        #successes += 1
                        
    for v in prompt_json['negative_values']:
        for p in v['prompts']:
            if(p['text'] != '' and p['embedding'] == []):
                embedding = query_model(p['text'], model_path)
                if('error' in embedding):
                    p['embedding'] = []
                    errors += 1
                else: 
                    p['embedding'] = embedding.tolist()
                    #successes += 1
    return prompt_json

def populate_centroids(prompt_json):
    for v in prompt_json['positive_values']:
        v['centroid'] = get_centroid(v, dimension = 384, k = 10)
    for v in prompt_json['negative_values']:
        v['centroid'] = get_centroid(v, dimension = 384, k = 10)
    return prompt_json

    # Saving the embeddings for a specific LLM
def save_json(prompt_json, json_out_file_name):
    with open(json_out_file_name, 'w') as outfile:
        json.dump(prompt_json, outfile)