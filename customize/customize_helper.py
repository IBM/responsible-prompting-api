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

__author__ = "Vagner Santana, Melina Alberio, Cassia Sanctos, Ashwath Vaithinathan Aravindan, Luan Soares de Souza and Tiago Machado"
__copyright__ = "IBM Corporation 2024"
__credits__ = ["Vagner Santana, Melina Alberio, Cassia Sanctos, Ashwath Vaithinathan Aravindan, Luan Soares de Souza, Tiago Machado"]
__license__ = "Apache 2.0"
__version__ = "0.0.1"

import os
import json
import numpy as np
import math
from sentence_transformers import SentenceTransformer

from utils import get_distance

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

# Returns the centroid for a given value
def get_centroid(v, dimension = 384, k = 10):
    if len(v['prompts']) == 0:
        raise ValueError("List of prompts must not be empty")

    embeddings = np.array([p['embedding'] for p in v['prompts']])
    value_centroid = embeddings.mean(axis=0)

    if len(v['prompts']) > k:
        distances = np.array([get_distance(value_centroid, emb) for emb in embeddings])
        k_nearest_indices = np.argsort(distances)[:k]
        centroid = np.mean(embeddings[k_nearest_indices], axis=0)
        return centroid.tolist()
    else:
        return value_centroid.tolist()

def populate_embeddings(prompt_json, model_path, prompts_embeddings):
    errors, successes = 0, 0
    for v in prompt_json['positive_values']:
        for p in v['prompts']:
            if (p['text'] in prompts_embeddings):
                p['embedding'] = prompts_embeddings[p['text']]
            else:            
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
            if (p['text'] in prompts_embeddings):
                p['embedding'] = prompts_embeddings[p['text']]
            else:
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

# load existing data from a JSON file
def load_json(json_out_file):
    if os.path.exists(json_out_file):
        with open(json_out_file, 'r') as infile:
            return json.load(infile)
    else:
        return None