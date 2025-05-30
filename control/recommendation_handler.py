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
Python lib to recommend prompts.
"""

__author__ = "Vagner Santana, Melina Alberio, Cassia Sanctos and Tiago Machado"
__copyright__ = "IBM Corporation 2024"
__credits__ = ["Vagner Santana, Melina Alberio, Cassia Sanctos, Tiago Machado"]
__license__ = "Apache 2.0"
__version__ = "0.0.1"

import requests
import json
import math
import re
import warnings
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import os
#os.environ['TRANSFORMERS_CACHE'] ="./models/allmini/cache"
import os.path
from sentence_transformers import SentenceTransformer
from umap import UMAP
import tensorflow as tf
from umap.parametric_umap import ParametricUMAP, load_ParametricUMAP
from sentence_transformers import SentenceTransformer

def populate_json(json_file_path = './prompt-sentences-main/prompt_sentences-all-minilm-l6-v2.json',
                    existing_json_populated_file_path = './prompt-sentences-main/prompt_sentences-all-minilm-l6-v2.json'):
    """
    Function that receives a default json file with
    empty embeddings and checks whether there is a
    partially populated json file.

    Args:
        json_file_path: Path to json default file with
        empty embeddings.
        existing_json_populated_file_path: Path to partially
        populated json file.

    Returns:
        A json.

    Raises:
        Exception when json file can't be loaded.
    """
    json_file = json_file_path
    if(os.path.isfile(existing_json_populated_file_path)):
        json_file = existing_json_populated_file_path
    try:
        prompt_json = json.load(open(json_file))
        json_error = None
        return prompt_json, json_error
    except Exception as e:
        json_error = e
        print(f'Error when loading sentences json file: {json_error}')
        prompt_json = None
        return prompt_json, json_error

def query(texts, api_url, headers):
    """
    Function that requests embeddings for a given sentence.

    Args:
        texts: The sentence or entered prompt text.
        api_url: API url for HF request.
        headers: Content headers for HF request.

    Returns:
        A json with the sentence embeddings.

    Raises:
        Warning: Warns about sentences that have more
        than 256 words.
    """
    for t in texts:
        n_words = len(re.split(r"\s+", t))
        if(n_words > 256):
            # warning in case of prompts longer than 256 words
            warnings.warn("Warning: Sentence provided is longer than 256 words. Model all-MiniLM-L6-v2 expects sentences up to 256 words.")
            warnings.warn("Word count:{}".format(n_words))
    if('sentence-transformers/all-MiniLM-L6-v2' in api_url):
        model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        out = model.encode(texts).tolist()
    else:
        response = requests.post(api_url, headers=headers, json={"inputs": texts, "options":{"wait_for_model":True}})
        out = response.json()
    return out

def split_into_sentences(prompt):
    """
    Function that splits the input text into sentences based
    on punctuation (.!?). The regular expression pattern
    '(?<=[.!?]) +' ensures that we split after a sentence-ending
    punctuation followed by one or more spaces.

    Args:
        prompt: The entered prompt text.

    Returns:
        A list of extracted sentences.

    Raises:
        Nothing.
    """
    sentences = re.split(r'(?<=[.!?]) +', prompt)
    return sentences


def get_similarity(embedding1, embedding2):
    """
    Function that returns cosine similarity between
    two embeddings.

    Args:
        embedding1: first embedding.
        embedding2: second embedding.

    Returns:
        The similarity value.

    Raises:
        Nothing.
    """
    v1 = np.array( embedding1 ).reshape( 1, -1 )
    v2 = np.array( embedding2 ).reshape( 1, -1 )
    similarity = cosine_similarity( v1, v2 )
    return similarity[0, 0]

def get_distance(embedding1, embedding2):
    """
    Function that returns euclidean distance between
    two embeddings.

    Args:
        embedding1: first embedding.
        embedding2: second embedding.

    Returns:
        The euclidean distance value.

    Raises:
        Nothing.
    """
    total = 0
    if(len(embedding1) != len(embedding2)):
        return math.inf
    for i, obj in enumerate(embedding1):
        total += math.pow(embedding2[0][i] - embedding1[0][i], 2)
    return(math.sqrt(total))

def sort_by_similarity(e):
    """
    Function that sorts by similarity.

    Args:
        e:

    Returns:
        The sorted similarity value.

    Raises:
        Nothing.
    """
    return e['similarity']

def recommend_prompt(prompt, prompt_json, api_url, headers, add_lower_threshold = 0.3,
                     add_upper_threshold = 0.5, remove_lower_threshold = 0.1,
                     remove_upper_threshold = 0.5, model_id = 'sentence-transformers/all-minilm-l6-v2'):
    """
    Function that recommends prompts additions or removals.

    Args:
        prompt: The entered prompt text.
        prompt_json: Json file populated with embeddings.
        api_url: API url for HF request.
        headers: Content headers for HF request.
        add_lower_threshold: Lower threshold for sentence addition,
        the default value is 0.3.
        add_upper_threshold: Upper threshold for sentence addition,
        the default value is 0.5.
        remove_lower_threshold: Lower threshold for sentence removal,
        the default value is 0.3.
        remove_upper_threshold: Upper threshold for sentence removal,
        the default value is 0.5.
        model_id: Id of the model, the default value is all-minilm-l6-v2 movel.

    Returns:
        Prompt values to add or remove.

    Raises:
        Nothing.
    """
    if(model_id == 'baai/bge-large-en-v1.5' ):
        json_file = './prompt-sentences-main/prompt_sentences-bge-large-en-v1.5.json'
        umap_folder = './models/umap/BAAI/bge-large-en-v1.5/'
    elif(model_id == 'intfloat/multilingual-e5-large'):
        json_file = './prompt-sentences-main/prompt_sentences-multilingual-e5-large.json'
        umap_folder = './models/umap/intfloat/multilingual-e5-large/'
    else: # fall back to all-minilm as default
        json_file = './prompt-sentences-main/prompt_sentences-all-minilm-l6-v2.json'
        umap_folder = './models/umap/sentence-transformers/all-MiniLM-L6-v2/'

    # Loading the encoder and config separately due to a bug
    encoder = tf.keras.models.load_model( umap_folder )
    with open( f"{umap_folder}umap_config.json", "r" ) as f:
        config = json.load( f )
    umap_model = ParametricUMAP( encoder=encoder, **config )
    prompt_json = json.load( open( json_file ) )

    # Output initialization
    out, out['input'], out['add'], out['remove'] = {}, {}, {}, {}
    input_items, items_to_add, items_to_remove = [], [], []

    # Spliting prompt into sentences
    input_sentences = split_into_sentences(prompt)

    # TODO: Request embeddings for input an d store in a input_embeddingS

    # Recommendation of values to add to the current prompt
    # Using only the last sentence for the add recommendation
    input_embedding = query(input_sentences[-1], api_url, headers)
    for v in prompt_json['positive_values']:
        # Dealing with values without prompts and makinig sure they have the same dimensions
        if(len(v['centroid']) == len(input_embedding)):
            if(get_similarity(pd.DataFrame(input_embedding), pd.DataFrame(v['centroid'])) > add_lower_threshold):
                closer_prompt = -1
                for p in v['prompts']:
                    d_prompt = get_similarity(pd.DataFrame(input_embedding), pd.DataFrame(p['embedding']))
                    # The sentence_threshold is being used as a ceiling meaning that for high similarities the sentence/value might already be presente in the prompt
                    # So, we don't want to recommend adding something that is already there
                    if(d_prompt > closer_prompt and d_prompt > add_lower_threshold and d_prompt < add_upper_threshold):
                        closer_prompt = d_prompt
                        items_to_add.append({
                        'value': v['label'],
                        'prompt': p['text'],
                        'similarity': d_prompt,
                        'x': p['x'],
                        'y': p['y']})
                out['add'] = items_to_add

    # Recommendation of values to remove from the current prompt
    i = 0

    # Recommendation of values to remove from the current prompt
    for sentence in input_sentences:
        input_embedding = query(sentence, api_url, headers) # remote
        # Obtaining XY coords for input sentences from a parametric UMAP model
        if(len(prompt_json['negative_values'][0]['centroid']) == len(input_embedding) and sentence != ''):
            embeddings_umap = umap_model.transform(tf.expand_dims(pd.DataFrame(input_embedding), axis=0))
            input_items.append({
                'sentence': sentence,
                'x': str(embeddings_umap[0][0]),
                'y': str(embeddings_umap[0][1])
            })

        for v in prompt_json['negative_values']:
        # Dealing with values without prompts and makinig sure they have the same dimensions
            if(len(v['centroid']) == len(input_embedding)):
                if(get_similarity(pd.DataFrame(input_embedding), pd.DataFrame(v['centroid'])) > remove_lower_threshold):
                    closer_prompt = -1
                    for p in v['prompts']:
                        d_prompt = get_similarity(pd.DataFrame(input_embedding), pd.DataFrame(p['embedding']))
                        # A more restrict threshold is used here to prevent false positives
                        # The sentence_threshold is being used to indicate that there must be a sentence in the prompt that is similiar to one of our adversarial prompts
                        # So, yes, we want to recommend the removal of something adversarial we've found
                        if(d_prompt > closer_prompt and d_prompt > remove_upper_threshold):
                            closer_prompt = d_prompt
                            items_to_remove.append({
                            'value': v['label'],
                            'sentence': sentence,
                            'sentence_index': i,
                            'closest_harmful_sentence': p['text'],
                            'similarity': d_prompt,
                            'x': p['x'],
                            'y': p['y']})
                    out['remove'] = items_to_remove
        i += 1

    out['input'] = input_items

    out['add'] = sorted(out['add'], key=sort_by_similarity, reverse=True)
    values_map = {}
    for item in out['add'][:]:
        if(item['value'] in values_map):
            out['add'].remove(item)
        else:
            values_map[item['value']] = item['similarity']
    out['add'] = out['add'][0:5]

    out['remove'] = sorted(out['remove'], key=sort_by_similarity, reverse=True)
    values_map = {}
    for item in out['remove'][:]:
        if(item['value'] in values_map):
            out['remove'].remove(item)
        else:
            values_map[item['value']] = item['similarity']
    out['remove'] = out['remove'][0:5]
    return out

def get_thresholds(prompts, prompt_json, api_url, headers, model_id = 'sentence-transformers/all-minilm-l6-v2'):
    """
    Function that recommends thresholds given an array of prompts.

    Args:
        prompts: The array with samples of prompts to be used in the system.
        prompt_json: Sentences to be forwarded to the recommendation endpoint.
        model_id: Id of the model, the default value is all-minilm-l6-v2 model.

    Returns:
        A map with thresholds for the sample prompts and the informed model.

    Raises:
        Nothing.
    """
    # Array limits for retrieving the thresholds
    # if( len( prompts ) < 10 or len( prompts ) > 30 ):
    #     return -1
    add_similarities = []
    remove_similarities = []

    for p_id, p in enumerate(prompts):
        out = recommend_prompt(p, prompt_json, api_url, headers, 0, 1, 0, 0, model_id) # Wider possible range

        for r in out['add']:
            add_similarities.append(r['similarity'])
        for r in out['remove']:
            remove_similarities.append(r['similarity'])

    add_similarities_df = pd.DataFrame({'similarity': add_similarities})
    remove_similarities_df = pd.DataFrame({'similarity': remove_similarities})

    thresholds = {}
    thresholds['add_lower_threshold'] = round(add_similarities_df.describe([.1]).loc['10%', 'similarity'], 1)
    thresholds['add_higher_threshold'] = round(add_similarities_df.describe([.9]).loc['90%', 'similarity'], 1)
    thresholds['remove_lower_threshold'] = round(remove_similarities_df.describe([.1]).loc['10%', 'similarity'], 1)
    thresholds['remove_higher_threshold'] = round(remove_similarities_df.describe([.9]).loc['90%', 'similarity'], 1)

    return thresholds

def recommend_local(prompt, prompt_json, model_id, model_path = './models/all-MiniLM-L6-v2/', add_lower_threshold = 0.3,
                     add_upper_threshold = 0.5, remove_lower_threshold = 0.1,
                     remove_upper_threshold = 0.5):
    """
    Function that recommends prompts additions or removals
    using a local model.

    Args:
        prompt: The entered prompt text.
        prompt_json: Json file populated with embeddings.
        model_id: Id of the local model.
        model_path: Path to the local model.

    Returns:
        Prompt values to add or remove.

    Raises:
        Nothing.
    """
    if(model_id == 'baai/bge-large-en-v1.5' ):
        json_file = './prompt-sentences-main/prompt_sentences-bge-large-en-v1.5.json'
        umap_folder = './models/umap/BAAI/bge-large-en-v1.5/'
    elif(model_id == 'intfloat/multilingual-e5-large'):
        json_file = './prompt-sentences-main/prompt_sentences-multilingual-e5-large.json'
        umap_folder = './models/umap/intfloat/multilingual-e5-large/'
    else: # fall back to all-minilm as default
        json_file = './prompt-sentences-main/prompt_sentences-all-minilm-l6-v2.json'
        umap_folder = './models/umap/sentence-transformers/all-MiniLM-L6-v2/'

    # Loading the encoder and config separately due to a bug
    encoder = tf.keras.models.load_model( umap_folder )
    with open( f"{umap_folder}umap_config.json", "r" ) as f:
        config = json.load( f )
    umap_model = ParametricUMAP( encoder=encoder, **config )
    prompt_json = json.load( open( json_file ) )

    # Output initialization
    out, out['input'], out['add'], out['remove'] = {}, {}, {}, {}
    input_items, items_to_add, items_to_remove = [], [], []

    # Spliting prompt into sentences
    input_sentences = split_into_sentences(prompt)

    # Recommendation of values to add to the current prompt
    # Using only the last sentence for the add recommendation
    model = SentenceTransformer(model_path)
    input_embedding = model.encode(input_sentences[-1])

    for v in prompt_json['positive_values']:
        # Dealing with values without prompts and makinig sure they have the same dimensions
        if(len(v['centroid']) == len(input_embedding)):
            if(get_similarity(pd.DataFrame(input_embedding), pd.DataFrame(v['centroid'])) > add_lower_threshold):
                closer_prompt = -1
                for p in v['prompts']:
                    d_prompt = get_similarity(pd.DataFrame(input_embedding), pd.DataFrame(p['embedding']))
                    # The sentence_threshold is being used as a ceiling meaning that for high similarities the sentence/value might already be presente in the prompt
                    # So, we don't want to recommend adding something that is already there
                    if(d_prompt > closer_prompt and d_prompt > add_lower_threshold and d_prompt < add_upper_threshold):
                        closer_prompt = d_prompt
                        items_to_add.append({
                        'value': v['label'],
                        'prompt': p['text'],
                        'similarity': d_prompt,
                        'x': p['x'],
                        'y': p['y']})
                out['add'] = items_to_add

    # Recommendation of values to remove from the current prompt
    i = 0

    # Recommendation of values to remove from the current prompt
    for sentence in input_sentences:
        input_embedding = model.encode(sentence) # local
        # Obtaining XY coords for input sentences from a parametric UMAP model
        if(len(prompt_json['negative_values'][0]['centroid']) == len(input_embedding) and sentence != ''):
            embeddings_umap = umap_model.transform(tf.expand_dims(pd.DataFrame(input_embedding), axis=0))
            input_items.append({
                'sentence': sentence,
                'x': str(embeddings_umap[0][0]),
                'y': str(embeddings_umap[0][1])
            })

        for v in prompt_json['negative_values']:
        # Dealing with values without prompts and makinig sure they have the same dimensions
            if(len(v['centroid']) == len(input_embedding)):
                if(get_similarity(pd.DataFrame(input_embedding), pd.DataFrame(v['centroid'])) > remove_lower_threshold):
                    closer_prompt = -1
                    for p in v['prompts']:
                        d_prompt = get_similarity(pd.DataFrame(input_embedding), pd.DataFrame(p['embedding']))
                        # A more restrict threshold is used here to prevent false positives
                        # The sentence_threhold is being used to indicate that there must be a sentence in the prompt that is similiar to one of our adversarial prompts
                        # So, yes, we want to recommend the revolval of something adversarial we've found
                        if(d_prompt > closer_prompt and d_prompt > remove_upper_threshold):
                            closer_prompt = d_prompt
                            items_to_remove.append({
                            'value': v['label'],
                            'sentence': sentence,
                            'sentence_index': i,
                            'closest_harmful_sentence': p['text'],
                            'similarity': d_prompt,
                            'x': p['x'],
                            'y': p['y']})
                    out['remove'] = items_to_remove
        i += 1

    out['input'] = input_items

    out['add'] = sorted(out['add'], key=sort_by_similarity, reverse=True)
    values_map = {}
    for item in out['add'][:]:
        if(item['value'] in values_map):
            out['add'].remove(item)
        else:
            values_map[item['value']] = item['similarity']
    out['add'] = out['add'][0:5]

    out['remove'] = sorted(out['remove'], key=sort_by_similarity, reverse=True)
    values_map = {}
    for item in out['remove'][:]:
        if(item['value'] in values_map):
            out['remove'].remove(item)
        else:
            values_map[item['value']] = item['similarity']
    out['remove'] = out['remove'][0:5]
    return out
