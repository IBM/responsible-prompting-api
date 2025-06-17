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
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import os
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
    prompt_json = json.load(open(json_file))
    return prompt_json

def get_embedding_func(inference = 'huggingface', **kwargs):
    if inference == 'local':
        if 'model_id' not in kwargs:
            raise TypeError("Missing required argument: model_id")
        model = SentenceTransformer(kwargs['model_id'])

        def embedding_fn(texts):
            return model.encode(texts).tolist()

    elif inference == 'huggingface':
        if 'api_url' not in kwargs:
            raise TypeError("Missing required argument: api_url")
        if 'headers' not in kwargs:
            raise TypeError("Missing required argument: headers")
        
        def embedding_fn(texts):
            response = requests.post(kwargs['api_url'], headers=kwargs['headers'], json={"inputs": texts, "options":{"wait_for_model":True}})
            return response.json()
    else:
        raise ValueError(f"Inference type {inference} is not supported. Please choose one of ['local', 'huggingface'].")
    
    return embedding_fn

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

def recommend_prompt(
    prompt,
    prompt_json,
    embedding_fn = None,
    add_lower_threshold = 0.3,
    add_upper_threshold = 0.5,
    remove_lower_threshold = 0.1,
    remove_upper_threshold = 0.5,
    umap_model = None
):
    """
    Function that recommends prompts additions or removals.

    Args:
        prompt: The entered prompt text.
        prompt_json: Json file populated with embeddings.
        embedding_fn: Embedding function to convert prompt sentences into embeddings.
        If None, uses all-MiniLM-L6-v2 run locally.
        add_lower_threshold: Lower threshold for sentence addition,
        the default value is 0.3.
        add_upper_threshold: Upper threshold for sentence addition,
        the default value is 0.5.
        remove_lower_threshold: Lower threshold for sentence removal,
        the default value is 0.3.
        remove_upper_threshold: Upper threshold for sentence removal,
        the default value is 0.5.
        umap_model: Umap model used for visualization.
        If None, the projected embeddings of input sentences will not be returned.

    Returns:
        Prompt values to add or remove.

    Raises:
        Nothing.
    """
    if embedding_fn is None:
        # Use all-MiniLM-L6-v2 locally by default
        embedding_fn = get_embedding_func('local', model_id='sentence-transformers/all-MiniLM-L6-v2')

    # Output initialization
    out, out['input'], out['add'], out['remove'] = {}, {}, {}, {}
    input_items, items_to_add, items_to_remove = [], [], []

    # Spliting prompt into sentences
    input_sentences = split_into_sentences(prompt)

    # TODO: Request embeddings for input an d store in a input_embeddingS

    # Recommendation of values to add to the current prompt
    # Using only the last sentence for the add recommendation
    input_embedding = embedding_fn(input_sentences[-1])
    input_embedding = np.array(input_embedding)

    sentence_embeddings = np.array(
        [v['centroid'] for v in prompt_json['positive_values']]
    )

    similarities_positive_sent = cosine_similarity(np.expand_dims(input_embedding, axis=0), sentence_embeddings)[0, :]

    for value_idx, v in enumerate(prompt_json['positive_values']):
        # Dealing with values without prompts and makinig sure they have the same dimensions
        if(len(v['centroid']) != len(input_embedding)):
            continue

        if(similarities_positive_sent[value_idx] < add_lower_threshold):
            continue

        value_sents_similarity = cosine_similarity(
            np.expand_dims(input_embedding, axis=0),
            np.array([p['embedding'] for p in v['prompts']])
        )[0, :]
        closer_prompt_idxs = np.nonzero((add_lower_threshold < value_sents_similarity) & (value_sents_similarity < add_upper_threshold))[0]

        for idx in closer_prompt_idxs:
            items_to_add.append({
                'value': v['label'],
                'prompt': v['prompts'][idx]['text'],
                'similarity': value_sents_similarity[idx],
                'x': v['prompts'][idx]['x'],
                'y': v['prompts'][idx]['y']
            })
        out['add'] = items_to_add

    inp_sentence_embeddings = np.array([embedding_fn(sent) for sent in input_sentences])
    pairwise_similarities = cosine_similarity(
        inp_sentence_embeddings,
        np.array([v['centroid'] for v in prompt_json['negative_values']])
    )

    # Recommendation of values to remove from the current prompt
    for sent_idx, sentence in enumerate(input_sentences):
        input_embedding = inp_sentence_embeddings[sent_idx]
        if umap_model:
            # Obtaining XY coords for input sentences from a parametric UMAP model
            if(len(prompt_json['negative_values'][0]['centroid']) == len(input_embedding) and sentence != ''):
                embeddings_umap = umap_model.transform(np.expand_dims(pd.DataFrame(input_embedding).squeeze(), axis=0))
                input_items.append({
                    'sentence': sentence,
                    'x': str(embeddings_umap[0][0]),
                    'y': str(embeddings_umap[0][1])
                })

        for value_idx, v in enumerate(prompt_json['negative_values']):
            # Dealing with values without prompts and making sure they have the same dimensions
            if(len(v['centroid']) != len(input_embedding)):
                continue
            if(pairwise_similarities[sent_idx][value_idx] < remove_lower_threshold):
                continue

            # A more restrict threshold is used here to prevent false positives
            # The sentence_threshold is being used to indicate that there must be a sentence in the prompt that is similiar to one of our adversarial prompts
            # So, yes, we want to recommend the removal of something adversarial we've found
            value_sents_similarity = cosine_similarity(
                np.expand_dims(input_embedding, axis=0),
                np.array([p['embedding'] for p in v['prompts']])
            )[0, :]
            closer_prompt_idxs = np.nonzero(value_sents_similarity > remove_upper_threshold)[0]

            for idx in closer_prompt_idxs:
                items_to_remove.append({
                    'value': v['label'],
                    'sentence': sentence,
                    'sentence_index': sent_idx,
                    'closest_harmful_sentence': v['prompts'][idx]['text'],
                    'similarity': value_sents_similarity[idx],
                    'x': v['prompts'][idx]['x'],
                    'y': v['prompts'][idx]['y']
                })
            out['remove'] = items_to_remove

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

def get_thresholds(
    prompts,
    prompt_json,
    embedding_fn = None,
):
    """
    Function that recommends thresholds given an array of prompts.

    Args:
        prompts: The array with samples of prompts to be used in the system.
        prompt_json: Sentences to be forwarded to the recommendation endpoint.
        embedding_fn: Embedding function to convert prompt sentences into embeddings.
        If None, uses all-MiniLM-L6-v2 run locally.

    Returns:
        A map with thresholds for the sample prompts and the informed model.

    Raises:
        Nothing.
    """

    if embedding_fn is None:
        embedding_fn = get_embedding_func('local', model_id='sentence-transformers/all-MiniLM-L6-v2')

    add_similarities = []
    remove_similarities = []

    for p_id, p in enumerate(prompts):
        out = recommend_prompt(p, prompt_json, embedding_fn, 0, 1, 0, 0, None) # Wider possible range

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