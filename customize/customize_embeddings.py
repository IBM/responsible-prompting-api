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
Python function to customize json sentences locally.
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
import customize_helper

# Sentence transformer model HF
model_path = 'models/all-MiniLM-L6-v2'
model_id = model_path.split("/")[1]

# INPUT FILE
# Default file with empty embeddings
json_in_file = 'prompt-sentences-main/prompt_sentences.json'
json_in_file_name = json_in_file.split(".json")[0]

# OUTPUT FILE
json_out_file_name = f'{json_in_file_name}-{model_id}.json'

# check if the output file already exists
if os.path.exists(json_out_file_name):
    try:
        # Load existing data from the output file
        existing_data = customize_helper.load_json(json_out_file_name)
    except Exception as e:
        existing_data = None

# hashmap 
prompts_embeddings = {}
if existing_data:
    for d in existing_data["positive_values"]:
        for p in d["prompts"]:
            prompts_embeddings[p["text"]] = p["embedding"]
    for d in existing_data["negative_values"]:
        for p in d["prompts"]:
            prompts_embeddings[p["text"]] = p["embedding"]



prompt_json = json.load(open(json_in_file))
prompt_json_embeddings = customize_helper.populate_embeddings(prompt_json, model_path, prompts_embeddings)
prompt_json_centroids = customize_helper.populate_centroids(prompt_json_embeddings)
customize_helper.save_json(prompt_json_centroids, json_out_file_name)