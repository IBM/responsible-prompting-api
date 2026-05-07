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
Flask API app and routes.
"""

__author__ = "Vagner Santana, Melina Alberio, Cassia Sanctos, Ashwath Vaithinathan Aravindan, Luan Soares de Souza and Tiago Machado"
__copyright__ = "IBM Corporation 2024"
__credits__ = ["Vagner Santana, Melina Alberio, Cassia Sanctos, Ashwath Vaithinathan Aravindan, Luan Soares de Souza, Tiago Machado"]
__license__ = "Apache 2.0"
__version__ = "0.0.1"


from flask import Flask, request, jsonify
from flask_cors import CORS, cross_origin
from flask_restful import Resource, Api, reqparse
import control.recommendation_handler as recommendation_handler
from helpers import get_credentials, authenticate_api, save_model, inference
import config as cfg
import requests
import logging
import uuid
import json
import os
import pickle
import numpy as np
from functools import lru_cache

app = Flask(__name__)

# configure logging
logging.basicConfig(
    filename='app.log',  # Log file name
    level=logging.INFO,  # Log level (INFO, DEBUG, WARNING, ERROR, CRITICAL)
    format='%(asctime)s - %(levelname)s - %(message)s'  # Log message format
)

# access the app's logger
logger = app.logger
# create user id
id = str(uuid.uuid4())

# swagger configs
app.register_blueprint(cfg.SWAGGER_BLUEPRINT, url_prefix = cfg.SWAGGER_URL)
FRONT_LOG_FILE = 'front_log.json'

@lru_cache(maxsize=1)
def get_values_embedding_function():
    """
    Getting the embedding function for the /values endpoint.
    Cached to avoid reloading the model multiple times.
    
    Returns:
        Embedding function callable
    """
    model_id, model_path = save_model.save_model()
    return recommendation_handler.get_embedding_func(inference='local', model_id=model_path)

@lru_cache(maxsize=1)
def get_values_centroids():
    """
    Getting the positive and negative value centroids for the /values endpoint.
    Cached to avoid reloading the data multiple times.
    
    Returns:
        Dictionary with 'positive' and 'negative' centroid embeddings
    """
    prompt_json = recommendation_handler.populate_json()
    positive_category_centroid = {}
    negative_category_centroid = {}

    for category in prompt_json['positive_values']:
        positive_category_centroid[category['label']] = np.array(category['centroid'])

    for category in prompt_json['negative_values']:
        negative_category_centroid[category['label']] = np.array(category['centroid'])

    return {
        'positive': positive_category_centroid,
        'negative': negative_category_centroid
    }

@app.route("/")
def index():
    user_ip = request.remote_addr
    logger.info(f'USER {user_ip} - ID {id} - started the app')
    return "Ready!"

@app.route("/recommend", methods=['GET'])
@cross_origin()
def recommend():
    model_id, _ =save_model.save_model()
    prompt_json = recommendation_handler.populate_json()
    args = request.args
    print("args list = ", args)
    prompt = args.get("prompt")

    umap_model_file = './models/umap/sentence-transformers/all-MiniLM-L6-v2/umap.pkl'
    with open(umap_model_file, 'rb') as f:
        umap_model = pickle.load(f)

    # Embeddings from HF API
    # hf_token, hf_url = get_credentials.get_hf_credentials()
    # api_url, headers = authenticate_api.authenticate_api(hf_token, hf_url)
    # api_url = f'https://router.huggingface.co/hf-inference/models/{model_id}/pipeline/feature-extraction'
    # embedding_fn = recommendation_handler.get_embedding_func(inference='huggingface', model_id=model_id, api_url= api_url, headers = headers)

    # Embeddings from local inference
    embedding_fn = recommendation_handler.get_embedding_func(inference='local', model_id=model_id)

    recommendation_json = recommendation_handler.recommend_prompt(prompt, prompt_json, embedding_fn, umap_model=umap_model)

    user_ip = request.remote_addr
    logger.info(f'USER - {user_ip} - ID {id} - accessed recommend route')
    logger.info(f'RECOMMEND ROUTE - request: {prompt} response: {recommendation_json}')

    return recommendation_json

@app.route("/get_thresholds", methods=['GET'])
@cross_origin()
def get_thresholds():
    prompt_json = recommendation_handler.populate_json()
    prompts = [
        prompt.strip()
        for prompt in request.args.getlist("prompts")
        if prompt.strip()
    ]
    if not prompts:
        return jsonify({"error": "Missing required query parameter: prompts"}), 400

    model_id = request.headers.get("model_id") or "./models/all-MiniLM-L6-v2/"
    embedding_fn = recommendation_handler.get_embedding_func(inference='local', model_id=model_id)

    thresholds_json = recommendation_handler.get_thresholds(prompts, prompt_json, embedding_fn)
    return jsonify({key: float(value) for key, value in thresholds_json.items()})

@app.route("/recommend_local", methods=['GET'])
@cross_origin()
def recommend_local():
    model_id, _ = save_model.save_model()
    prompt_json = recommendation_handler.populate_json()
    args = request.args
    print("args list = ", args)
    prompt = args.get("prompt")
    
    umap_model_file = './models/umap/sentence-transformers/all-MiniLM-L6-v2/umap.pkl'
    with open(umap_model_file, 'rb') as f:
        umap_model = pickle.load(f)

    embedding_fn = recommendation_handler.get_embedding_func(inference='local', model_id=model_id)

    local_recommendation_json = recommendation_handler.recommend_prompt(prompt, prompt_json, embedding_fn, umap_model=umap_model)
    return local_recommendation_json

@app.route("/log", methods=['POST'])
@cross_origin()
def log():
    f_path = 'static/demo/log/'
    new_data = request.get_json()

    try:
        with open(f_path+FRONT_LOG_FILE, 'r') as f:
            existing_data = json.load(f)
    except FileNotFoundError:
        existing_data = []

    existing_data.update(new_data)

    #log_data = request.json
    with open(f_path+FRONT_LOG_FILE, 'w') as f:
        json.dump(existing_data, f)
    return jsonify({'message': 'Data added successfully', 'data': existing_data}), 201

@app.route("/demo_inference", methods=['GET'])
@cross_origin()
def demo_inference():
    args = request.args

    inference_provider = args.get('inference_provider', default='replicate')
    model_id = args.get('model_id', default="ibm-granite/granite-3.3-8b-instruct")
    temperature = args.get('temperature', default=0.5)
    max_new_tokens = args.get('max_new_tokens', default=1000)

    prompt = args.get('prompt')

    try:
        response = inference.INFERENCE_HANDLER[inference_provider](prompt, model_id, temperature, max_new_tokens)
        response.update({
            'inference_provider': inference_provider,
            'model_id': model_id,
            'temperature': temperature,
            'max_new_tokens': max_new_tokens,
        })

        return response
    except:
        return "Model Inference failed.", 500
    
@app.route("/values", methods=['GET'])
@cross_origin()
def get_values():
    """
    Getting positive and negative values for a given prompt using cached embedding function and centroids for performance.
    """
    args = request.args
    prompt = args.get("prompt")
    
    # validating input
    if not prompt:
        return jsonify({"error": "Missing required parameter: prompt"}), 400
    
    if not isinstance(prompt, str):
        return jsonify({"error": "Parameter 'prompt' must be a string"}), 400
    
    if len(prompt.strip()) == 0:
        return jsonify({"error": "Parameter 'prompt' cannot be empty"}), 400
    
    try:
        embedding_fn = get_values_embedding_function()
        centroids = get_values_centroids()
        
        values = recommendation_handler.get_values(
            prompt, 
            centroids['positive'], 
            centroids['negative'],
            embedding_fn
        )
        
        return jsonify(values)
    
    except Exception as e:
        logger.error(f'Error in /values endpoint: {str(e)}')
        return jsonify({"error": "Internal server error processing prompt"}), 500

if __name__=='__main__':
    debug_mode = os.getenv('FLASK_DEBUG', 'False').lower() in ['true', '1', 't']
    app.run(host='0.0.0.0', port='8080', debug=debug_mode)