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

__author__ = "Vagner Santana, Melina Alberio, Cassia Sanctos and Tiago Machado"
__copyright__ = "IBM Corporation 2024"
__credits__ = ["Vagner Santana, Melina Alberio, Cassia Sanctos, Tiago Machado"]
__license__ = "Apache 2.0"
__version__ = "0.0.1"

from flask import Flask, request, render_template
from flask_cors import CORS, cross_origin
from flask_restful import Resource, Api, reqparse
import control.recommendation_handler as recommendation_handler
from helpers import get_credentials, authenticate_api, save_model
from helpers import get_credentials, authenticate_api, save_model, validate_json
import config as cfg

app = Flask(__name__, template_folder='templates')

# swagger configs
app.register_blueprint(cfg.SWAGGER_BLUEPRINT, url_prefix = cfg.SWAGGER_URL)

@app.route("/")
def index():
    return "Ready!"

@app.route("/recommend", methods=['GET'])
@cross_origin()
def recommend():
    hf_token, hf_url = get_credentials.get_credentials()
    api_url, headers = authenticate_api.authenticate_api(hf_token, hf_url)
    prompt_json, json_error = recommendation_handler.populate_json()
    if json_error is None:
        args = request.args
        print("args list = ", args)
        prompt = args.get("prompt")
        recommendation_json = recommendation_handler.recommend_prompt(prompt, prompt_json, 
                                                                        api_url, headers)
        return recommendation_json
    else:
        return render_template('invalid_json.html', error_message=json_error)
    

@app.route("/get_thresholds", methods=['GET'])
@cross_origin()
def get_thresholds():
    hf_token, hf_url = get_credentials.get_credentials()
    api_url, headers = authenticate_api.authenticate_api(hf_token, hf_url)
    prompt_json = recommendation_handler.populate_json()
    model_id = 'sentence-transformers/all-minilm-l6-v2'
    args = request.args
    print("args list = ", args)
    prompt = args.get("prompt")
    thresholds_json = recommendation_handler.get_thresholds(prompt, prompt_json, api_url, 
                                                            headers, model_id)
    return thresholds_json

@app.route("/recommend_local", methods=['GET'])
@cross_origin()
def recommend_local():
    model_id, model_path = save_model.save_model()
    prompt_json = recommendation_handler.populate_json()
    args = request.args
    print("args list = ", args)
    prompt = args.get("prompt")
    local_recommendation_json = recommendation_handler.recommend_local(prompt, prompt_json,
                                                                       model_id, model_path)
    return local_recommendation_json

if __name__=='__main__':
	app.run(host='0.0.0.0', port='8080', debug=True)
