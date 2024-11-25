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
Python helper function to get HF credentials.
"""

__author__ = "Vagner Santana, Melina Alberio, Cassia Sanctos and Tiago Machado"
__copyright__ = "IBM Corporation 2024"
__credits__ = ["Vagner Santana, Melina Alberio, Cassia Sanctos, Tiago Machado"]
__license__ = "Apache 2.0"
__version__ = "0.0.1"

import os

def get_credentials():
    """
    Function that loads HF credentials from env file.

    Args:
        None.

    Returns:
        hf_token: personal HuggingFace token.
        hf_url: HuggingFace url to be used.

    Raises:
        Nothing.
    """
    # Loading hugging face token from env file
    hf_token = os.environ.get('HF_TOKEN')
    hf_url = os.environ.get('HF_URL')

    return hf_token, hf_url