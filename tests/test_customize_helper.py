"""
Unit tests for customize/customize_helper.py
"""

import pytest
import json
import math
import os
import tempfile
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock

# Import the module under test
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from customize.customize_helper import (
    query_model,
    get_centroid,
    populate_embeddings,
    populate_centroids,
)

class TestGetCentroid:
    """Tests for the get_centroid function."""

    def test_single_prompt_centroid_equals_embedding(self):
        """Centroid of single prompt should equal its embedding."""
        value = {
            'prompts': [{'embedding': [1.0, 2.0, 3.0]}]
        }
        centroid = get_centroid(value, dimension=3, k=10)
        assert centroid == [1.0, 2.0, 3.0]

    def test_multiple_prompts_average(self):
        """Centroid should be average of embeddings when <= k prompts."""
        value = {
            'prompts': [
                {'embedding': [0.0, 0.0, 0.0]},
                {'embedding': [2.0, 4.0, 6.0]}
            ]
        }
        centroid = get_centroid(value, dimension=3, k=10)
        assert centroid == [1.0, 2.0, 3.0]

    def test_k_nearest_filtering(self):
        """When prompts > k, only k nearest should be used."""
        # Create prompts where some are far from the initial centroid
        value = {
            'prompts': [
                {'embedding': [1.0, 1.0]},
                {'embedding': [2.0, 2.0]},
                {'embedding': [3.0, 3.0]},
                {'embedding': [100.0, 100.0]},  # outlier
            ]
        }
        # With k=3, the outlier should be excluded
        centroid = get_centroid(value, dimension=2, k=3)
        # All 3 nearest should be [2.0, 2.0]
        assert centroid == [2.0, 2.0]

    def test_empty_prompts(self):
        """Empty prompts should return zero centroid."""
        value = {'prompts': []}
        with pytest.raises(ValueError):
            get_centroid(value, dimension=3, k=10)

class TestPopulateEmbeddings:
    """Tests for the populate_embeddings function."""

    @patch('customize.customize_helper.query_model')
    def test_uses_cached_embeddings(self, mock_query):
        """Test that cached embeddings are used when available."""
        prompt_json = {
            'positive_values': [
                {'prompts': [{'text': 'test', 'embedding': []}]}
            ],
            'negative_values': []
        }
        prompts_embeddings = {'test': [0.1, 0.2, 0.3]}
        
        result = populate_embeddings(prompt_json, "model/path", prompts_embeddings)
        
        # Should not call query_model when embedding is in cache
        mock_query.assert_not_called()
        assert result['positive_values'][0]['prompts'][0]['embedding'] == [0.1, 0.2, 0.3]

    @patch('customize.customize_helper.query_model')
    def test_queries_missing_embeddings(self, mock_query):
        """Test that missing embeddings are queried."""
        mock_query.return_value = np.array([0.1, 0.2, 0.3])
        
        prompt_json = {
            'positive_values': [
                {'prompts': [{'text': 'test', 'embedding': []}]}
            ],
            'negative_values': []
        }
        
        result = populate_embeddings(prompt_json, "model/path", {})
        
        mock_query.assert_called_once_with('test', "model/path")
        assert result['positive_values'][0]['prompts'][0]['embedding'] == [0.1, 0.2, 0.3]

    @patch('customize.customize_helper.query_model')
    def test_skips_non_empty_embeddings(self, mock_query):
        """Test that existing embeddings are not overwritten."""
        prompt_json = {
            'positive_values': [
                {'prompts': [{'text': 'test', 'embedding': [1.0, 2.0, 3.0]}]}
            ],
            'negative_values': []
        }
        
        result = populate_embeddings(prompt_json, "model/path", {})
        
        mock_query.assert_not_called()
        assert result['positive_values'][0]['prompts'][0]['embedding'] == [1.0, 2.0, 3.0]

    @patch('customize.customize_helper.query_model')
    def test_handles_negative_values(self, mock_query):
        """Test that negative values are also processed."""
        mock_query.return_value = np.array([0.4, 0.5, 0.6])
        
        prompt_json = {
            'positive_values': [],
            'negative_values': [
                {'prompts': [{'text': 'negative test', 'embedding': []}]}
            ]
        }
        
        result = populate_embeddings(prompt_json, "model/path", {})
        
        mock_query.assert_called_once_with('negative test', "model/path")
        assert result['negative_values'][0]['prompts'][0]['embedding'] == [0.4, 0.5, 0.6]


class TestPopulateCentroids:
    """Tests for the populate_centroids function."""

    @patch('customize.customize_helper.get_centroid')
    def test_populates_positive_centroids(self, mock_centroid):
        """Test centroid population for positive values."""
        mock_centroid.return_value = [0.1, 0.2, 0.3]
        
        prompt_json = {
            'positive_values': [{'prompts': [{'embedding': [1.0, 2.0, 3.0]}]}],
            'negative_values': []
        }
        
        result = populate_centroids(prompt_json)
        
        assert result['positive_values'][0]['centroid'] == [0.1, 0.2, 0.3]

    @patch('customize.customize_helper.get_centroid')
    def test_populates_negative_centroids(self, mock_centroid):
        """Test centroid population for negative values."""
        mock_centroid.return_value = [0.4, 0.5, 0.6]
        
        prompt_json = {
            'positive_values': [],
            'negative_values': [{'prompts': [{'embedding': [4.0, 5.0, 6.0]}]}]
        }
        
        result = populate_centroids(prompt_json)
        
        assert result['negative_values'][0]['centroid'] == [0.4, 0.5, 0.6]