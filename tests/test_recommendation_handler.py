#!/usr/bin/env python
# coding: utf-8

"""
Unit tests for control/recommendation_handler.py
"""

import pytest
import json
import math
import os
import tempfile
import numpy as np
from unittest.mock import patch, MagicMock

# Import the module under test
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from control.recommendation_handler import (
    populate_json,
    get_embedding_func,
    split_into_sentences,
    get_similarity,
    recommend_prompt,
    get_thresholds,
    get_values
)


class TestSplitIntoSentences:
    """Tests for the split_into_sentences function."""

    def test_single_sentence(self):
        """Test splitting a single sentence."""
        result = split_into_sentences("Hello world.")
        assert result == ["Hello world."]

    def test_multiple_sentences_period(self):
        """Test splitting sentences with periods."""
        result = split_into_sentences("First sentence. Second sentence. Third sentence.")
        assert result == ["First sentence.", "Second sentence.", "Third sentence."]

    def test_multiple_sentences_mixed_punctuation(self):
        """Test splitting with mixed punctuation."""
        result = split_into_sentences("Is this a question? Yes it is! And here's more.")
        assert result == ["Is this a question?", "Yes it is!", "And here's more."]

    def test_empty_string(self):
        """Test with empty string."""
        result = split_into_sentences("")
        assert result == [""]

    def test_no_punctuation(self):
        """Test sentence without punctuation."""
        result = split_into_sentences("This sentence has no ending punctuation")
        assert result == ["This sentence has no ending punctuation"]

    def test_multiple_sentences_ending_with_no_punctuation(self):
        """Test splitting with multiple sentences ending with no punctuation."""
        result = split_into_sentences("Is this a question? Yes it is! And here's more")
        assert result == ["Is this a question?", "Yes it is!", "And here's more"]

    def test_multiple_spaces(self):
        """Test handling of multiple spaces after punctuation."""
        result = split_into_sentences("First.  Second.")
        # The regex splits on one or more spaces after punctuation
        assert "First." in result
        assert "Second." in result


class TestGetSimilarity:
    """Tests for the get_similarity function."""

    def test_returns_similarity_value(self):
        """Test that similarity value is returned."""
        item = {'value': 'test', 'similarity': 0.75}
        assert get_similarity(item) == 0.75

    def test_sorting_list(self):
        """Test that items can be sorted using this function."""
        items = [
            {'similarity': 0.3},
            {'similarity': 0.9},
            {'similarity': 0.5}
        ]
        sorted_items = sorted(items, key=get_similarity, reverse=True)
        assert sorted_items[0]['similarity'] == 0.9
        assert sorted_items[1]['similarity'] == 0.5
        assert sorted_items[2]['similarity'] == 0.3

    def test_raises_error_when_similarity_key_missing(self):
        """Test that ValueError is raised when similarity key is missing."""
        item = {'value': 'test'}
        with pytest.raises(ValueError, match="Key 'similarity' not found"):
            get_similarity(item)


class TestGetEmbeddingFunc:
    """Tests for the get_embedding_func function."""

    @patch('control.recommendation_handler.SentenceTransformer')
    def test_local_inference(self, mock_transformer):
        """Test local inference mode."""
        mock_model = MagicMock()
        mock_model.encode.return_value = np.array([[0.1, 0.2, 0.3]])
        mock_transformer.return_value = mock_model

        embedding_fn = get_embedding_func('local', model_id='test-model')
        result = embedding_fn("test text")

        mock_transformer.assert_called_once_with('test-model')
        mock_model.encode.assert_called_once_with("test text")
        assert result == [[0.1, 0.2, 0.3]]

    def test_local_inference_missing_model_id(self):
        """Test local inference without model_id raises error."""
        with pytest.raises(TypeError, match="Missing required argument: model_id"):
            get_embedding_func('local')

    @patch('control.recommendation_handler.requests.post')
    def test_huggingface_inference(self, mock_post):
        """Test HuggingFace inference mode."""
        mock_response = MagicMock()
        mock_response.json.return_value = [[0.1, 0.2, 0.3]]
        mock_post.return_value = mock_response

        embedding_fn = get_embedding_func(
            'huggingface',
            api_url='https://api.example.com',
            headers={'Authorization': 'Bearer token'}
        )
        result = embedding_fn("test text")

        mock_post.assert_called_once()
        assert result == [[0.1, 0.2, 0.3]]

    def test_huggingface_inference_missing_api_url(self):
        """Test HuggingFace inference without api_url raises error."""
        with pytest.raises(TypeError, match="Missing required argument: api_url"):
            get_embedding_func('huggingface', headers={})

    def test_huggingface_inference_missing_headers(self):
        """Test HuggingFace inference without headers raises error."""
        with pytest.raises(TypeError, match="Missing required argument: headers"):
            get_embedding_func('huggingface', api_url='https://api.example.com')

    def test_unsupported_inference_type(self):
        """Test unsupported inference type raises error."""
        with pytest.raises(ValueError, match="Inference type invalid is not supported"):
            get_embedding_func('invalid')


class TestRecommendPrompt:
    """Tests for the recommend_prompt function."""

    def create_mock_prompt_json(self):
        """Create a mock prompt JSON structure for testing."""
        return {
            'positive_values': [
                {
                    'label': 'accountability',
                    'centroid': [0.1] * 384,
                    'prompts': [
                        {'text': 'Be accountable', 'embedding': [0.1] * 384, 'x': 0.5, 'y': 0.5}
                    ]
                }
            ],
            'negative_values': [
                {
                    'label': 'harmful',
                    'centroid': [0.1] * 200 + [-0.1] * 184,
                    'prompts': [
                        {'text': 'Harmful content', 'embedding': [0.1] * 200 + [-0.1] * 184, 'x': -0.5, 'y': -0.5}
                    ]
                }
            ]
        }

    def test_recommend_prompt(self):
        """Test that recommend_prompt returns expected structure."""
        prompt_json = self.create_mock_prompt_json()
        
        # Create a mock embedding function
        def mock_embedding_fn(text):
            return [0.1] * 284 + [-0.1] * 100
        
        result = recommend_prompt(
            "Test prompt.",
            prompt_json,
            embedding_fn=mock_embedding_fn
        )
        
        assert 'input' in result
        assert 'add' in result
        assert 'remove' in result

        assert len(result['add']) == 1
        assert result['add'][0]['value'] == 'accountability'

        assert len(result['remove']) == 1
        assert result['remove'][0]['value'] == 'harmful'

    def test_recommend_prompt_empty_prompt(self):
        """Test behavior with empty prompt."""
        prompt_json = self.create_mock_prompt_json()
        
        def mock_embedding_fn(text):
            return [0.2] * 384
        
        result = recommend_prompt(
            "",
            prompt_json,
            embedding_fn=mock_embedding_fn
        )
        
        assert 'input' in result
        assert 'add' in result
        assert 'remove' in result

    def test_recommend_prompt_uses_default_embedding_fn(self):
        """Test that default embedding function is used when none provided."""
        prompt_json = self.create_mock_prompt_json()
        
        with patch('control.recommendation_handler.get_embedding_func') as mock_get_fn:
            mock_fn = MagicMock(return_value=[0.1] * 384)
            mock_get_fn.return_value = mock_fn
            
            result = recommend_prompt(
                "Test prompt.",
                prompt_json,
                embedding_fn=None
            )
            
            mock_get_fn.assert_called_once_with(
                'local',
                model_id='sentence-transformers/all-MiniLM-L6-v2'
            )


class TestGetThresholds:
    """Tests for the get_thresholds function."""

    def create_mock_prompt_json(self):
        """Create a mock prompt JSON structure for testing."""
        return {
            'positive_values': [
                {
                    'label': 'accountability',
                    'centroid': [0.1] * 384,
                    'prompts': [
                        {'text': 'Be accountable', 'embedding': [0.1] * 384, 'x': 0.5, 'y': 0.5}
                    ]
                }
            ],
            'negative_values': [
                {
                    'label': 'harmful',
                    'centroid': [0.9] * 384,
                    'prompts': [
                        {'text': 'Harmful content', 'embedding': [0.9] * 384, 'x': -0.5, 'y': -0.5}
                    ]
                }
            ]
        }

    def test_get_thresholds_returns_expected_keys(self):
        """Test that get_thresholds returns all expected threshold keys."""
        prompt_json = self.create_mock_prompt_json()
        prompts = ["Sample prompt one.", "Sample prompt two."]
        
        def mock_embedding_fn(text):
            return [0.5] * 384
        
        result = get_thresholds(prompts, prompt_json, mock_embedding_fn)
        
        assert 'add_lower_threshold' in result
        assert 'add_higher_threshold' in result
        assert 'remove_lower_threshold' in result
        assert 'remove_higher_threshold' in result


class TestGetValues:
    """Tests for the get_values function."""

    def create_mock_embeddings(self):
        """Create mock positive and negative embeddings."""
        positive_embeddings = {
            'accountability': [0.1] * 384,
            'transparency': [0.2] * 384
        }
        negative_embeddings = {
            'harmful': [0.8] * 384,
            'biased': [0.9] * 384
        }
        return positive_embeddings, negative_embeddings

    def test_get_values_returns_prompts_key(self):
        """Test that get_values returns dictionary with prompts key."""
        positive_embeddings, negative_embeddings = self.create_mock_embeddings()
        
        def mock_embedding_fn(texts):
            if isinstance(texts, list):
                return [[0.5] * 384 for _ in texts]
            return [0.5] * 384
        
        result = get_values(
            "Test prompt sentence.",
            positive_embeddings,
            negative_embeddings,
            mock_embedding_fn
        )
        
        assert 'prompts' in result
        assert len(result['prompts']) == 1
        
        result = get_values(
            "First sentence. Second sentence. Third sentence.",
            positive_embeddings,
            negative_embeddings,
            mock_embedding_fn
        )
        
        assert len(result['prompts']) == 3

    def test_get_values_each_prompt_has_required_fields(self):
        """Test that each prompt has required fields."""
        positive_embeddings, negative_embeddings = self.create_mock_embeddings()
        
        def mock_embedding_fn(texts):
            if isinstance(texts, list):
                return [[0.5] * 384 for _ in texts]
            return [0.5] * 384
        
        result = get_values(
            "Test sentence.",
            positive_embeddings,
            negative_embeddings,
            mock_embedding_fn
        )
        
        for prompt in result['prompts']:
            assert 'sentence' in prompt
            assert 'positive_value' in prompt
            assert 'negative_value' in prompt
            assert 'label' in prompt['positive_value']
            assert 'similarity' in prompt['positive_value']
            assert 'label' in prompt['negative_value']
            assert 'similarity' in prompt['negative_value']

    def test_get_values_empty_prompt_returns_empty_prompts(self):
        """Test that empty prompt returns empty prompts list."""
        positive_embeddings, negative_embeddings = self.create_mock_embeddings()
        
        def mock_embedding_fn(texts):
            return [[0.5] * 384]
        
        result = get_values(
            "",
            positive_embeddings,
            negative_embeddings,
            mock_embedding_fn
        )
        
        assert result['prompts'] == []

    def test_get_values_uses_default_embedding_fn(self):
        """Test that default embedding function is used when none provided."""
        positive_embeddings, negative_embeddings = self.create_mock_embeddings()
        
        with patch('control.recommendation_handler.get_embedding_func') as mock_get_fn:
            mock_fn = MagicMock(return_value=[[0.5] * 384])
            mock_get_fn.return_value = mock_fn
            
            result = get_values(
                "Test.",
                positive_embeddings,
                negative_embeddings,
                embedding_fn=None
            )
            
            mock_get_fn.assert_called_once_with(
                'local',
                model_id='sentence-transformers/all-MiniLM-L6-v2'
            )

