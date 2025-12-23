#!/usr/bin/env python
# coding: utf-8

"""
Unit tests for utils.py
"""

import pytest
import math
import os
import numpy as np

# Import the module under test
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import get_distance

class TestGetDistance:
    """Tests for the get_distance function."""

    def test_same_embeddings_returns_zero(self):
        """Same embeddings should have zero distance."""
        embedding = np.array([1.0, 2.0, 3.0])
        assert get_distance(embedding, embedding) == 0.0

    def test_different_lengths_returns_infinity(self):
        """Embeddings of different lengths should return infinity."""
        embedding1 = np.array([1.0, 2.0, 3.0])
        embedding2 = np.array([1.0, 2.0])
        with pytest.raises(ValueError):
            get_distance(embedding1, embedding2)

    def test_euclidean_distance_calculation(self):
        """Test correct euclidean distance calculation."""
        embedding1 = np.array([0.0, 0.0, 0.0])
        embedding2 = np.array([3.0, 4.0, 0.0])
        # print(get_distance(embedding1, embedding2))
        # Distance should be 5 (3-4-5 triangle)
        assert get_distance(embedding1, embedding2) == 5.0

    def test_negative_values(self):
        """Test distance with negative values."""
        embedding1 = np.array([-1.0, -1.0])
        embedding2 = np.array([1.0, 1.0])
        expected = math.sqrt(8)  # sqrt((2)^2 + (2)^2)
        assert abs(get_distance(embedding1, embedding2) - expected) < 1e-10

    def test_empty_embeddings(self):
        """Empty embeddings should have zero distance."""
        embedding1 = np.array([])
        embedding2 = np.array([])
        assert get_distance(embedding1, embedding2) == 0.0