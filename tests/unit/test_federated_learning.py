#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Unit tests for the Federated Learning module.
Tests the differential privacy, secure aggregation, and model compression components.
"""

import os
import sys
import unittest
import time
import numpy as np
from unittest.mock import MagicMock, patch

# Add parent directory to path to import QTrust modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from qtrust.federated.federated_learning import (
    ModelType, AggregationMethod, CompressionMethod, DPMethod, SecAggMethod,
    FederatedModel, LinearModel, NeuralNetworkModel, ModelCompressor,
    DifferentialPrivacy, SecureAggregation, FederatedClient, FederatedServer,
    FederatedLearning
)


class TestLinearModel(unittest.TestCase):
    """Test cases for the LinearModel class."""

    def setUp(self):
        """Set up test environment before each test."""
        self.model_config = {
            'input_dim': 5,
            'output_dim': 1,
            'learning_rate': 0.01,
            'regularization': 0.001
        }
        self.model = LinearModel(self.model_config)

    def test_initialization(self):
        """Test model initialization."""
        self.assertEqual(self.model.model_type, ModelType.LINEAR)
        self.assertEqual(self.model.model_config, self.model_config)
        self.assertEqual(self.model.input_dim, 5)
        self.assertEqual(self.model.output_dim, 1)
        self.assertEqual(self.model.learning_rate, 0.01)
        self.assertEqual(self.model.regularization, 0.001)
        self.assertEqual(self.model.weights.shape, (5, 1))
        self.assertEqual(self.model.bias.shape, (1, 1))

    def test_train(self):
        """Test model training."""
        # Create random data
        X = np.random.randn(10, 5)
        y = np.random.randn(10, 1)
        
        # Train model
        metrics = self.model.train((X, y), epochs=2)
        
        # Check metrics
        self.assertIn('loss', metrics)
        self.assertIn('training_time', metrics)
        self.assertGreater(metrics['training_time'], 0)
        
        # Check model updates
        self.assertEqual(self.model.version, 1)
        self.assertGreater(self.model.last_update, 0)

    def test_predict(self):
        """Test model prediction."""
        # Create random data
        X = np.random.randn(10, 5)
        
        # Make predictions
        y_pred = self.model.predict(X)
        
        # Check predictions
        self.assertEqual(y_pred.shape, (10, 1))

    def test_get_set_weights(self):
        """Test getting and setting weights."""
        # Get weights
        weights = self.model.get_weights()
        
        # Check weights
        self.assertIn('weights', weights)
        self.assertIn('bias', weights)
        self.assertEqual(weights['weights'].shape, (5, 1))
        self.assertEqual(weights['bias'].shape, (1, 1))
        
        # Modify weights
        new_weights = {
            'weights': np.ones((5, 1)),
            'bias': np.zeros((1, 1))
        }
        
        # Set weights
        self.model.set_weights(new_weights)
        
        # Check updated weights
        weights = self.model.get_weights()
        np.testing.assert_array_equal(weights['weights'], np.ones((5, 1)))
        np.testing.assert_array_equal(weights['bias'], np.zeros((1, 1)))
        
        # Check version update
        self.assertEqual(self.model.version, 1)

    def test_evaluate(self):
        """Test model evaluation."""
        # Create random data
        X = np.random.randn(10, 5)
        y = np.random.randn(10, 1)
        
        # Evaluate model
        metrics = self.model.evaluate((X, y))
        
        # Check metrics
        self.assertIn('mse', metrics)
        self.assertIn('rmse', metrics)
        self.assertIn('mae', metrics)
        self.assertIn('inference_time', metrics)
        self.assertGreater(metrics['inference_time'], 0)

    def test_save_load(self):
        """Test saving and loading model."""
        # Create temporary file
        temp_file = 'temp_model.json'
        
        # Save model
        self.model.save(temp_file)
        
        # Create new model
        new_model = LinearModel()
        
        # Load model
        new_model.load(temp_file)
        
        # Check model parameters
        self.assertEqual(new_model.input_dim, 5)
        self.assertEqual(new_model.output_dim, 1)
        
        # Check weights
        np.testing.assert_array_equal(new_model.weights, self.model.weights)
        np.testing.assert_array_equal(new_model.bias, self.model.bias)
        
        # Clean up
        os.remove(temp_file)


class TestNeuralNetworkModel(unittest.TestCase):
    """Test cases for the NeuralNetworkModel class."""

    def setUp(self):
        """Set up test environment before each test."""
        self.model_config = {
            'input_dim': 5,
            'hidden_dims': [10, 5],
            'output_dim': 1,
            'activation': 'relu',
            'learning_rate': 0.01,
            'regularization': 0.001,
            'task': 'regression'
        }
        self.model = NeuralNetworkModel(self.model_config)

    def test_initialization(self):
        """Test model initialization."""
        self.assertEqual(self.model.model_type, ModelType.NEURAL_NETWORK)
        self.assertEqual(self.model.model_config, self.model_config)
        self.assertEqual(self.model.input_dim, 5)
        self.assertEqual(self.model.hidden_dims, [10, 5])
        self.assertEqual(self.model.output_dim, 1)
        self.assertEqual(self.model.activation, 'relu')
        self.assertEqual(self.model.learning_rate, 0.01)
        self.assertEqual(self.model.regularization, 0.001)
        
        # Check weights and biases
        self.assertEqual(len(self.model.weights), 3)
        self.assertEqual(len(self.model.biases), 3)
        self.assertEqual(self.model.weights[0].shape, (5, 10))
        self.assertEqual(self.model.weights[1].shape, (10, 5))
        self.assertEqual(self.model.weights[2].shape, (5, 1))
        self.assertEqual(self.model.biases[0].shape, (1, 10))
        self.assertEqual(self.model.biases[1].shape, (1, 5))
        self.assertEqual(self.model.biases[2].shape, (1, 1))

    def test_activation_functions(self):
        """Test activation functions."""
        # Test ReLU
        x = np.array([-1.0, 0.0, 1.0])
        relu = self.model._activate(x, 'relu')
        np.testing.assert_array_equal(relu, np.array([0.0, 0.0, 1.0]))
        
        # Test sigmoid
        sigmoid = self.model._activate(x, 'sigmoid')
        np.testing.assert_almost_equal(
            sigmoid, np.array([0.26894142, 0.5, 0.73105858]), decimal=6)
        
        # Test tanh
        tanh = self.model._activate(x, 'tanh')
        np.testing.assert_almost_equal(
            tanh, np.array([-0.76159416, 0.0, 0.76159416]), decimal=6)

    def test_forward_backward(self):
        """Test forward and backward passes."""
        # Create random data
        X = np.random.randn(10, 5)
        y = np.random.randn(10, 1)
        
        # Forward pass
        activations, pre_activations = self.model._forward(X)
        
        # Check activations
        self.assertEqual(len(activations), 4)
        self.assertEqual(activations[0].shape, (10, 5))  # Input
        self.assertEqual(activations[1].shape, (10, 10))  # First hidden layer
        self.assertEqual(activations[2].shape, (10, 5))  # Second hidden layer
        self.assertEqual(activations[3].shape, (10, 1))  # Output
        
        # Check pre-activations
        self.assertEqual(len(pre_activations), 3)
        self.assertEqual(pre_activations[0].shape, (10, 10))
        self.assertEqual(pre_activations[1].shape, (10, 5))
        self.assertEqual(pre_activations[2].shape, (10, 1))
        
        # Backward pass
        dw, db = self.model._backward(X, y, activations, pre_activations)
        
        # Check gradients
        self.assertEqual(len(dw), 3)
        self.assertEqual(len(db), 3)
        self.assertEqual(dw[0].shape, (5, 10))
        self.assertEqual(dw[1].shape, (10, 5))
        self.assertEqual(dw[2].shape, (5, 1))
        self.assertEqual(db[0].shape, (1, 10))
        self.assertEqual(db[1].shape, (1, 5))
        self.assertEqual(db[2].shape, (1, 1))

    def test_train(self):
        """Test model training."""
        # Create random data
        X = np.random.randn(10, 5)
        y = np.random.randn(10, 1)
        
        # Train model
        metrics = self.model.train((X, y), epochs=2)
        
        # Check metrics
        self.assertIn('loss', metrics)
        self.assertIn('training_time', metrics)
        self.assertGreater(metrics['training_time'], 0)
        
        # Check model updates
        self.assertEqual(self.model.version, 1)
        self.assertGreater(self.model.last_update, 0)

    def test_predict(self):
        """Test model prediction."""
        # Create random data
        X = np.random.randn(10, 5)
        
        # Make predictions
        y_pred = self.model.predict(X)
        
        # Check predictions
        self.assertEqual(y_pred.shape, (10, 1))

    def test_get_set_weights(self):
        """Test getting and setting weights."""
        # Get weights
        weights = self.model.get_weights()
        
        # Check weights
        self.assertIn('weights', weights)
        self.assertIn('biases', weights)
        self.assertEqual(len(weights['weights']), 3)
        self.assertEqual(len(weights['biases']), 3)
        
        # Create new weights
        new_weights = {
            'weights': [
                np.ones((5, 10)),
                np.ones((10, 5)),
                np.ones((5, 1))
            ],
            'biases': [
                np.zeros((1, 10)),
                np.zeros((1, 5)),
                np.zeros((1, 1))
            ]
        }
        
        # Set weights
        self.model.set_weights(new_weights)
        
        # Check updated weights
        weights = self.model.get_weights()
        for i in range(3):
            np.testing.assert_array_equal(weights['weights'][i], np.ones(new_weights['weights'][i].shape))
            np.testing.assert_array_equal(weights['biases'][i], np.zeros(new_weights['biases'][i].shape))
        
        # Check version update
        self.assertEqual(self.model.version, 1)

    def test_evaluate_regression(self):
        """Test model evaluation for regression."""
        # Create random data
        X = np.random.randn(10, 5)
        y = np.random.randn(10, 1)
        
        # Evaluate model
        metrics = self.model.evaluate((X, y))
        
        # Check metrics
        self.assertIn('mse', metrics)
        self.assertIn('rmse', metrics)
        self.assertIn('mae', metrics)
        self.assertIn('inference_time', metrics)
        self.assertGreater(metrics['inference_time'], 0)

    def test_evaluate_classification(self):
        """Test model evaluation for classification."""
        # Create classification model
        model_config = self.model_config.copy()
        model_config['task'] = 'classification'
        model = NeuralNetworkModel(model_config)
        
        # Create random data
        X = np.random.randn(10, 5)
        y = np.random.randint(0, 2, (10, 1))
        
        # Evaluate model
        metrics = model.evaluate((X, y))
        
        # Check metrics
        self.assertIn('accuracy', metrics)
        self.assertIn('precision', metrics)
        self.assertIn('recall', metrics)
        self.assertIn('f1', metrics)
        self.assertIn('log_loss', metrics)
        self.assertIn('inference_time', metrics)
        self.assertGreater(metrics['inference_time'], 0)


class TestModelCompressor(unittest.TestCase):
    """Test cases for the ModelCompressor class."""

    def setUp(self):
        """Set up test environment before each test."""
        # Create test weights
        self.weights = {
            'weights': np.random.randn(5, 10),
            'bias': np.random.randn(1, 10)
        }
        
        # Create test weights with list
        self.weights_list = {
            'weights': [
                np.random.randn(5, 10),
                np.random.randn(10, 5),
                np.random.randn(5, 1)
            ],
            'biases': [
                np.random.randn(1, 10),
                np.random.randn(1, 5),
                np.random.randn(1, 1)
            ]
        }

    def test_no_compression(self):
        """Test no compression."""
        compressor = ModelCompressor(CompressionMethod.NONE)
        
        # Compress weights
        compressed, metadata = compressor.compress(self.weights)
        
        # Check compressed weights
        self.assertIs(compressed, self.weights)
        self.assertEqual(metadata, {})
        
        # Decompress weights
        decompressed = compressor.decompress(compressed, metadata)
        
        # Check decompressed weights
        self.assertIs(decompressed, compressed)

    def test_quantization(self):
        """Test quantization compression."""
        compressor = ModelCompressor(CompressionMethod.QUANTIZATION, {'precision': 16})
        
        # Compress weights
        compressed, metadata = compressor.compress(self.weights)
        
        # Check compressed weights
        self.assertIn('weights', compressed)
        self.assertIn('bias', compressed)
        self.assertEqual(compressed['weights'].dtype, np.uint16)
        self.assertEqual(compressed['bias'].dtype, np.uint16)
        
        # Check metadata
        self.assertIn('scales', metadata)
        self.assertIn('zero_points', metadata)
        self.assertIn('weights', metadata['scales'])
        self.assertIn('bias', metadata['scales'])
        self.assertIn('weights', metadata['zero_points'])
        self.assertIn('bias', metadata['zero_points'])
        
        # Decompress weights
        decompressed = compressor.decompress(compressed, metadata)
        
        # Check decompressed weights
        self.assertIn('weights', decompressed)
        self.assertIn('bias', decompressed)
        self.assertEqual(decompressed['weights'].shape, self.weights['weights'].shape)
        self.assertEqual(decompressed['bias'].shape, self.weights['bias'].shape)
        
        # Test with list of arrays
        compressed, metadata = compressor.compress(self.weights_list)
        
        # Check compressed weights
        self.assertIn('weights', compressed)
        self.assertIn('biases', compressed)
        self.assertEqual(len(compressed['weights']), 3)
        self.assertEqual(len(compressed['biases']), 3)
        
        # Decompress weights
        decompressed = compressor.decompress(compressed, metadata)
        
        # Check decompressed weights
        self.assertIn('weights', decompressed)
        self.assertIn('biases', decompressed)
        self.assertEqual(len(decompressed['weights']), 3)
        self.assertEqual(len(decompressed['biases']), 3)
        for i in range(3):
            self.assertEqual(decompressed['weights'][i].shape, self.weights_list['weights'][i].shape)
            self.assertEqual(decompressed['biases'][i].shape, self.weights_list['biases'][i].shape)

    def test_pruning(self):
        """Test pruning compression."""
        compressor = ModelCompressor(CompressionMethod.PRUNING, {'sparsity': 0.5})
        
        # Compress weights
        compressed, metadata = compressor.compress(self.weights)
        
        # Check compressed weights
        self.assertIn('weights', compressed)
        self.assertIn('bias', compressed)
        
        # Check metadata
        self.assertIn('masks', metadata)
        self.assertIn('weights', metadata['masks'])
        self.assertIn('bias', metadata['masks'])
        
        # Check sparsity
        sparsity_weights = np.mean(compressed['weights'] == 0)
        self.assertGreaterEqual(sparsity_weights, 0.4)  # Allow some tolerance
        
        # Decompress weights
        decompressed = compressor.decompress(compressed, metadata)
        
        # Check decompressed weights
        self.assertIn('weights', decompressed)
        self.assertIn('bias', decompressed)
        self.assertEqual(decompressed['weights'].shape, self.weights['weights'].shape)
        self.assertEqual(decompressed['bias'].shape, self.weights['bias'].shape)
        
        # Test with list of arrays
        compressed, metadata = compressor.compress(self.weights_list)
        
        # Check compressed weights
        self.assertIn('weights', compressed)
        self.assertIn('biases', compressed)
        self.assertEqual(len(compressed['weights']), 3)
        self.assertEqual(len(compressed['biases']), 3)
        
        # Decompress weights
        decompressed = compressor.decompress(compressed, metadata)
        
        # Check decompressed weights
        self.assertIn('weights', decompressed)
        self.assertIn('biases', decompressed)
        self.assertEqual(len(decompressed['weights']), 3)
        self.assertEqual(len(decompressed['biases']), 3)
        for i in range(3):
            self.assertEqual(decompressed['weights'][i].shape, self.weights_list['weights'][i].shape)
            self.assertEqual(decompressed['biases'][i].shape, self.weights_list['biases'][i].shape)

    def test_sparsification(self):
        """Test sparsification compression."""
        compressor = ModelCompressor(CompressionMethod.SPARSIFICATION, {'sparsity': 0.5})
        
        # Compress weights
        compressed, metadata = compressor.compress(self.weights)
        
        # Check compressed weights
        self.assertIn('weights', compressed)
        self.assertIn('bias', compressed)
        
        # Check metadata
        self.assertIn('indices', metadata)
        self.assertIn('shapes', metadata)
        self.assertIn('weights', metadata['indices'])
        self.assertIn('bias', metadata['indices'])
        self.assertIn('weights', metadata['shapes'])
        self.assertIn('bias', metadata['shapes'])
        
        # Check compression ratio
        compression_ratio_weights = compressed['weights'].size / self.weights['weights'].size
        self.assertLessEqual(compression_ratio_weights, 0.6)  # Allow some tolerance
        
        # Decompress weights
        decompressed = compressor.decompress(compressed, metadata)
        
        # Check decompressed weights
        self.assertIn('weights', decompressed)
        self.assertIn('bias', decompressed)
        self.assertEqual(decompressed['weights'].shape, self.weights['weights'].shape)
        self.assertEqual(decompressed['bias'].shape, self.weights['bias'].shape)
        
        # Test with list of arrays
        compressed, metadata = compressor.compress(self.weights_list)
        
        # Check compressed weights
        self.assertIn('weights', compressed)
        self.assertIn('biases', compressed)
        self.assertEqual(len(compressed['weights']), 3)
        self.assertEqual(len(compressed['biases']), 3)
        
        # Decompress weights
        decompressed = compressor.decompress(compressed, metadata)
        
        # Check decompressed weights
        self.assertIn('weights', decompressed)
        self.assertIn('biases', decompressed)
        self.assertEqual(len(decompressed['weights']), 3)
        self.assertEqual(len(decompressed['biases']), 3)
        for i in range(3):
            self.assertEqual(decompressed['weights'][i].shape, self.weights_list['weights'][i].shape)
            self.assertEqual(decompressed['biases'][i].shape, self.weights_list['biases'][i].shape)

    def test_sketching(self):
        """Test sketching compression."""
        compressor = ModelCompressor(CompressionMethod.SKETCHING, {'sketch_size': 10})
        
        # Compress weights
        compressed, metadata = compressor.compress(self.weights)
        
        # Check compressed weights
        self.assertIn('weights', compressed)
        self.assertIn('bias', compressed)
        
        # Check metadata
        self.assertIn('shapes', metadata)
        self.assertIn('weights', metadata['shapes'])
        self.assertIn('bias', metadata['shapes'])
        
        # Check compression
        self.assertEqual(compressed['weights'].size, 10)
        
        # Decompress weights
        decompressed = compressor.decompress(compressed, metadata)
        
        # Check decompressed weights
        self.assertIn('weights', decompressed)
        self.assertIn('bias', decompressed)
        self.assertEqual(decompressed['weights'].shape, self.weights['weights'].shape)
        self.assertEqual(decompressed['bias'].shape, self.weights['bias'].shape)
        
        # Test with list of arrays
        compressed, metadata = compressor.compress(self.weights_list)
        
        # Check compressed weights
        self.assertIn('weights', compressed)
        self.assertIn('biases', compressed)
        self.assertEqual(len(compressed['weights']), 3)
        self.assertEqual(len(compressed['biases']), 3)
        
        # Decompress weights
        decompressed = compressor.decompress(compressed, metadata)
        
        # Check decompressed weights
        self.assertIn('weights', decompressed)
        self.assertIn('biases', decompressed)
        self.assertEqual(len(decompressed['weights']), 3)
        self.assertEqual(len(decompressed['biases']), 3)
        for i in range(3):
            self.assertEqual(decompressed['weights'][i].shape, self.weights_list['weights'][i].shape)
            self.assertEqual(decompressed['biases'][i].shape, self.weights_list['biases'][i].shape)


class TestDifferentialPrivacy(unittest.TestCase):
    """Test cases for the DifferentialPrivacy class."""

    def setUp(self):
        """Set up test environment before each test."""
        # Create test weights
        self.weights = {
            'weights': np.random.randn(5, 10),
            'bias': np.random.randn(1, 10)
        }
        
        # Create test weights with list
        self.weights_list = {
            'weights': [
                np.random.randn(5, 10),
                np.random.randn(10, 5),
                np.random.randn(5, 1)
            ],
            'biases': [
                np.random.randn(1, 10),
                np.random.randn(1, 5),
                np.random.randn(1, 1)
            ]
        }

    def test_no_privacy(self):
        """Test no privacy."""
        dp = DifferentialPrivacy(DPMethod.NONE)
        
        # Privatize weights
        privatized = dp.privatize(self.weights)
        
        # Check privatized weights
        self.assertIs(privatized, self.weights)

    def test_gaussian_noise(self):
        """Test Gaussian noise privacy."""
        dp = DifferentialPrivacy(DPMethod.GAUSSIAN, {
            'epsilon': 1.0,
            'delta': 1e-5,
            'sensitivity': 1.0,
            'clip_norm': 1.0
        })
        
        # Privatize weights
        privatized = dp.privatize(self.weights)
        
        # Check privatized weights
        self.assertIn('weights', privatized)
        self.assertIn('bias', privatized)
        self.assertEqual(privatized['weights'].shape, self.weights['weights'].shape)
        self.assertEqual(privatized['bias'].shape, self.weights['bias'].shape)
        
        # Check that noise was added
        self.assertFalse(np.array_equal(privatized['weights'], self.weights['weights']))
        self.assertFalse(np.array_equal(privatized['bias'], self.weights['bias']))
        
        # Test with list of arrays
        privatized = dp.privatize(self.weights_list)
        
        # Check privatized weights
        self.assertIn('weights', privatized)
        self.assertIn('biases', privatized)
        self.assertEqual(len(privatized['weights']), 3)
        self.assertEqual(len(privatized['biases']), 3)
        for i in range(3):
            self.assertEqual(privatized['weights'][i].shape, self.weights_list['weights'][i].shape)
            self.assertEqual(privatized['biases'][i].shape, self.weights_list['biases'][i].shape)
            self.assertFalse(np.array_equal(privatized['weights'][i], self.weights_list['weights'][i]))
            self.assertFalse(np.array_equal(privatized['biases'][i], self.weights_list['biases'][i]))

    def test_laplacian_noise(self):
        """Test Laplacian noise privacy."""
        dp = DifferentialPrivacy(DPMethod.LAPLACIAN, {
            'epsilon': 1.0,
            'sensitivity': 1.0,
            'clip_norm': 1.0
        })
        
        # Privatize weights
        privatized = dp.privatize(self.weights)
        
        # Check privatized weights
        self.assertIn('weights', privatized)
        self.assertIn('bias', privatized)
        self.assertEqual(privatized['weights'].shape, self.weights['weights'].shape)
        self.assertEqual(privatized['bias'].shape, self.weights['bias'].shape)
        
        # Check that noise was added
        self.assertFalse(np.array_equal(privatized['weights'], self.weights['weights']))
        self.assertFalse(np.array_equal(privatized['bias'], self.weights['bias']))
        
        # Test with list of arrays
        privatized = dp.privatize(self.weights_list)
        
        # Check privatized weights
        self.assertIn('weights', privatized)
        self.assertIn('biases', privatized)
        self.assertEqual(len(privatized['weights']), 3)
        self.assertEqual(len(privatized['biases']), 3)
        for i in range(3):
            self.assertEqual(privatized['weights'][i].shape, self.weights_list['weights'][i].shape)
            self.assertEqual(privatized['biases'][i].shape, self.weights_list['biases'][i].shape)
            self.assertFalse(np.array_equal(privatized['weights'][i], self.weights_list['weights'][i]))
            self.assertFalse(np.array_equal(privatized['biases'][i], self.weights_list['biases'][i]))

    def test_pate(self):
        """Test PATE privacy."""
        dp = DifferentialPrivacy(DPMethod.PATE, {
            'epsilon': 1.0,
            'delta': 1e-5,
            'sensitivity': 1.0,
            'clip_norm': 1.0
        })
        
        # Privatize weights
        privatized = dp.privatize(self.weights)
        
        # Check privatized weights
        self.assertIn('weights', privatized)
        self.assertIn('bias', privatized)
        self.assertEqual(privatized['weights'].shape, self.weights['weights'].shape)
        self.assertEqual(privatized['bias'].shape, self.weights['bias'].shape)
        
        # Check that noise was added
        self.assertFalse(np.array_equal(privatized['weights'], self.weights['weights']))
        self.assertFalse(np.array_equal(privatized['bias'], self.weights['bias']))


class TestSecureAggregation(unittest.TestCase):
    """Test cases for the SecureAggregation class."""

    def setUp(self):
        """Set up test environment before each test."""
        # Create test weights
        self.weights = {
            'weights': np.random.randn(5, 10),
            'bias': np.random.randn(1, 10)
        }
        
        # Create test weights with list
        self.weights_list = {
            'weights': [
                np.random.randn(5, 10),
                np.random.randn(10, 5),
                np.random.randn(5, 1)
            ],
            'biases': [
                np.random.randn(1, 10),
                np.random.randn(1, 5),
                np.random.randn(1, 1)
            ]
        }
        
        # Create client IDs
        self.client_ids = ['client1', 'client2', 'client3']

    def test_no_secure_aggregation(self):
        """Test no secure aggregation."""
        secagg = SecureAggregation(SecAggMethod.NONE)
        
        # Mask weights
        masked, metadata = secagg.mask(self.weights, 'client1', ['client2', 'client3'])
        
        # Check masked weights
        self.assertIs(masked, self.weights)
        self.assertEqual(metadata, {})
        
        # Unmask weights
        unmasked = secagg.unmask([self.weights, self.weights], [{}, {}])
        
        # Check unmasked weights
        self.assertIn('weights', unmasked)
        self.assertIn('bias', unmasked)
        self.assertEqual(unmasked['weights'].shape, self.weights['weights'].shape)
        self.assertEqual(unmasked['bias'].shape, self.weights['bias'].shape)

    def test_additive_masking(self):
        """Test additive secret sharing."""
        secagg = SecureAggregation(SecAggMethod.ADDITIVE)
        
        # Mask weights for each client
        masked1, metadata1 = secagg.mask(self.weights.copy(), 'client1', ['client2', 'client3'])
        masked2, metadata2 = secagg.mask(self.weights.copy(), 'client2', ['client1', 'client3'])
        masked3, metadata3 = secagg.mask(self.weights.copy(), 'client3', ['client1', 'client2'])
        
        # Check masked weights
        self.assertIn('weights', masked1)
        self.assertIn('bias', masked1)
        self.assertEqual(masked1['weights'].shape, self.weights['weights'].shape)
        self.assertEqual(masked1['bias'].shape, self.weights['bias'].shape)
        
        # Check metadata
        self.assertIn('client_id', metadata1)
        self.assertIn('other_clients', metadata1)
        self.assertEqual(metadata1['client_id'], 'client1')
        self.assertEqual(metadata1['other_clients'], ['client2', 'client3'])
        
        # Unmask weights
        unmasked = secagg.unmask([masked1, masked2, masked3], [metadata1, metadata2, metadata3])
        
        # Check unmasked weights
        self.assertIn('weights', unmasked)
        self.assertIn('bias', unmasked)
        self.assertEqual(unmasked['weights'].shape, self.weights['weights'].shape)
        self.assertEqual(unmasked['bias'].shape, self.weights['bias'].shape)
        
        # Test with list of arrays
        masked1, metadata1 = secagg.mask(self.weights_list.copy(), 'client1', ['client2', 'client3'])
        masked2, metadata2 = secagg.mask(self.weights_list.copy(), 'client2', ['client1', 'client3'])
        masked3, metadata3 = secagg.mask(self.weights_list.copy(), 'client3', ['client1', 'client2'])
        
        # Unmask weights
        unmasked = secagg.unmask([masked1, masked2, masked3], [metadata1, metadata2, metadata3])
        
        # Check unmasked weights
        self.assertIn('weights', unmasked)
        self.assertIn('biases', unmasked)
        self.assertEqual(len(unmasked['weights']), 3)
        self.assertEqual(len(unmasked['biases']), 3)
        for i in range(3):
            self.assertEqual(unmasked['weights'][i].shape, self.weights_list['weights'][i].shape)
            self.assertEqual(unmasked['biases'][i].shape, self.weights_list['biases'][i].shape)

    def test_threshold_masking(self):
        """Test threshold secret sharing."""
        secagg = SecureAggregation(SecAggMethod.THRESHOLD, {'threshold': 2})
        
        # Mask weights for each client
        masked1, metadata1 = secagg.mask(self.weights.copy(), 'client1', ['client2', 'client3'])
        masked2, metadata2 = secagg.mask(self.weights.copy(), 'client2', ['client1', 'client3'])
        masked3, metadata3 = secagg.mask(self.weights.copy(), 'client3', ['client1', 'client2'])
        
        # Check masked weights
        self.assertIn('weights', masked1)
        self.assertIn('bias', masked1)
        self.assertEqual(masked1['weights'].shape, self.weights['weights'].shape)
        self.assertEqual(masked1['bias'].shape, self.weights['bias'].shape)
        
        # Check metadata
        self.assertIn('shares', metadata1)
        self.assertIn('client_id', metadata1)
        self.assertIn('other_clients', metadata1)
        self.assertEqual(metadata1['client_id'], 'client1')
        self.assertEqual(metadata1['other_clients'], ['client2', 'client3'])
        
        # Unmask weights with just 2 clients (threshold is 2)
        unmasked = secagg.unmask([masked1, masked2], [metadata1, metadata2])
        
        # Check unmasked weights
        self.assertIn('weights', unmasked)
        self.assertIn('bias', unmasked)
        self.assertEqual(unmasked['weights'].shape, self.weights['weights'].shape)
        self.assertEqual(unmasked['bias'].shape, self.weights['bias'].shape)

    def test_homomorphic_masking(self):
        """Test homomorphic encryption."""
        secagg = SecureAggregation(SecAggMethod.HOMOMORPHIC)
        
        # Mask weights for each client
        masked1, metadata1 = secagg.mask(self.weights.copy(), 'client1', ['client2', 'client3'])
        masked2, metadata2 = secagg.mask(self.weights.copy(), 'client2', ['client1', 'client3'])
        masked3, metadata3 = secagg.mask(self.weights.copy(), 'client3', ['client1', 'client2'])
        
        # Check masked weights
        self.assertIn('weights', masked1)
        self.assertIn('bias', masked1)
        self.assertEqual(masked1['weights'].shape, self.weights['weights'].shape)
        self.assertEqual(masked1['bias'].shape, self.weights['bias'].shape)
        
        # Check metadata
        self.assertIn('masks', metadata1)
        
        # Unmask weights
        unmasked = secagg.unmask([masked1, masked2, masked3], [metadata1, metadata2, metadata3])
        
        # Check unmasked weights
        self.assertIn('weights', unmasked)
        self.assertIn('bias', unmasked)
        self.assertEqual(unmasked['weights'].shape, self.weights['weights'].shape)
        self.assertEqual(unmasked['bias'].shape, self.weights['bias'].shape)


class TestFederatedClient(unittest.TestCase):
    """Test cases for the FederatedClient class."""

    def setUp(self):
        """Set up test environment before each test."""
        # Create model
        self.model_config = {
            'input_dim': 5,
            'output_dim': 1,
            'learning_rate': 0.01
        }
        self.model = LinearModel(self.model_config)
        
        # Create client
        self.client_config = {
            'local_epochs': 2,
            'batch_size': 10,
            'learning_rate': 0.01,
            'dp_method': DPMethod.GAUSSIAN,
            'dp_config': {
                'epsilon': 1.0,
                'delta': 1e-5
            },
            'compression_method': CompressionMethod.QUANTIZATION,
            'compression_config': {
                'precision': 16
            },
            'secagg_method': SecAggMethod.ADDITIVE,
            'secagg_config': {}
        }
        self.client = FederatedClient('client1', self.model, self.client_config)
        
        # Create data
        self.X_train = np.random.randn(20, 5)
        self.y_train = np.random.randn(20, 1)
        self.X_val = np.random.randn(10, 5)
        self.y_val = np.random.randn(10, 1)
        self.X_test = np.random.randn(10, 5)
        self.y_test = np.random.randn(10, 1)
        
        # Set data
        self.client.set_data(
            (self.X_train, self.y_train),
            (self.X_val, self.y_val),
            (self.X_test, self.y_test)
        )

    def test_initialization(self):
        """Test client initialization."""
        self.assertEqual(self.client.client_id, 'client1')
        self.assertEqual(self.client.model, self.model)
        self.assertEqual(self.client.config, self.client_config)
        self.assertEqual(self.client.local_epochs, 2)
        self.assertEqual(self.client.batch_size, 10)
        self.assertEqual(self.client.learning_rate, 0.01)
        self.assertIsInstance(self.client.differential_privacy, DifferentialPrivacy)
        self.assertEqual(self.client.differential_privacy.method, DPMethod.GAUSSIAN)
        self.assertIsInstance(self.client.model_compressor, ModelCompressor)
        self.assertEqual(self.client.model_compressor.method, CompressionMethod.QUANTIZATION)
        self.assertIsInstance(self.client.secure_aggregation, SecureAggregation)
        self.assertEqual(self.client.secure_aggregation.method, SecAggMethod.ADDITIVE)

    def test_train(self):
        """Test client training."""
        # Train client
        metrics = self.client.train()
        
        # Check metrics
        self.assertIn('loss', metrics)
        self.assertIn('training_time', metrics)
        self.assertGreater(metrics['training_time'], 0)
        
        # Check client metrics
        self.assertIn('train_loss', self.client.metrics)
        self.assertIn('val_loss', self.client.metrics)
        self.assertGreater(len(self.client.metrics['train_loss']), 0)
        self.assertGreater(len(self.client.metrics['val_loss']), 0)
        self.assertGreater(self.client.metrics['total_training_time'], 0)

    def test_get_model_update(self):
        """Test getting model update."""
        # Get model update
        update, metadata = self.client.get_model_update(['client2', 'client3'])
        
        # Check update
        self.assertIn('weights', update)
        self.assertIn('bias', update)
        
        # Check metadata
        self.assertIn('client_id', metadata)
        self.assertIn('model_version', metadata)
        self.assertIn('compression', metadata)
        self.assertIn('secure_aggregation', metadata)
        self.assertEqual(metadata['client_id'], 'client1')
        
        # Check client metrics
        self.assertEqual(self.client.metrics['communication_rounds'], 1)
        self.assertGreater(self.client.metrics['total_communication_size'], 0)

    def test_apply_model_update(self):
        """Test applying model update."""
        # Create global weights
        global_weights = {
            'weights': np.ones((5, 1)),
            'bias': np.zeros((1, 1))
        }
        
        # Apply model update
        self.client.apply_model_update(global_weights)
        
        # Check model weights
        weights = self.client.model.get_weights()
        np.testing.assert_array_equal(weights['weights'], global_weights['weights'])
        np.testing.assert_array_equal(weights['bias'], global_weights['bias'])
        
        # Check client metrics
        self.assertIn('test_loss', self.client.metrics)
        self.assertGreater(len(self.client.metrics['test_loss']), 0)


class TestFederatedServer(unittest.TestCase):
    """Test cases for the FederatedServer class."""

    def setUp(self):
        """Set up test environment before each test."""
        # Create model
        self.model_config = {
            'input_dim': 5,
            'output_dim': 1,
            'learning_rate': 0.01
        }
        self.model = LinearModel(self.model_config)
        
        # Create server
        self.server_config = {
            'aggregation_method': AggregationMethod.FEDAVG,
            'min_clients': 2,
            'client_fraction': 1.0,
            'rounds': 3,
            'evaluate_every': 1,
            'secagg_method': SecAggMethod.ADDITIVE,
            'secagg_config': {},
            'compression_method': CompressionMethod.QUANTIZATION,
            'compression_config': {
                'precision': 16
            }
        }
        self.server = FederatedServer(self.model, self.server_config)
        
        # Create clients
        self.client_config = {
            'local_epochs': 2,
            'batch_size': 10,
            'learning_rate': 0.01,
            'dp_method': DPMethod.GAUSSIAN,
            'dp_config': {
                'epsilon': 1.0,
                'delta': 1e-5
            },
            'compression_method': CompressionMethod.QUANTIZATION,
            'compression_config': {
                'precision': 16
            },
            'secagg_method': SecAggMethod.ADDITIVE,
            'secagg_config': {}
        }
        
        # Create data
        np.random.seed(42)
        X = np.random.randn(100, 5)
        y = np.random.randn(100, 1)
        
        # Add clients
        for i in range(3):
            # Create client model
            client_model = LinearModel(self.model_config)
            client_model.set_weights(self.model.get_weights())
            
            # Create client
            client = FederatedClient(f'client{i}', client_model, self.client_config)
            
            # Set client data (different subset for each client)
            start_idx = i * 30
            end_idx = start_idx + 30
            client.set_data(
                (X[start_idx:end_idx], y[start_idx:end_idx]),
                (X[90:95], y[90:95]),
                (X[95:100], y[95:100])
            )
            
            # Add client to server
            self.server.add_client(client)

    def test_initialization(self):
        """Test server initialization."""
        self.assertEqual(self.server.model, self.model)
        self.assertEqual(self.server.config, self.server_config)
        self.assertEqual(self.server.aggregation_method, AggregationMethod.FEDAVG)
        self.assertEqual(self.server.min_clients, 2)
        self.assertEqual(self.server.client_fraction, 1.0)
        self.assertEqual(self.server.rounds, 3)
        self.assertEqual(self.server.evaluate_every, 1)
        self.assertEqual(len(self.server.clients), 3)
        self.assertIsInstance(self.server.secure_aggregation, SecureAggregation)
        self.assertEqual(self.server.secure_aggregation.method, SecAggMethod.ADDITIVE)
        self.assertIsInstance(self.server.model_compressor, ModelCompressor)
        self.assertEqual(self.server.model_compressor.method, CompressionMethod.QUANTIZATION)

    def test_add_remove_client(self):
        """Test adding and removing clients."""
        # Check initial clients
        self.assertEqual(len(self.server.clients), 3)
        self.assertIn('client0', self.server.clients)
        self.assertIn('client1', self.server.clients)
        self.assertIn('client2', self.server.clients)
        
        # Create new client
        client_model = LinearModel(self.model_config)
        client_model.set_weights(self.model.get_weights())
        client = FederatedClient('client3', client_model, self.client_config)
        
        # Add client
        self.server.add_client(client)
        
        # Check clients
        self.assertEqual(len(self.server.clients), 4)
        self.assertIn('client3', self.server.clients)
        
        # Remove client
        self.server.remove_client('client3')
        
        # Check clients
        self.assertEqual(len(self.server.clients), 3)
        self.assertNotIn('client3', self.server.clients)

    def test_select_clients(self):
        """Test client selection."""
        # Select clients
        selected_clients = self.server._select_clients()
        
        # Check selected clients
        self.assertEqual(len(selected_clients), 3)  # All clients selected (client_fraction = 1.0)
        self.assertIn('client0', selected_clients)
        self.assertIn('client1', selected_clients)
        self.assertIn('client2', selected_clients)
        
        # Change client fraction
        self.server.client_fraction = 0.5
        
        # Select clients
        selected_clients = self.server._select_clients()
        
        # Check selected clients
        self.assertGreaterEqual(len(selected_clients), self.server.min_clients)  # At least min_clients
        self.assertLessEqual(len(selected_clients), 2)  # At most 50% of clients

    def test_aggregate_model_updates(self):
        """Test model update aggregation."""
        # Create client weights
        weights1 = {
            'weights': np.ones((5, 1)),
            'bias': np.zeros((1, 1))
        }
        weights2 = {
            'weights': np.ones((5, 1)) * 2,
            'bias': np.ones((1, 1))
        }
        weights3 = {
            'weights': np.ones((5, 1)) * 3,
            'bias': np.ones((1, 1)) * 2
        }
        
        # Create metadata
        metadata1 = {'client_id': 'client1', 'compression': {}, 'secure_aggregation': {}}
        metadata2 = {'client_id': 'client2', 'compression': {}, 'secure_aggregation': {}}
        metadata3 = {'client_id': 'client3', 'compression': {}, 'secure_aggregation': {}}
        
        # Aggregate weights
        aggregated = self.server._aggregate_model_updates(
            [weights1, weights2, weights3],
            [metadata1, metadata2, metadata3],
            [10, 20, 30]
        )
        
        # Check aggregated weights
        self.assertIn('weights', aggregated)
        self.assertIn('bias', aggregated)
        self.assertEqual(aggregated['weights'].shape, (5, 1))
        self.assertEqual(aggregated['bias'].shape, (1, 1))
        
        # For FedAvg, should be average of all weights
        expected_weights = (weights1['weights'] + weights2['weights'] + weights3['weights']) / 3
        expected_bias = (weights1['bias'] + weights2['bias'] + weights3['bias']) / 3
        np.testing.assert_array_almost_equal(aggregated['weights'], expected_weights)
        np.testing.assert_array_almost_equal(aggregated['bias'], expected_bias)

    def test_train_round(self):
        """Test training round."""
        # Train one round
        metrics = self.server.train_round()
        
        # Check metrics
        self.assertIn('round', metrics)
        self.assertIn('active_clients', metrics)
        self.assertIn('training_time', metrics)
        self.assertEqual(metrics['round'], 1)
        self.assertEqual(metrics['active_clients'], 3)
        self.assertGreater(metrics['training_time'], 0)
        
        # Check server metrics
        self.assertEqual(self.server.metrics['rounds'], 1)
        self.assertEqual(self.server.metrics['active_clients_per_round'][0], 3)
        self.assertGreater(self.server.metrics['global_model_size'], 0)
        self.assertGreater(self.server.metrics['communication_size_per_round'][0], 0)
        self.assertGreater(self.server.metrics['training_time_per_round'][0], 0)

    def test_train(self):
        """Test training for multiple rounds."""
        # Train for 2 rounds
        metrics = self.server.train(2)
        
        # Check metrics
        self.assertEqual(self.server.metrics['rounds'], 2)
        self.assertEqual(len(self.server.metrics['active_clients_per_round']), 2)
        self.assertEqual(len(self.server.metrics['communication_size_per_round']), 2)
        self.assertEqual(len(self.server.metrics['training_time_per_round']), 2)
        
        # Check client metrics
        for client_id in self.server.clients:
            self.assertIn(client_id, self.server.metrics['client_metrics'])
            client_metrics = self.server.metrics['client_metrics'][client_id]
            self.assertIn('train_loss', client_metrics)
            self.assertIn('val_loss', client_metrics)
            self.assertIn('test_loss', client_metrics)
            self.assertGreater(len(client_metrics['train_loss']), 0)
            self.assertGreater(len(client_metrics['val_loss']), 0)
            self.assertGreater(len(client_metrics['test_loss']), 0)


class TestFederatedLearning(unittest.TestCase):
    """Test cases for the FederatedLearning class."""

    def setUp(self):
        """Set up test environment before each test."""
        # Create configuration
        self.config = {
            'model_type': ModelType.LINEAR,
            'model_config': {
                'input_dim': 5,
                'output_dim': 1,
                'learning_rate': 0.01
            },
            'server_config': {
                'aggregation_method': AggregationMethod.FEDAVG,
                'min_clients': 2,
                'client_fraction': 1.0,
                'rounds': 2,
                'evaluate_every': 1,
                'secagg_method': SecAggMethod.ADDITIVE,
                'secagg_config': {},
                'compression_method': CompressionMethod.QUANTIZATION,
                'compression_config': {
                    'precision': 16
                }
            },
            'client_config': {
                'local_epochs': 2,
                'batch_size': 10,
                'learning_rate': 0.01,
                'dp_method': DPMethod.GAUSSIAN,
                'dp_config': {
                    'epsilon': 1.0,
                    'delta': 1e-5
                },
                'compression_method': CompressionMethod.QUANTIZATION,
                'compression_config': {
                    'precision': 16
                },
                'secagg_method': SecAggMethod.ADDITIVE,
                'secagg_config': {}
            }
        }
        
        # Create federated learning system
        self.fl = FederatedLearning(self.config)
        
        # Create data
        np.random.seed(42)
        self.X = np.random.randn(100, 5)
        self.y = np.random.randn(100, 1)

    def test_initialization(self):
        """Test federated learning initialization."""
        self.assertEqual(self.fl.config, self.config)
        self.assertEqual(self.fl.model_type, ModelType.LINEAR)
        self.assertEqual(self.fl.model_config, self.config['model_config'])
        self.assertEqual(self.fl.server_config, self.config['server_config'])
        self.assertEqual(self.fl.client_config, self.config['client_config'])
        self.assertIsInstance(self.fl.model, LinearModel)
        self.assertIsInstance(self.fl.server, FederatedServer)
        self.assertEqual(len(self.fl.clients), 0)

    def test_add_remove_client(self):
        """Test adding and removing clients."""
        # Add clients
        for i in range(3):
            start_idx = i * 30
            end_idx = start_idx + 30
            self.fl.add_client(
                f'client{i}',
                (self.X[start_idx:end_idx], self.y[start_idx:end_idx]),
                (self.X[90:95], self.y[90:95]),
                (self.X[95:100], self.y[95:100])
            )
        
        # Check clients
        self.assertEqual(len(self.fl.clients), 3)
        self.assertIn('client0', self.fl.clients)
        self.assertIn('client1', self.fl.clients)
        self.assertIn('client2', self.fl.clients)
        
        # Check server clients
        self.assertEqual(len(self.fl.server.clients), 3)
        self.assertIn('client0', self.fl.server.clients)
        self.assertIn('client1', self.fl.server.clients)
        self.assertIn('client2', self.fl.server.clients)
        
        # Remove client
        self.fl.remove_client('client2')
        
        # Check clients
        self.assertEqual(len(self.fl.clients), 2)
        self.assertNotIn('client2', self.fl.clients)
        
        # Check server clients
        self.assertEqual(len(self.fl.server.clients), 2)
        self.assertNotIn('client2', self.fl.server.clients)

    def test_train(self):
        """Test federated learning training."""
        # Add clients
        for i in range(3):
            start_idx = i * 30
            end_idx = start_idx + 30
            self.fl.add_client(
                f'client{i}',
                (self.X[start_idx:end_idx], self.y[start_idx:end_idx]),
                (self.X[90:95], self.y[90:95]),
                (self.X[95:100], self.y[95:100])
            )
        
        # Train
        metrics = self.fl.train(2)
        
        # Check metrics
        self.assertIn('server', metrics)
        self.assertIn('clients', metrics)
        self.assertIn('model', metrics)
        self.assertEqual(metrics['server']['rounds'], 2)
        self.assertEqual(len(metrics['clients']), 3)
        for client_id in self.fl.clients:
            self.assertIn(client_id, metrics['clients'])
            self.assertIn('train_loss', metrics['clients'][client_id])
            self.assertIn('val_loss', metrics['clients'][client_id])
            self.assertIn('test_loss', metrics['clients'][client_id])

    def test_save_load_model(self):
        """Test saving and loading model."""
        # Create temporary file
        temp_file = 'temp_federated_model.json'
        
        # Save model
        self.fl.save_model(temp_file)
        
        # Create new federated learning system
        new_fl = FederatedLearning(self.config)
        
        # Load model
        new_fl.load_model(temp_file)
        
        # Check model weights
        original_weights = self.fl.model.get_weights()
        loaded_weights = new_fl.model.get_weights()
        
        for key in original_weights:
            np.testing.assert_array_equal(loaded_weights[key], original_weights[key])
        
        # Clean up
        os.remove(temp_file)


if __name__ == '__main__':
    unittest.main()
