import unittest
import numpy as np
import sys
import os

# Add the parent directory to the path so we can import the qtrust package
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# Import the implementation switch module
from qtrust.implementation_switch import get_privacy_preserving_fl, set_use_pytorch

# Set to use mock implementation for testing
set_use_pytorch(False)

class TestMockPrivacyPreservingFL(unittest.TestCase):
    """Test cases for the mock Privacy-Preserving Federated Learning."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.num_clients = 5
        self.dp_params = {
            "epsilon": 0.5,
            "delta": 1e-6,
            "clip_norm": 0.8
        }
        # Pass dp_params as a dictionary
        self.fl = get_privacy_preserving_fl(num_clients=self.num_clients, dp_params=self.dp_params)
        
    def test_initialization(self):
        """Test initialization."""
        self.assertEqual(self.fl.num_clients, self.num_clients)
        self.assertEqual(self.fl.dp.epsilon, self.dp_params["epsilon"])
        self.assertEqual(self.fl.dp.delta, self.dp_params["delta"])
        self.assertEqual(self.fl.dp.clip_norm, self.dp_params["clip_norm"])
        
    def test_differential_privacy(self):
        """Test differential privacy mechanism."""
        # Create a test tensor
        test_tensor = np.random.rand(10, 5)
        
        # Apply noise
        noisy_tensor = self.fl.dp.add_noise(test_tensor)
        
        # Check that noise was added
        self.assertFalse(np.array_equal(test_tensor, noisy_tensor))
        
        # Check that shape is preserved
        self.assertEqual(test_tensor.shape, noisy_tensor.shape)
        
    def test_secure_aggregation(self):
        """Test secure aggregation."""
        # Create test models
        model_shape = {"layer1": (10, 5), "layer2": (5, 2)}
        models = []
        for _ in range(3):
            model = {}
            for key, shape in model_shape.items():
                model[key] = np.random.rand(*shape)
            models.append(model)
            
        # Perform secure aggregation
        aggregated = self.fl.secure_aggregate(models)
        
        # Check that result has the same structure
        self.assertEqual(set(aggregated.keys()), set(models[0].keys()))
        for key in aggregated:
            self.assertEqual(aggregated[key].shape, models[0][key].shape)
            
    def test_federated_learning_round(self):
        """Test one round of federated learning."""
        # Initialize model
        model_shape = {"layer1": (10, 5), "layer2": (5, 2)}
        self.fl.global_model = self.fl.initialize_model(model_shape)
        
        # Create mock client data
        client_ids = list(range(3))
        data = {}
        labels = {}
        for client_id in client_ids:
            data[client_id] = np.random.rand(20, 10)
            labels[client_id] = np.random.randint(0, 2, size=(20,))
            
        # Initialize client models
        for client_id in client_ids:
            self.fl.client_models[client_id] = self.fl.initialize_model(model_shape)
            
        # Perform one round of federated learning
        updated_model = self.fl.federated_learning_round(
            client_ids, data, labels, epochs=1, batch_size=4, secure=True
        )
        
        # Check that result has the same structure
        self.assertEqual(set(updated_model.keys()), set(model_shape.keys()))
        for key in updated_model:
            self.assertEqual(updated_model[key].shape, (model_shape[key]))
            
    def test_evaluation(self):
        """Test evaluation."""
        # Create test data
        data = np.random.rand(20, 10)
        labels = np.random.randint(0, 2, size=(20,))
        
        # Perform evaluation
        metrics = self.fl.evaluate(data, labels)
        
        # Check metrics
        self.assertIn("accuracy", metrics)
        self.assertIn("loss", metrics)
        self.assertIn("f1_score", metrics)

if __name__ == '__main__':
    unittest.main()
