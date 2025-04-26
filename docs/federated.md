# Federated Learning

This document details the Federated Learning system implementation in the QTrust framework.

## Overview

The QTrust Federated Learning system enables distributed model training across blockchain nodes while preserving privacy and security. It incorporates differential privacy, secure aggregation, and model compression techniques.

## Core Components

### Model Types

The system supports multiple model types:

1. **Linear Models**

   ```python
   class LinearModel:
       def __init__(self, input_dim, output_dim):
           self.weights = np.random.randn(input_dim, output_dim) * 0.01
           self.bias = np.zeros(output_dim)

       def predict(self, X):
           return np.dot(X, self.weights) + self.bias

       def train(self, X, y, learning_rate=0.01, epochs=10):
           for _ in range(epochs):
               predictions = self.predict(X)
               error = predictions - y

               # Gradient descent
               d_weights = np.dot(X.T, error) / len(X)
               d_bias = np.sum(error, axis=0) / len(X)

               self.weights -= learning_rate * d_weights
               self.bias -= learning_rate * d_bias
   ```

2. **Neural Network Models**
   ```python
   class NeuralNetworkModel:
       def __init__(self, layer_sizes, activation='relu'):
           self.layers = []
           self.activation = self._get_activation_function(activation)

           # Initialize layers
           for i in range(len(layer_sizes) - 1):
               self.layers.append({
                   'weights': np.random.randn(layer_sizes[i], layer_sizes[i+1]) * 0.01,
                   'bias': np.zeros(layer_sizes[i+1])
               })

       def forward(self, X):
           # Forward pass through the network
           activation = X
           activations = [X]

           for layer in self.layers:
               z = np.dot(activation, layer['weights']) + layer['bias']
               activation = self.activation(z)
               activations.append(activation)

           return activations
   ```

### Differential Privacy

The system implements three differential privacy mechanisms:

1. **Gaussian Mechanism**

   ```python
   def apply_gaussian_noise(self, model_update, sensitivity, epsilon, delta):
       """Apply Gaussian noise for differential privacy."""
       # Calculate sigma based on privacy parameters
       sigma = np.sqrt(2 * np.log(1.25/delta)) * sensitivity / epsilon

       # Apply noise to each parameter
       noisy_update = {}
       for param_name, param_value in model_update.items():
           noise = np.random.normal(0, sigma, param_value.shape)
           noisy_update[param_name] = param_value + noise

       return noisy_update
   ```

2. **Laplacian Mechanism**

   ```python
   def apply_laplacian_noise(self, model_update, sensitivity, epsilon):
       """Apply Laplacian noise for differential privacy."""
       # Calculate scale based on privacy parameters
       scale = sensitivity / epsilon

       # Apply noise to each parameter
       noisy_update = {}
       for param_name, param_value in model_update.items():
           noise = np.random.laplace(0, scale, param_value.shape)
           noisy_update[param_name] = param_value + noise

       return noisy_update
   ```

3. **PATE (Private Aggregation of Teacher Ensembles)**
   ```python
   def apply_pate(self, teacher_predictions, epsilon):
       """Apply PATE mechanism for differential privacy."""
       # Count votes for each class
       vote_counts = np.sum(teacher_predictions, axis=0)

       # Add noise to vote counts
       sensitivity = 1  # One teacher can change its vote
       noisy_counts = vote_counts + np.random.laplace(0, sensitivity/epsilon, vote_counts.shape)

       # Return the class with the highest noisy count
       return np.argmax(noisy_counts, axis=1)
   ```

### Secure Aggregation

The system implements three secure aggregation protocols:

1. **Additive Masking**

   ```python
   def aggregate_with_additive_masking(self, model_updates):
       """Aggregate model updates using additive masking."""
       n_clients = len(model_updates)

       # Generate random masks that sum to zero
       masks = []
       for i in range(n_clients - 1):
           mask = {param: np.random.randn(*value.shape) for param, value in model_updates[0].items()}
           masks.append(mask)

       # Last mask is negative sum of all other masks
       last_mask = {param: -np.sum([mask[param] for mask in masks], axis=0)
                   for param in model_updates[0].keys()}
       masks.append(last_mask)

       # Apply masks to model updates
       masked_updates = []
       for i, update in enumerate(model_updates):
           masked_update = {param: value + masks[i][param]
                           for param, value in update.items()}
           masked_updates.append(masked_update)

       # Aggregate masked updates
       aggregated = {param: np.mean([update[param] for update in masked_updates], axis=0)
                    for param in model_updates[0].keys()}

       return aggregated
   ```

2. **Threshold Masking**

   ```python
   def aggregate_with_threshold(self, model_updates, threshold):
       """Aggregate model updates using threshold-based masking."""
       # Only include updates from clients with trust score above threshold
       filtered_updates = [update for i, update in enumerate(model_updates)
                          if self.client_trust_scores[i] >= threshold]

       if not filtered_updates:
           return None

       # Compute weighted average based on trust scores
       weights = [self.client_trust_scores[i] for i, update in enumerate(model_updates)
                 if self.client_trust_scores[i] >= threshold]
       weights = np.array(weights) / np.sum(weights)

       aggregated = {}
       for param in filtered_updates[0].keys():
           param_updates = [update[param] for update in filtered_updates]
           aggregated[param] = np.sum([w * p for w, p in zip(weights, param_updates)], axis=0)

       return aggregated
   ```

3. **Homomorphic Encryption**
   ```python
   def aggregate_with_homomorphic_encryption(self, model_updates):
       """Aggregate model updates using homomorphic encryption."""
       # In a real implementation, this would use a homomorphic encryption library
       # For simplicity, we simulate the process

       # Generate key pair
       public_key, private_key = self._generate_he_keypair()

       # Encrypt model updates
       encrypted_updates = []
       for update in model_updates:
           encrypted_update = {param: self._encrypt(value, public_key)
                              for param, value in update.items()}
           encrypted_updates.append(encrypted_update)

       # Aggregate encrypted updates (addition is possible with homomorphic encryption)
       encrypted_aggregated = {}
       for param in encrypted_updates[0].keys():
           encrypted_aggregated[param] = self._he_sum([update[param] for update in encrypted_updates])

       # Decrypt the result
       aggregated = {param: self._decrypt(value, private_key)
                    for param, value in encrypted_aggregated.items()}

       # Scale by 1/n
       aggregated = {param: value / len(model_updates)
                    for param, value in aggregated.items()}

       return aggregated
   ```

### Model Compression

The system implements four model compression techniques:

1. **Quantization**

   ```python
   def quantize(self, model_update, bits=8):
       """Quantize model parameters to reduce communication cost."""
       quantized_update = {}
       for param_name, param_value in model_update.items():
           # Find min and max values
           min_val = np.min(param_value)
           max_val = np.max(param_value)

           # Calculate scale and zero point
           scale = (max_val - min_val) / (2**bits - 1)
           zero_point = -min_val / scale

           # Quantize values
           quantized_values = np.round(param_value / scale + zero_point).astype(np.int32)

           # Store quantized values and quantization parameters
           quantized_update[param_name] = {
               'values': quantized_values,
               'scale': scale,
               'zero_point': zero_point
           }

       return quantized_update

   def dequantize(self, quantized_update):
       """Dequantize model parameters."""
       dequantized_update = {}
       for param_name, param_data in quantized_update.items():
           # Extract quantization parameters
           quantized_values = param_data['values']
           scale = param_data['scale']
           zero_point = param_data['zero_point']

           # Dequantize values
           dequantized_values = scale * (quantized_values - zero_point)
           dequantized_update[param_name] = dequantized_values

       return dequantized_update
   ```

2. **Pruning**

   ```python
   def prune(self, model_update, sparsity=0.9):
       """Prune model parameters to achieve target sparsity."""
       pruned_update = {}
       for param_name, param_value in model_update.items():
           # Flatten the array
           flat_value = param_value.flatten()

           # Calculate threshold for target sparsity
           k = int((1 - sparsity) * flat_value.size)
           if k == 0:
               threshold = np.max(np.abs(flat_value)) + 1
           else:
               threshold = np.partition(np.abs(flat_value), -k)[-k]

           # Create mask for values above threshold
           mask = np.abs(param_value) >= threshold

           # Store sparse representation
           pruned_update[param_name] = {
               'values': param_value[mask],
               'mask': mask
           }

       return pruned_update

   def unprune(self, pruned_update):
       """Reconstruct model parameters from pruned representation."""
       unpruned_update = {}
       for param_name, param_data in pruned_update.items():
           # Extract pruned data
           values = param_data['values']
           mask = param_data['mask']

           # Create full array filled with zeros
           full_shape = mask.shape
           full_array = np.zeros(full_shape)

           # Fill in non-zero values
           full_array[mask] = values
           unpruned_update[param_name] = full_array

       return unpruned_update
   ```

3. **Sparsification**

   ```python
   def sparsify(self, model_update, keep_ratio=0.1):
       """Sparsify model updates by keeping only top values."""
       sparsified_update = {}
       for param_name, param_value in model_update.items():
           # Flatten the array
           flat_value = param_value.flatten()

           # Calculate number of values to keep
           k = int(keep_ratio * flat_value.size)
           if k == 0:
               k = 1

           # Find indices of top k values by magnitude
           top_indices = np.argsort(np.abs(flat_value))[-k:]

           # Create sparse representation
           values = flat_value[top_indices]
           indices = np.unravel_index(top_indices, param_value.shape)

           sparsified_update[param_name] = {
               'values': values,
               'indices': indices,
               'shape': param_value.shape
           }

       return sparsified_update

   def desparsify(self, sparsified_update):
       """Reconstruct model parameters from sparsified representation."""
       desparsified_update = {}
       for param_name, param_data in sparsified_update.items():
           # Extract sparsified data
           values = param_data['values']
           indices = param_data['indices']
           shape = param_data['shape']

           # Create full array filled with zeros
           full_array = np.zeros(shape)

           # Fill in non-zero values
           full_array[indices] = values
           desparsified_update[param_name] = full_array

       return desparsified_update
   ```

4. **Sketching**

   ```python
   def sketch(self, model_update, sketch_dim=100):
       """Compress model updates using count-min sketch."""
       sketched_update = {}
       for param_name, param_value in model_update.items():
           # Flatten the array
           flat_value = param_value.flatten()

           # Create hash functions
           num_hash_functions = 3
           hash_functions = [lambda x, seed=i: hash(str(x) + str(seed)) % sketch_dim
                            for i in range(num_hash_functions)]

           # Create sketch
           sketch = np.zeros((num_hash_functions, sketch_dim))

           # Fill sketch
           for i, val in enumerate(flat_value):
               for h in range(num_hash_functions):
                   j = hash_functions[h](i)
                   sketch[h, j] += val

           sketched_update[param_name] = {
               'sketch': sketch,
               'hash_functions': hash_functions,
               'shape': param_value.shape
           }

       return sketched_update

   def desketch(self, sketched_update):
       """Reconstruct model parameters from sketched representation."""
       # Note: This is a lossy reconstruction
       desketched_update = {}
       for param_name, param_data in sketched_update.items():
           # Extract sketched data
           sketch = param_data['sketch']
           hash_functions = param_data['hash_functions']
           shape = param_data['shape']

           # Create full array
           full_array = np.zeros(np.prod(shape))

           # Reconstruct values (using median to reduce noise)
           for i in range(len(full_array)):
               estimates = []
               for h in range(len(hash_functions)):
                   j = hash_functions[h](i)
                   estimates.append(sketch[h, j])
               full_array[i] = np.median(estimates)

           desketched_update[param_name] = full_array.reshape(shape)

       return desketched_update
   ```

## Client-Server Architecture

The system implements a flexible client-server architecture:

```python
class FederatedServer:
    def __init__(self, model, config):
        self.model = model
        self.config = config
        self.clients = {}
        self.client_trust_scores = {}
        self.round = 0
        self.privacy_mechanism = self._create_privacy_mechanism()
        self.aggregation_method = self._create_aggregation_method()
        self.compression_method = self._create_compression_method()

    def add_client(self, client_id, trust_score=0.5):
        """Add a client to the federated learning system."""
        self.clients[client_id] = None  # Will store client model updates
        self.client_trust_scores[client_id] = trust_score

    def remove_client(self, client_id):
        """Remove a client from the federated learning system."""
        if client_id in self.clients:
            del self.clients[client_id]
            del self.client_trust_scores[client_id]

    def select_clients(self, ratio=0.1, min_trust=0.0):
        """Select a subset of clients for training."""
        eligible_clients = [cid for cid, score in self.client_trust_scores.items()
                           if score >= min_trust]

        num_to_select = max(1, int(ratio * len(eligible_clients)))
        selected_clients = np.random.choice(eligible_clients,
                                           size=min(num_to_select, len(eligible_clients)),
                                           replace=False)

        return list(selected_clients)

    def train_round(self, selected_clients, data):
        """Conduct one round of federated training."""
        # Distribute model to clients
        for client_id in selected_clients:
            # In a real system, this would send the model to the client
            # Here we simulate client training
            client = FederatedClient(client_id, self.model.copy(), self.config)
            client_data = data.get(client_id, [])
            model_update = client.train(client_data)

            # Apply privacy mechanism
            if self.privacy_mechanism:
                model_update = self.privacy_mechanism.apply(model_update)

            # Apply compression
            if self.compression_method:
                model_update = self.compression_method.compress(model_update)

            self.clients[client_id] = model_update

        # Aggregate updates
        aggregated_update = self._aggregate_model_updates(selected_clients)

        # Apply update to global model
        self.model.apply_update(aggregated_update)

        self.round += 1
        return aggregated_update
```

## Integration with Other Components

The Federated Learning system integrates with:

1. **HTDCM**: Uses trust scores to weight client contributions
2. **Consensus**: Ensures agreement on model updates
3. **MAD-RAPID**: Optimizes communication for model distribution
4. **Monitoring**: Tracks model performance and client participation

## Configuration

The system supports flexible configuration:

```python
config = {
    "privacy": {
        "mechanism": "gaussian",  # gaussian, laplacian, pate, or none
        "epsilon": 0.1,
        "delta": 1e-5
    },
    "aggregation": {
        "method": "fedavg",  # fedavg, weighted, threshold, or secure
        "threshold": 0.7
    },
    "compression": {
        "method": "quantization",  # quantization, pruning, sparsification, sketching, or none
        "bits": 8,
        "sparsity": 0.9
    },
    "training": {
        "rounds": 100,
        "client_fraction": 0.1,
        "min_trust_score": 0.5,
        "local_epochs": 5,
        "nodes": 768,
        "shards": 64,
        "validators": 64
    }
}

federated_system = FederatedLearning(model, config)
```
