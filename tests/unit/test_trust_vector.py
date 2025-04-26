import unittest
from qtrust.trust.trust_vector import TrustVector

class TestTrustVector(unittest.TestCase):
    def setUp(self):
        self.trust_vector = TrustVector()
        
    def test_initialization(self):
        """Test that trust vector initializes with default values."""
        self.assertEqual(self.trust_vector.get_dimension('transaction_validation'), 0.5)
        self.assertEqual(self.trust_vector.get_dimension('block_proposal'), 0.5)
        self.assertEqual(self.trust_vector.get_dimension('response_time'), 0.5)
        self.assertEqual(self.trust_vector.get_dimension('uptime'), 0.5)
        self.assertEqual(self.trust_vector.get_dimension('resource_contribution'), 0.5)
        
    def test_update_dimension(self):
        """Test updating a trust dimension."""
        self.trust_vector.update_dimension('transaction_validation', 0.8)
        self.assertEqual(self.trust_vector.get_dimension('transaction_validation'), 0.8)
        
    def test_update_multiple_dimensions(self):
        """Test updating multiple trust dimensions."""
        self.trust_vector.update_dimension('transaction_validation', 0.8)
        self.trust_vector.update_dimension('block_proposal', 0.7)
        self.trust_vector.update_dimension('response_time', 0.9)
        
        self.assertEqual(self.trust_vector.get_dimension('transaction_validation'), 0.8)
        self.assertEqual(self.trust_vector.get_dimension('block_proposal'), 0.7)
        self.assertEqual(self.trust_vector.get_dimension('response_time'), 0.9)
        
    def test_invalid_dimension(self):
        """Test handling of invalid dimension."""
        with self.assertRaises(ValueError):
            self.trust_vector.update_dimension('invalid_dimension', 0.8)
            
    def test_invalid_value(self):
        """Test handling of invalid trust value."""
        with self.assertRaises(ValueError):
            self.trust_vector.update_dimension('transaction_validation', 1.5)
        
        with self.assertRaises(ValueError):
            self.trust_vector.update_dimension('transaction_validation', -0.2)
            
    def test_get_aggregate_trust(self):
        """Test calculation of aggregate trust score."""
        self.trust_vector.update_dimension('transaction_validation', 0.8)
        self.trust_vector.update_dimension('block_proposal', 0.7)
        self.trust_vector.update_dimension('response_time', 0.9)
        self.trust_vector.update_dimension('uptime', 0.6)
        self.trust_vector.update_dimension('resource_contribution', 0.5)
        
        # Default weights are equal, so this should be the average
        expected_aggregate = (0.8 + 0.7 + 0.9 + 0.6 + 0.5) / 5
        self.assertAlmostEqual(self.trust_vector.get_aggregate_trust(), expected_aggregate)
        
    def test_custom_weights(self):
        """Test calculation of aggregate trust with custom weights."""
        self.trust_vector.update_dimension('transaction_validation', 0.8)
        self.trust_vector.update_dimension('block_proposal', 0.7)
        self.trust_vector.update_dimension('response_time', 0.9)
        self.trust_vector.update_dimension('uptime', 0.6)
        self.trust_vector.update_dimension('resource_contribution', 0.5)
        
        weights = {
            'transaction_validation': 0.3,
            'block_proposal': 0.3,
            'response_time': 0.2,
            'uptime': 0.1,
            'resource_contribution': 0.1
        }
        
        expected_aggregate = (0.8 * 0.3 + 0.7 * 0.3 + 0.9 * 0.2 + 0.6 * 0.1 + 0.5 * 0.1)
        self.assertAlmostEqual(self.trust_vector.get_aggregate_trust(weights), expected_aggregate)
        
    def test_decay(self):
        """Test trust decay over time."""
        self.trust_vector.update_dimension('transaction_validation', 0.8)
        self.trust_vector.decay(0.1)  # 10% decay
        self.assertAlmostEqual(self.trust_vector.get_dimension('transaction_validation'), 0.72)  # 0.8 * 0.9
        
    def test_boost(self):
        """Test trust boost."""
        self.trust_vector.update_dimension('transaction_validation', 0.5)
        self.trust_vector.boost('transaction_validation', 0.1)  # 10% boost
        self.assertAlmostEqual(self.trust_vector.get_dimension('transaction_validation'), 0.55)  # 0.5 + (1-0.5)*0.1
        
    def test_serialization(self):
        """Test serialization and deserialization."""
        self.trust_vector.update_dimension('transaction_validation', 0.8)
        self.trust_vector.update_dimension('block_proposal', 0.7)
        
        serialized = self.trust_vector.serialize()
        new_vector = TrustVector.deserialize(serialized)
        
        self.assertEqual(new_vector.get_dimension('transaction_validation'), 0.8)
        self.assertEqual(new_vector.get_dimension('block_proposal'), 0.7)

if __name__ == '__main__':
    unittest.main()
