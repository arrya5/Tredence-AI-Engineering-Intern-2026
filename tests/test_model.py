import pytest
import torch
from src.model import PrunableLinear, SelfPruningNet

def test_prunable_linear_forward():
    """Verify strictly expected tensor shapes in the custom layer."""
    batch_size = 8
    in_features = 16
    out_features = 32
    
    layer = PrunableLinear(in_features, out_features)
    x = torch.randn(batch_size, in_features)
    
    out = layer(x)
    
    # Shape correctness
    assert out.shape == (batch_size, out_features)
    
    # State tracking
    assert hasattr(layer, 'weight')
    assert hasattr(layer, 'gate_scores')
    assert layer.weight.shape == layer.gate_scores.shape

def test_prunable_linear_gradients():
    """Confirm gradients reliably flow backwards to our custom parameter injection."""
    layer = PrunableLinear(10, 5)
    x = torch.randn(2, 10)
    out = layer(x)
    loss = out.sum()
    loss.backward()
    
    # Crucial property for self-pruning architecture
    assert layer.weight.grad is not None
    assert layer.gate_scores.grad is not None, "Gradients failed to flow back to gate_scores!"

def test_self_pruning_net_gates_collection():
    """Ensure network orchestration properly extracts nested parameters."""
    net = SelfPruningNet(100, 50, 25, 10)
    gates = net.get_all_gates()
    assert len(gates) == 3, "Network failed to collect all PrunableLinear layers"
