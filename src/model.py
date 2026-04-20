import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class PrunableLinear(nn.Module):
    """
    A custom linear layer that learns to prune its own weights using trainable gates.
    Mathematical Intent: We parameterize a set of unbounded scores and project them
    through a Sigmoid activation. This allows backpropagation to continuously push 
    less-impactful parameters towards negative infinity, resulting in a zero mask.
    """
    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        # Standard weights and bias parameters
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
            
        # The key innovation: A matching tensor of scores that control pruning
        self.gate_scores = nn.Parameter(torch.Tensor(out_features, in_features))
        
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Initialize parameters using sound mathematical defaults."""
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in = self.weight.size(1)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)
            
        # Initialize gate_scores to 0.5.
        # Sigmoid(0.5) is approx 0.62. This allows them to cross into negative 
        # space faster so they can hit the tight 1e-2 sparsity threshold in fewer epochs.
        nn.init.constant_(self.gate_scores, 0.5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass applying the dynamic gates to the weights.
        """
        # Step 1: Transform gate_scores to probability [0, 1] domain
        gates = torch.sigmoid(self.gate_scores)
        
        # [CRITICAL FIX]: Implement HARD Pruning during Evaluation.
        # Otherwise, the tiny gates (e.g. 0.009) still mathematically contribute 
        # to the output, creating an illusion of high accuracy despite severe pruning.
        if not self.training:
            # We use 1e-2 to match our sparsity tracking threshold strictly
            mask = (gates >= 1e-2).float()
            gates = gates * mask
            
        # Step 2: Element-wise multiplication to zero-out pruned weights
        pruned_weights = self.weight * gates
        
        # Step 3: Compute standard linear output
        return F.linear(x, pruned_weights, self.bias)

    def get_gates(self) -> torch.Tensor:
        """Returns the [0, 1] bounded gates for regularization calculation."""
        return torch.sigmoid(self.gate_scores)


class SelfPruningNet(nn.Module):
    """
    A standard Feed-Forward Neural Network composed entirely of PrunableLinear layers.
    Used for CIFAR-10 image classification.
    """
    def __init__(self, input_size: int = 3072, hidden1: int = 512, hidden2: int = 256, num_classes: int = 10):
        super().__init__()
        
        self.fc1 = PrunableLinear(input_size, hidden1)
        self.fc2 = PrunableLinear(hidden1, hidden2)
        self.out = PrunableLinear(hidden2, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Flatten CIFAR images: (Batch, Channels, Height, Width) -> (Batch, InputSize)
        x = x.view(x.size(0), -1)
        
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.out(x)
        return x

    def get_all_gates(self) -> list[torch.Tensor]:
        """Collect gates across all network layers to compute structural Sparsity Loss."""
        gates = []
        for module in self.modules():
            if isinstance(module, PrunableLinear):
                gates.append(module.get_gates())
        return gates
