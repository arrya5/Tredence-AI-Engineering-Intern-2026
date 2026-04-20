import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from typing import Dict, Any

from src.model import SelfPruningNet
from src.config import config, logger
from src.utils import calculate_sparsity, plot_gate_distribution

def train_and_evaluate(lmbda: float) -> Dict[str, Any]:
    """
    Completes a full CIFAR-10 training iteration using custom L1 gate penalty.
    """
    logger.info(f"--- Triggering network training with Lambda = {lmbda} ---")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Computational backend: {device}")
    
    # Pipeline: Normalize CIFAR-10 inputs uniformly
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False)
    
    model = SelfPruningNet(
        input_size=config.input_size, 
        hidden1=config.hidden_size_1, 
        hidden2=config.hidden_size_2,
        num_classes=config.num_classes
    ).to(device)
    
    criterion = nn.CrossEntropyLoss()
    # Optimizing our weights and our gate_scores together
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
    
    # Execution Loop
    for epoch in range(1, config.epochs + 1):
        model.train()
        running_loss = 0.0
        
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            
            # The Classification Base Loss
            cls_loss = criterion(outputs, targets)
            
            # Formulating the Sparsity Loss specifically over the bounded gates
            gates = model.get_all_gates()
            sparsity_loss = sum(torch.sum(torch.abs(g)) for g in gates)
            
            # The Final Combined Regularized Loss Strategy
            loss = cls_loss + lmbda * sparsity_loss
            
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
        logger.info(f"Epoch {epoch}/{config.epochs} mapped with average Loss: {running_loss/len(train_loader):.4f}")
        
    # Validation Loop
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()
            
    accuracy = 100 * correct / total
    
    # Output the architectural change
    final_gates = model.get_all_gates()
    sparsity = calculate_sparsity(final_gates, threshold=config.sparsity_threshold)
    
    logger.info(f"Final Configuration metrics: Accuracy -> {accuracy:.2f}% | Derived Sparsity -> {sparsity:.2f}%")
    
    return {
        "lambda": lmbda,
        "accuracy": accuracy,
        "sparsity": sparsity,
        "model": model
    }

if __name__ == "__main__":
    test_results = []
    optimal_pruned_model = None
    
    for l in config.lambdas:
        metrics = train_and_evaluate(l)
        test_results.append(metrics)
        
        # Save a moderately pruned model object to visualize the graph.
        if optimal_pruned_model is None or metrics["sparsity"] > 40.0:
            optimal_pruned_model = metrics["model"]
            
    # Markdown Table formatting output for report writing
    print("\n" + "="*60)
    print("FINAL SPARSIFICATION REPORTING TABLE")
    print("="*60)
    print(f"{'Lambda':<15} | {'Test Accuracy (%)':<20} | {'Sparsity Level (%)':<20}")
    print("-" * 60)
    for res in test_results:
        print(f"{res['lambda']:<15} | {res['accuracy']:<20.2f} | {res['sparsity']:<20.2f}")
        
    if optimal_pruned_model:
        logger.info("Executing visualization graph render logic...")
        plot_gate_distribution(optimal_pruned_model.get_all_gates(), save_path="gate_distribution.png")
