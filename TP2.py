import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from tqdm import tqdm
import os
import time

# -------------------------------------------
# Dataset Preparation
# -------------------------------------------
transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(32, padding=4),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# train_set = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
# test_set = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transforms.Compose([
#     transforms.ToTensor(),
#     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
# ]))

train_set  = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transforms.ToTensor())
test_set = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transforms.ToTensor())

train_loader = torch.utils.data.DataLoader(train_set, batch_size=128)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=128)

# -------------------------------------------
# Model Definitions
# -------------------------------------------

    # -------------------------------------------
    # Simple CNN
    # -------------------------------------------
    
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.a = nn.Linear(3 * 32 * 32, 3 * 16 * 16)
        
        self.b = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        self.c = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        self.d = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        self.e = nn.Linear(2 * 16 * 16, 10)
        
    def forward(self, x):
        x = x.view(x.size(0), -1)
        h1 = self.a(x)
        h1 = h1.view(-1, 3, 16, 16)
        
        # Residual connections
        residual = h1
        x = self.b(h1)
        if x.shape == residual.shape:
            x += residual

        residual = x
        x = self.c(x)
        if x.shape == residual.shape:
            x += residual

        residual = x
        x = self.d(x)
        if x.shape == residual.shape:
            x += residual
            
        x = x.view(-1, 2 * 16 * 16)
        return self.e(x)
    
    # -------------------------------------------
    # Bottleneck
    # -------------------------------------------
    
class Bottleneck(nn.Module):
    def __init__(self, in_channels, bottleneck_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(Bottleneck, self).__init__()
        
        self.reduce = nn.Sequential(
            nn.Conv2d(in_channels, bottleneck_channels, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(bottleneck_channels),
            nn.ReLU()
        )
        
        self.maintain = nn.Sequential(
            nn.Conv2d(bottleneck_channels, bottleneck_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False),
            nn.BatchNorm2d(bottleneck_channels),
            nn.ReLU()
        )
        
        self.expand = nn.Sequential(
            nn.Conv2d(bottleneck_channels, out_channels, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(out_channels)
        )
        
        self.residual_connection = (in_channels == out_channels) and (stride == 1)
        
    def forward(self, x):
        residual = x
        x = self.reduce(x)
        x = self.maintain(x)
        x = self.expand(x)
        if self.residual_connection:
            x += residual
        return F.relu(x)
    
class BottleneckStack(nn.Module):
    def __init__(self, in_channels, bottleneck_channels, out_channels, num_blocks=3, kernel_size=3):
        super(BottleneckStack, self).__init__()
        layers = []
        
        # Premier bloc : Réduction
        layers.append(Bottleneck(in_channels, bottleneck_channels, bottleneck_channels, kernel_size=kernel_size))
        
        # Blocs intermédiaires : Maintien
        for _ in range(num_blocks - 2):
            layers.append(Bottleneck(bottleneck_channels, bottleneck_channels, bottleneck_channels, kernel_size=kernel_size))
        
        # Dernier bloc : Expansion ou retour à la dimension initiale
        layers.append(Bottleneck(bottleneck_channels, bottleneck_channels, out_channels, kernel_size=kernel_size))
        
        self.blocks = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.blocks(x)
    
class BottleneckNet(nn.Module):
    def __init__(self, num_classes=10):
        super(BottleneckNet, self).__init__()
        self.initial_conv = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        
        # Stacking de BottleneckStack avec différentes tailles de canaux
        self.stack1 = BottleneckStack(64, 32, 128, num_blocks=3, kernel_size=3)
        self.stack2 = BottleneckStack(128, 64, 256, num_blocks=3, kernel_size=3)
        self.stack3 = BottleneckStack(256, 128, 512, num_blocks=3, kernel_size=3)
        
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)
    
    def forward(self, x):
        x = self.initial_conv(x)
        x = self.stack1(x)
        x = self.stack2(x)
        x = self.stack3(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x
    
    # -------------------------------------------
    # Inverted Bottleneck
    # -------------------------------------------
    
class InvertedBottleneck(nn.Module):
    def __init__(self, in_channels, expanded_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(InvertedBottleneck, self).__init__()
        
        # Expansion (1x1 convolution)
        self.expand = nn.Sequential(
            nn.Conv2d(in_channels, expanded_channels, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(expanded_channels),
            nn.ReLU6(inplace=True)
        )
        
        # Depthwise convolution
        self.depthwise = nn.Sequential(
            nn.Conv2d(expanded_channels, expanded_channels, kernel_size=kernel_size, stride=stride, padding=padding, 
                        groups=expanded_channels, bias=False),
            nn.BatchNorm2d(expanded_channels),
            nn.ReLU6(inplace=True)
        )
        
        # Compression (1x1 convolution)
        self.compress = nn.Sequential(
            nn.Conv2d(expanded_channels, out_channels, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(out_channels)
        )
        
        # Residual connection condition
        self.use_residual = (in_channels == out_channels) and (stride == 1)

    def forward(self, x):
        residual = x
        x = self.expand(x)
        x = self.depthwise(x)
        x = self.compress(x)
        if self.use_residual:
            x += residual
        return x
    
class InvertedBottleneckStack(nn.Module):
    def __init__(self, in_channels, expanded_channels, out_channels, num_blocks=3, kernel_size=3):
        super(InvertedBottleneckStack, self).__init__()
        
        layers = []
        # Premier bloc : Expansion
        layers.append(InvertedBottleneck(in_channels, expanded_channels, out_channels, kernel_size=kernel_size))
        
        # Blocs intermédiaires : Maintien
        for _ in range(num_blocks - 1):
            layers.append(InvertedBottleneck(out_channels, expanded_channels, out_channels, kernel_size=kernel_size))
        
        self.blocks = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.blocks(x)

class InvertedBottleneckNet(nn.Module):
    def __init__(self, num_classes=10):
        super(InvertedBottleneckNet, self).__init__()
        
        self.initial_conv = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU6(inplace=True)
        )
        
        self.stack1 = InvertedBottleneckStack(32, 64, 64, num_blocks=3, kernel_size=3)
        self.stack2 = InvertedBottleneckStack(64, 128, 128, num_blocks=3, kernel_size=3)
        self.stack3 = InvertedBottleneckStack(128, 256, 256, num_blocks=3, kernel_size=3)
        
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(256, num_classes)
    
    def forward(self, x):
        x = self.initial_conv(x)
        x = self.stack1(x)
        x = self.stack2(x)
        x = self.stack3(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

# -------------------------------------------
# Utility Functions
# -------------------------------------------
def accuracy(outputs, labels):
    """Calculate the accuracy given model outputs and labels."""
    _, preds = torch.max(outputs, dim=1)
    return torch.tensor(torch.sum(preds == labels).item() / len(preds))

def validate(model, loader):
    """Validate the model's accuracy on a given DataLoader."""
    model.eval() # Set model to evaluation mode
    # acc = [accuracy(model(x), y) for x, y in tqdm(loader)]
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in tqdm(loader):
            x, y = x.to(device), y.to(device)
            outputs = model(x)
            _, predicted = torch.max(outputs.data, 1)
            total += y.size(0)
            correct += (predicted == y).sum().item()
    return 100 * correct / total

def fit_one_cycle(model, loss_fn, opt, train_loader, scheduler=None):
    """Train for one epoch and validate."""
    model.train() # Set model to training mode
    loss = 0.0
    for x, y in tqdm(train_loader):
        x, y = x.to(device), y.to(device)
        preds = model(x)
        loss = loss_fn(preds, y)
        loss.backward()
        opt.step()
        if scheduler:
            scheduler
        opt.zero_grad()
        loss += loss.item()

# -------------------------------------------
# Main Training Loop
# -------------------------------------------
def wait(time_ms):
    """Wait for a given time in milliseconds."""
    start = time.time()
    while time.time() - start < time_ms / 1000:
        pass

device = "cpu"
if torch.cuda.is_available():
    device = "cuda"
    print("-----------------------------")
    print("Using NVIDIA CUDA backend for GPU computations")
    print("-----------------------------")
  
# Check if using Linux and if ROCm is installed
if os.name == "posix":
    try:
        if torch.backends.hip.is_available():
            device = "hip"
            print("-----------------------------")
            print("Using AMD ROCm backend for GPU computations")
            print("-----------------------------")
    except:
        pass

if device == "cpu":
    print("-----------------------------")
    print("Using CPU")
    print("-----------------------------")
    
wait(1000)
    
model = SimpleCNN().to(device)
loss_fn = nn.CrossEntropyLoss()

print("-----------------------------")
print("Which type of model do you want to train?")
print("-----------------------------")
print("1. SimpleCNN")
print("2. BottleneckNet")
print("3. InvertedBottleneckNet")
print("-----------------------------")
print("> ", end="")
model_choice = int(input())
if model_choice == 2:
    model = BottleneckNet().to(device)
elif model_choice == 3:
    model = InvertedBottleneckNet().to(device)

print("-----------------------------")
print("How many epochs do you want to train for?")
print("-----------------------------")
print("> ", end="")
epochs = None
while epochs is None:
    try:
        epochs = int(input())
    except ValueError:
        print("Please enter a valid integer.")

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.01, steps_per_epoch=len(train_loader), epochs=epochs)

for epoch in range(epochs):
    print(f"Epoch {epoch + 1}/{epochs}")
    fit_one_cycle(model, loss_fn, optimizer, train_loader, scheduler)
    print("Validating...")
    train_acc = validate(model, train_loader)
    test_acc = validate(model, test_loader)
    print(f"Train Accuracy: {train_acc:.4f}, Test Accuracy: {test_acc:.4f}")