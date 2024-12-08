# -------------------------------------------
# Au cas où vous souhaitez voir la toute première version du TP, le voici :
# Il est assez simple, il m'a permit de comprendre comment fonctionne la librairie PyTorch et de faire quelques tests.
# Il est également assez dans le désordre, mais je l'ai laissé tel quel pour montrer ma progression.
# Il y a également certaines parties commentées (dont notamment les parties d'affichage d'images) que vous pouvez décommenter.
# -------------------------------------------

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms

# Load the CIFAR-10 dataset
train_set = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transforms.ToTensor())
test_set = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transforms.ToTensor())

# Display the first image
image, label = train_set[0]
# plt.imshow(image.numpy().transpose(1, 2, 0))
# plt.show()

# Transform image from tensor to numpy array
img = image.numpy()
# Afficher les canaux de l'image
# for i in range(3):
#     plt.imshow(img[i])
#     plt.show()

# DataLoader
train_loader = torch.utils.data.DataLoader(train_set, batch_size=4)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=4)

# Display
data_iter = iter(train_loader)
batch = next(data_iter)
print(batch)

fc = nn.Linear(3 * 32 * 32, 10)

# Run the fc with batch[0]
output = fc(batch[0].view(-1, 3 * 32 * 32))
# output = fc(batch[0])
print(output.shape)

class MyClass(nn.Module):
    def __init__(self):
        super().__init__()
        self.a = nn.Linear(3 * 32 * 32, 32)
        self.b = nn.Linear(32, 24)
        self.c = nn.Linear(24, 10)

    def forward(self, x):
        return self.c(self.b(self.a(x)))
    
model = MyClass()
output = model(batch[0].view(-1, 3 * 32 * 32))
print(output)
probs = nn.functional.softmax(output, dim=1)
print(probs)
maxProbs, index = torch.max(probs, dim=1)
print(maxProbs, index)

def accuracy(outputs, labels):
    _, preds = torch.max(outputs, dim=1)
    return torch.tensor(torch.sum(preds == labels).item() / len(preds))

# Test the model on the first batch
# correct = 0
# total = 0
# for images, labels in train_loader:
#     outputs = model(images.view(-1, 3 * 32 * 32))
#     _, predicted = torch.max(outputs, dim=1)
#     total += labels.size(0)
#     correct += int((predicted == labels).sum())
# print('Accuracy: ', correct / total)
def accuracy(outputs, labels):
    _, preds = torch.max(outputs, dim=1)
    return torch.tensor(torch.sum(preds == labels).item() / len(preds))

# Calculate accuracy on every batch
def validate(model, train_loader):
    acc = [accuracy(model(x.view(-1, 3 * 32 * 32)), y) for x, y in train_loader]
    return sum(acc) / len(acc)

print(validate(model, train_loader))

lossFn = F.cross_entropy

loss = lossFn(model(batch[0].view(-1, 3 * 32 * 32)), batch[1])
print(loss)

learning_rate = 0.001
opt = torch.optim.SGD(model.parameters(), lr=learning_rate)

preds = model(batch[0].view(-1, 3 * 32 * 32))
loss = lossFn(preds, batch[1])
loss.backward()

opt.step()
opt.zero_grad()

# We print the accuracy
print(validate(model, train_loader))

# Function that does one whole iteration on all the batches of DataLoader, does train and validation
def fit_one_cycle(model, lossFn, opt, train_loader, test_loader):
    for x, y in train_loader:
        preds = model(x.view(-1, 3 * 32 * 32))
        loss = lossFn(preds, y)
        loss.backward()
        opt.step()
        opt.zero_grad()
    print(validate(model, train_loader), validate(model, test_loader))

epochs = 10
for i in range(epochs):
    fit_one_cycle(model, lossFn, opt, train_loader, test_loader)

