# MoodWave

<img src="https://cdn.dribbble.com/users/319371/screenshots/7050507/media/425a3dacb0db3753ce4e6cfbf7039e82.gif" alt="MoodWave" width="600"/>

**MoodWave** is a deep learning-based expression classifier built using PyTorch. This project is designed to recognize facial expressions from images and is trained on a dataset that contains several classes of emotions.

## Table of Contents

- [Introduction](#introduction)
- [Setup and Installation](#setup-and-installation)
- [Data Preparation](#data-preparation)
- [Model Architecture](#model-architecture)
- [Training the Model](#training-the-model)
- [Evaluation](#evaluation)
- [Results](#results)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)

## Introduction

MoodWave uses a Convolutional Neural Network (CNN) to classify different facial expressions. This project is an implementation that can be extended to various real-world applications like emotion detection in customer service, human-computer interaction, and more.

## Setup and Installation

### Dependencies

Ensure you have the following dependencies installed:

- Python 3.x
- PyTorch
- Torchvision
- Matplotlib
- OpenCV

You can install these dependencies using pip:

```bash
pip install torch torchvision opencv-python matplotlib
```

### Device Setup

The project is device agnostic and will automatically use the GPU if available, or fall back to the CPU.

```python
import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
```

## Data Preparation

The dataset contains images categorized by different emotions. Each category is stored in separate directories, and these images are then processed, normalized, and transformed into tensors for training.

### Example Code:

```python
from torchvision import datasets, transforms

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=64, shuffle=False)
```

## Model Architecture

The model is a simple neural network with three fully connected layers designed to classify the MNIST dataset images into 10 classes.

### Example Code:

```python
import torch.nn as nn

class MNIST_model0(nn.Module):
    def __init__(self):
        super(MNIST_model0, self).__init__()
        self.layer_1 = nn.Linear(28*28, 512)
        self.layer_2 = nn.Linear(512, 256)
        self.layer_3 = nn.Linear(256, 10)

    def forward(self, x):
        x = x.view(-1, 28*28)
        x = torch.relu(self.layer_1(x))
        x = torch.relu(self.layer_2(x))
        x = self.layer_3(x)
        return x

model1 = MNIST_model0().to(device)
```

## Training the Model

The model is trained using the CrossEntropyLoss function and the Adam optimizer over a set number of epochs.

### Example Code:

```python
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(params=model1.parameters(), lr=0.001)

epochs = 5
for epoch in range(epochs):
    # Training code here...
```

## Evaluation

After training, the model is evaluated on the test dataset, and the accuracy is calculated.

### Example Code:

```python
model1.eval()
correct = 0
total = 0

with torch.no_grad():
    for data, target in test_loader:
        data, target = data.to(device), target.to(device)
        outputs = model1(data)
        _, predicted = torch.max(outputs.data, 1)
        total += target.size(0)
        correct += (predicted == target).sum().item()

print(f'Accuracy: {100 * correct / total}%')
```

## Results

The model achieved an accuracy of **98.81%** on the training set and **97.87%** on the test set after 5 epochs.

## Usage

To use this project:

1. Clone the repository.
2. Install the dependencies.
3. Run the provided notebook to train the model.
4. Use the trained model to classify new images.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License.

---

This `README.md` should serve as a comprehensive guide to understanding and using your MoodWave project. The provided GIF adds a visually engaging element to the introduction.
