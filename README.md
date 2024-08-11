# MNIST Recognition Project

![MNIST GIF](./image_processing20190826-2795-5y3Jjg.gif)



This project demonstrates a machine learning model for recognizing handwritten digits using the MNIST dataset. The model is implemented using PyTorch and trained on the MNIST dataset to classify digits from 0 to 9.

## Project Overview

The project involves the following key steps:

1. **Data Preparation**: The MNIST dataset is downloaded and preprocessed using PyTorch's `transforms` module. The data is normalized and converted to tensors for training and testing.

2. **Model Architecture**: A simple neural network model is defined with three fully connected layers:
   - Layer 1: Input layer with 784 features (28x28 pixels) and 512 neurons.
   - Layer 2: Hidden layer with 256 neurons.
   - Layer 3: Output layer with 10 neurons (one for each digit).

3. **Training**: The model is trained for 5 epochs using the Adam optimizer and CrossEntropyLoss as the loss function. The training loop involves forward propagation, loss calculation, backpropagation, and optimizer steps.

4. **Evaluation**: The model's performance is evaluated on the test dataset, and accuracy metrics are calculated for both training and testing phases.

5. **Visualization**: Sample images from the test dataset are visualized along with their ground truth and predicted labels.

6. **Model Saving**: The trained model is saved as `mnist_model.pth` for future use.

## Getting Started

### Prerequisites

- Python 3.x
- PyTorch
- Torchvision
- Matplotlib
- Numpy

### Installation

Clone the repository and install the required dependencies:

```bash
git clone https://github.com/your-username/MNIST_Recognition.git
cd MNIST_Recognition
pip install -r requirements.txt
```

### Running the Project

To run the project, simply execute the Jupyter Notebook or Python script:

```bash
jupyter notebook MNIST_Recognition_machine.ipynb
```

or

```bash
python mnist_recognition.py
```

### Usage

1. **Training the Model**: The notebook or script will automatically train the model on the MNIST dataset. Training progress, including loss and accuracy, will be displayed.

2. **Testing the Model**: After training, the model will be evaluated on the test dataset, and the results will be printed.

3. **Visualization**: The notebook includes a section for visualizing test images along with their predicted labels.

4. **Saving the Model**: The trained model will be saved as `mnist_model.pth` in the current directory.

## Results

After 5 epochs of training, the model achieves the following performance:

- **Train Loss**: 0.0370
- **Train Accuracy**: 98.81%
- **Test Loss**: 0.0732
- **Test Accuracy**: 97.87%

## Acknowledgments

- [PyTorch](https://pytorch.org/) - The deep learning framework used for this project.
- [MNIST Dataset](http://yann.lecun.com/exdb/mnist/) - The dataset used for training and testing.
