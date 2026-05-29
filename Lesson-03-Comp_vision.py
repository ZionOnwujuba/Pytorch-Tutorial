# Import PyTorch
import torch
from torch import nn
from torch.utils.data import DataLoader

# Import torchvision 
import torchvision
from torchvision import datasets
from torchvision.transforms import ToTensor

# Import matplotlib for visualization
import matplotlib.pyplot as plt

# Import accuracy metric
from helper_functions import accuracy_fn # Note: could also use torchmetrics.Accuracy(task = 'multiclass', 
                                         # num_classes=len(class_names)).to(device)

from timeit import default_timer as timer 
from tqdm.auto import tqdm # Makes loops show a smart progress meter, just wrap any iterable and it's ready

import requests
from pathlib import Path 

import pandas as pd
import random

device = "cuda" if torch.cuda.is_available() else "cpu"

import mlxtend 
print(mlxtend.__version__)
assert int(mlxtend.__version__.split(".")[1]) >= 19 # should be version 0.19.0 or higher

from torchmetrics import ConfusionMatrix
from mlxtend.plotting import plot_confusion_matrix

# Download helper functions from Learn PyTorch repo (if not already downloaded)
if Path("helper_functions.py").is_file():
  print("helper_functions.py already exists, skipping download")
else:
  print("Downloading helper_functions.py")
  # Note: you need the "raw" GitHub URL for this to work
  request = requests.get("https://raw.githubusercontent.com/mrdbourke/pytorch-deep-learning/main/helper_functions.py")
  with open("helper_functions.py", "wb") as f:
    f.write(request.content)


"""
Computer vision libraries in PyTorch

Structure:
Pytorch module
    What does it do

torchvision 	
    Contains datasets, model architectures and image 
    transformations often used for computer vision problems.

torchvision.datasets 	
    Here you'll find many example computer vision datasets for 
    a range of problems from image classification, object detection, 
    image captioning, video classification and more. It also contains 
    a series of base classes for making custom datasets.

torchvision.models 	
    This module contains well-performing and commonly used computer 
    vision model architectures implemented in PyTorch, you can use 
    these with your own problems.

torchvision.transforms 	
    Often images need to be transformed (turned into numbers/processed/augmented)
    before being used with a model, common image transformations are found here.

torch.utils.data.Dataset 	
    Base dataset class for PyTorch.

torch.utils.data.DataLoader 	
    Creates a Python iterable over a dataset (created with torch.utils.data.Dataset).

The torch.utils.data.Dataset and torch.utils.data.DataLoader classes 
aren't only for computer vision in PyTorch, they are capable of dealing 
with many different types of data.
"""

"""
We will get a computer vision dataset from FashionMNIST 
    (Fashion Modified National Institute of Standards and Technology).
    It contains grayscale images of 10 different kinds of clothing.

PyTorch has a bunch of common computer vision datasets stored in torchvision.datasets.
    Including FashionMNIST in torchvision.datasets.FashionMNIST().
        To download it, we provide the following parameters:
            root: str - which folder do you want to download the data to?
            train: Bool - do you want the training or test split?
            download: Bool - should the data be downloaded?
            transform: torchvision.transforms - what transformations would you like to do on the data?
            target_transform - you can transform the targets (labels) if you like too.
        Many other datasets in torchvision have these parameter options.
"""

print("\n\n\nCreating the data\n\n\n")

# Setup training data
train_data = datasets.FashionMNIST(
    root="data", # where to download data to
    train=True, # get training data
    download=True, # downlaod data if it doesn't exist on disk
    transform=ToTensor(), # images come as PIL format, we want to turn into Torch tensors
    target_transform=None # you can transform labels as well
)

# Setup testing data
test_data = datasets.FashionMNIST(
    root="data",
    train=False, # get test data
    download=True,
    transform=ToTensor()
)

image, label = train_data[0]
print(f"\nImage: {image}\n\nLabel: {label}\n\n")
"""
The shape of the image tensor is [1, 28, 28] or more specifically:
[color_channels=1, height=28, width=28]

Various problems will have various input and output shapes. 
    But the premise remains: encode data into numbers, build 
    a model to find patterns in those numbers, convert those 
    patterns into something meaningful.

The order of our current tensor is often referred to as CHW (Color Channels, Height, Width).
    There's debate on whether images should be represented as CHW (color channels first) or 
    HWC (color channels last).
        You'll also see NCHW and NHWC formats where N stands for number of images. 
        For example if you have a batch_size=32, your tensor shape may be 
        [32, 1, 28, 28]. We'll cover batch sizes later.
        PyTorch generally accepts NCHW (channels first) as the default for many operators.
            However, PyTorch also explains that NHWC (channels last) performs better and 
            is considered best practice.
    For now, since our dataset and models are relatively small, this won't make too much of a difference.
        But keep it in mind for when you're working on larger image datasets and using convolutional 
        neural networks (we'll see these later).

"""
print(f"Image Shape: {image.shape}\n")
print(f"Amount of training data: {len(train_data.data)}\n Amount of training targets: {len(train_data.targets)}\n Amount of test data: {len(test_data.data)}\n Amount of test targets: {len(test_data.targets)}\n")
class_names = train_data.classes
print(f"Class Names: {train_data.classes}\n")

"""
We can plot a grayscale of Index 0 of the data set
"""
image, label = train_data[0]
print(f"Index 0 image shape: {image.shape}")
plt.imshow(image.squeeze(), cmap="gray") # image shape is [1, 28, 28] (colour channels, height, width)
plt.title(class_names[label]);
plt.show(block=True)

"""
We can generate some more
"""

torch.manual_seed(42)
fig = plt.figure(figsize=(9, 9))
rows, cols = 4, 4
for i in range(1, rows * cols + 1):
    random_idx = torch.randint(0, len(train_data), size=[1]).item()
    img, label = train_data[random_idx]
    fig.add_subplot(rows, cols, i)
    plt.imshow(img.squeeze(), cmap="gray")
    plt.title(class_names[label])
    plt.axis(False)
    plt.show(block=True)

"""
Now we utilize the DataLoader which helps load data with the model for
    training and inference by turning a large Dataset into a Python
    iterable of smaller chunks called batches or mini-batches and can be
    set by the batch_size parameter
        With larger data sets, doing forward and backward passes on all the 
        data at once stresses the computing power of the machine so breaking
        it up into batches reduces that computational strain
        With mini-batches, gradient descent is perforned more often per
        epoch, giving the model more opportunites to improve

32 is a good batch size to start (powers of 2 are most common)
"""

# Setup the batch size hyperparameter
BATCH_SIZE = 32

# Turn datasets into iterables (batches)
train_dataloader = DataLoader(train_data, # dataset to turn into iterable
                              batch_size=BATCH_SIZE, # samples per batch
                              shuffle=True) # shuffle data per epoch

test_dataloader = DataLoader(test_data, # dataset to turn into iterable
                              batch_size=BATCH_SIZE, # samples per batch
                              shuffle=False) # don't shuffle data per epoch

print(f"\nDataloaders: {train_dataloader, test_dataloader}") 
print(f"\nLength of train dataloader: {len(train_dataloader)} batches of {BATCH_SIZE}")
print(f"\nLength of test dataloader: {len(test_dataloader)} batches of {BATCH_SIZE}")

# Checking what's in the training dataloader
train_features_batch, train_labels_batch = next(iter(train_dataloader))
print(f"\nTrain Features batch shape: {train_features_batch.shape}\n Train Label batch shape: {train_labels_batch.shape}")

# Show a sample
torch.manual_seed(42)
random_idx = torch.randint(0, len(train_features_batch), size=[1]).item()
img, label = train_features_batch[random_idx], train_labels_batch[random_idx]
plt.imshow(img.squeeze(), cmap="gray")
plt.title(class_names[label])
plt.axis("Off")
plt.show(block=True)
print(f"\nImage size: {img.shape}\n")
print(f"\nLabel: {label}, label size: {label.shape}\n")


print("\n\n\nCreating the model\n\n\n")

"""
Now we will create a baseline model to act as a starting point and improve
    on it with more complicated models

It will consist of 2 linear layers but with a nn.Flattern() layer which 
    compresses the tensor it is in into a single vector. Since we are using
    image data (which is a (1,3))

An example is below:
"""

# Create a flatten layer
flatten_model = nn.Flatten()

# Get a single sample
x = train_features_batch[0]

# Flatten the sample
output = flatten_model(x)

# Print out what happened
print(f"Shape before flattening: {x.shape} -> [color_channels, height, width]")
print(f"Shape after flattening: {output.shape} -> [color_channels, height*width]")

class FashionMNISTModelV0(nn.Module):
    def __init__(self, input_shape: int, hidden_units: int, output_shape: int):
        super().__init__()
        self.layer_stack = nn.Sequential(
            nn.Flatten(), # neural networks like their inputs in vector form
            nn.Linear(in_features=input_shape, out_features=hidden_units), # in_features = # of features in a data sample (784 pixels)
            nn.Linear(in_features=hidden_units, out_features=output_shape)
        )
    
    def forward(self, x):
        return self.layer_stack(x)
"""
Now we can instantiate a model

We'll need to set the following parameters:
    input_shape=784 
        this is how many features you've got going in the model, 
        in our case, it's one for every pixel in the target image 
        (28 pixels high by 28 pixels wide = 784 features).
    hidden_units=10
        number of units/neurons in the hidden layer(s), this number 
        could be whatever you want but to keep the model small we'll 
        start with 10.
    output_shape=len(class_names)
        since we're working with a multi-class classification problem, 
        we need an output neuron per class in our dataset.

"""
torch.manual_seed(42)
model_0 = FashionMNISTModelV0(input_shape=784, # one for every pixel (28x28)
                              hidden_units=10, # # of units in hidden layer
                              output_shape=len(class_names) # One for each class
                              )
print(f"\nModel 0: {model_0.to("cpu")}\n") # keep model on CPU to begin with

# Setup loss function and optimizer
loss_fn = nn.CrossEntropyLoss() # this is also called the "criterion"/"cost function"
optimizer = torch.optim.SGD(params=model_0.parameters(), lr=0.1)

"""
With the loss and optimizer ready, we can make a timing function to measure the time 
taken to train hte model on CPU vs GPU (Device has no GPU, so it only measures CPU)
"""

def print_train_time(start: float, end: float, device: torch.device = None):
    """Prints difference between start and end time.

    Args:
        start (float): Start time of computation (preferred in timeit format). 
        end (float): End time of computation.
        device ([type], optional): Device that compute is running on. Defaults to None.

    Returns:
        float: time between start and end in seconds (higher is longer).
    """
    total_time = end - start
    print(f"Train time on {device}: {total_time:.3f} seconds")
    return total_time

print("\n\n\nTraining and Testing\n\n\n")
"""
Now we will do the training and test loop, however since we have
batches we need a loop for each batch. So the steps now will be:
    Loop through epochs.
    Loop through training batches, perform training steps, calculate the train loss per batch.
    Loop through testing batches, perform testing steps, calculate the test loss per batch.
    Print out what's happening.
    Time it all.
"""

torch.manual_seed(42)
train_time_start_on_cpu = timer()

# Set the number of epochs (kept small for faster training time)
epochs = 3

# Create training and testing loop
for epoch in tqdm(range(epochs)):
    print(f"Epoch: {epoch}\n-------")
    # Training
    train_loss = 0
    # Add a loop to loop through training batches
    for batch, (X, y) in enumerate(train_dataloader):
        model_0.train()
        # 1. Forward pass
        y_pred = model_0(X)

        # 2. Calculate loss (per batch)
        loss = loss_fn(y_pred, y)
        train_loss += loss # accumulatively  add up the loss per epoch

        # 3. Optimizer zero grad
        optimizer.zero_grad()

        # 4. Loss backward
        loss.backward()

        # 5. Optimizer step
        optimizer.step()

        # Print out # of samples that have been seen
        if batch % 400 == 0:
            print(f"Looked at {batch * len(X)}/{len(train_dataloader.dataset)} samples")
    # Divide total train loss by length of train dataloader (avg loss/epoch)
    train_loss /= len(train_dataloader)

    ### Testing
    # Setup variables for accumulatively adding up loss and accuracy
    test_loss, test_acc = 0, 0
    model_0.eval()
    with torch.inference_mode():
        for X, y in test_dataloader:
            # 1. Forward pass
            test_pred = model_0(X)

            # 2. Calculate loss (accumulatively)
            test_loss += loss_fn(test_pred, y) # accumulatively add up the loss per epoch

            # 3. Calculate accuracy (preds need to be the same as y_true)
            test_acc += accuracy_fn(y_true=y, y_pred=test_pred.argmax(dim=1))
        # Calculations on test metrics need to happen inside torch.inference_mode()
        # Divide total test loss by length of test dataloader (per batch)
        test_loss /= len(test_dataloader)

        # Divide total accuracy by length of test dataloader (per batch)
        test_acc /= len(test_dataloader)
    # Print out what's happening
    print(f"\nTrain loss: {train_loss:.5f} | Test loss: {test_loss:.5f}, Test acc: {test_acc:.2f}%\n")   

# Calculate training time 
train_time_end_on_cpu = timer()
total_train_time_model_0 = print_train_time(start=train_time_start_on_cpu,
                                            end=train_time_end_on_cpu,
                                            device=str(next(model_0.parameters()).device))

torch.manual_seed(42)

"""
Since we're going to be building a few models, 
it's a good idea to write some code to evaluate them all in similar ways.
    Namely, let's create a function that takes in a trained 
    model, a DataLoader, a loss function and an accuracy function.
        The function will use the model to make predictions on the data 
        in the DataLoader and then we can evaluate those predictions 
        using the loss function and accuracy function.
"""

def eval_model(model: torch.nn.Module,
               data_loader: torch.utils.data.DataLoader,
               loss_fn: torch.nn.Module,
               accuracy_fn,
               device: torch.device = device):
    """Returns a dictionary containing the results of 
        model predicting on data_loader.

    Args:
        model (torch.nn.Module): A PyTorch model capable of making 
            predictions on data_loader.
        data_loader (torch.utils.data.DataLoader): 
            The target dataset to predict on.
        loss_fn (torch.nn.Module): The loss function of model.
        accuracy_fn: An accuracy function to compare the 
            models predictions to the truth labels.

    Returns:
        (dict): Results of model making predictions on data_loader.
    """
    loss, acc = 0, 0
    model.eval()
    with torch.inference_mode():
        for X, y in data_loader:
            # Make predictions with the model
            X, y = X.to(device), y.to(device)
            y_pred = model(X)

            # Accumulate the loss and accuracy values per batch
            loss += loss_fn(y_pred, y)
            acc += accuracy_fn(y_true=y,
                               y_pred=y_pred.argmax(dim=1)) # For accuracy, need the prediction labels
                                                            # (logits -> pred_prob -> pred_labels)
        
        # Scale loss and acc to find the average loss.acc per batch
        loss /= len(data_loader)
        acc /= len(data_loader)

    return {"model_name": model.__class__.__name__, # Only works when model was created with a class
            "model_loss": loss.item(),
            "model_acc": acc}
model_0_results = eval_model(model=model_0, data_loader=test_dataloader,
                             loss_fn=loss_fn, accuracy_fn=accuracy_fn,
                             device=device
                             )

print(f"\nModel 0 Results: {model_0_results}\n")

print("\n\n\nCreating a better model with non linearity\n\n\n")

class FashionMNISTModelV1(nn.Module):
    def __init__(self, input_shape: int, hidden_units: int, output_shape: int):
        super().__init__()
        self.layer_stack = nn.Sequential(
            nn.Flatten(), # neural networks like their inputs in vector form
            nn.Linear(in_features=input_shape, out_features=hidden_units), # in_features = # of features in a data sample (784 pixels)
            nn.ReLU(),
            nn.Linear(in_features=hidden_units, out_features=output_shape),
            nn.ReLU()
        )
    
    def forward(self, x):
        return self.layer_stack(x)
    
torch.manual_seed(42)
model_1 = FashionMNISTModelV1(input_shape=784, # one for every pixel (28x28)
                              hidden_units=10, # # of units in hidden layer
                              output_shape=len(class_names) # One for each class
                              ).to(device)

print(f"\nCheck model device: {next(model_1.parameters()).device}\n")

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(params=model_1.parameters(),
                            lr=0.1)

def train_step(model: torch.nn.Module,
               data_loader: torch.utils.data.DataLoader,
               loss_fn: torch.nn.Module,
               optimizer: torch.optim.Optimizer,
               accuracy_fn,
               device: torch.device = device):
    train_loss, train_acc = 0, 0
    model.to(device)
    for batch, (X, y) in enumerate(data_loader):
        # Send data to GPU
        X, y = X.to(device), y.to(device)

        # 1. Forward pass
        y_pred = model(X)

        # 2. Calculate loss
        loss = loss_fn(y_pred, y)
        train_loss += loss
        train_acc += accuracy_fn(y_true=y,
                                 y_pred=y_pred.argmax(dim=1)) # Go from logits -> pred labels
        
        # 3. Optimizer zero grad
        optimizer.zero_grad()

        # 4. Loss backwards
        loss.backward()

        # 5. Optimizer step
        optimizer.step()
    
    # Calculate loss and accuracy per epoch and print out what's happening
    train_loss /= len(data_loader)
    train_acc /= len(data_loader)
    print(f"Train loss: {train_loss:.5f} | Train accuracy: {train_acc:.2f}%")

def test_step(model: torch.nn.Module,
              data_loader: torch.utils.data.DataLoader,
              loss_fn: torch.nn.Module,
              accuracy_fn,
              device: torch.device = device):
    test_loss, test_acc = 0, 0
    model.to(device)
    model.eval()

    with torch.inference_mode():
        for X, y in data_loader:
            # Send data to GPU
            X, y = X.to(device), y.to(device)

            # 1. Forward pass
            test_pred = model(X)

            # 2. Calculate loss and accuracy 
            test_loss += loss_fn(test_pred, y)
            test_acc += accuracy_fn(y_true=y,
                                    y_pred=test_pred.argmax(dim=1) # Go from logits -> pred labels
                                    )
            
        # Adjust metrics and print out
        test_loss /= len(data_loader)
        test_acc /= len(data_loader)
        print(f"Test loss: {test_loss:.5f} | Test accuracy: {test_acc:.2f}%\n")

torch.manual_seed(42)

train_time_end_on_cpu = timer()
epochs = 3
for epoch in tqdm(range(epochs)):
    print(f"Epoch: {epoch}\n-------")
    train_step(data_loader=train_dataloader,
               model=model_1,
               loss_fn=loss_fn,
               optimizer=optimizer,
               accuracy_fn=accuracy_fn)
    test_step(data_loader=test_dataloader,
              model=model_1,
              loss_fn=loss_fn,
              accuracy_fn=accuracy_fn)
    
train_time_end_on_cpu = timer()
total_train_time_model_1 = print_train_time(start=train_time_start_on_cpu,
                                            end=train_time_end_on_cpu,
                                            device=device)

torch.manual_seed(42)
model_1_results = eval_model(model=model_1, data_loader=test_dataloader,
    loss_fn=loss_fn, accuracy_fn=accuracy_fn,
    device=device
)

print(f"\nModel 1 Results: {model_1_results}\n")

"""
In this case, it looks like adding non-linearities to our model made it perform worse 
    than the baseline.

It seems like our model is overfitting on the training data.
    Overfitting means our model is learning the training data well but those patterns 
    aren't generalizing to the testing data.
Two of the main ways to fix overfitting include:
    Using a smaller or different model 
        (some models fit certain kinds of data better than others).
    Using a larger dataset (the more data, the more chance a model 
        has to learn generalizable patterns).
"""

print("\n\n\nCreate a Convolutional Neural Network\n\n\n")
"""
It's time to create a Convolutional Neural Network (CNN or ConvNet). 
    CNN's are known for their capabilities to find patterns in visual data.
        And since we're dealing with visual data, let's see if using a CNN model 
        can improve upon our baseline.
The CNN model we're going to be using is known as TinyVGG from the CNN Explainer website.
    It follows the typical structure of a convolutional neural network:
        Input layer -> [Convolutional layer -> activation layer -> pooling layer] -> Output layer
            Where the contents of [Convolutional layer -> activation layer -> pooling layer] 
                can be upscaled and repeated multiple times, depending on requirements.

Which model to use?

Structure:
Problem type
    Model to use (generally)
        Code example

Structured data (Excel spreadsheets, row and column data) 	
    Gradient boosted models, Random Forests, XGBoost 	
        sklearn.ensemble, XGBoost library

Unstructured data (images, audio, language) 	
    Convolutional Neural Networks, Transformers 	
        torchvision.models, HuggingFace Transformers
"""

class FashionMNISTModelV2(nn.Module):
    """
    Model architecture copying TinyVGG from: 
    https://poloclub.github.io/cnn-explainer/
    """
    def __init__(self, input_shape: int, hidden_units: int, output_shape: int):
        super().__init__()
        self.block_1 = nn.Sequential(
            nn.Conv2d(in_channels=input_shape, 
                      out_channels=hidden_units, 
                      kernel_size=3, # how big is the square that's going over the image?
                      stride=1, # default
                      padding=1),# options = "valid" (no padding) or "same" (output has same shape as input) or int for specific number 
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_units, 
                      out_channels=hidden_units,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,
                         stride=2) # default stride value is same as kernel_size
        )
        self.block_2 = nn.Sequential(
            nn.Conv2d(hidden_units, hidden_units, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_units, hidden_units, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            # Where did this in_features shape come from? 
            # It's because each layer of our network compresses and changes the shape of our input data.
            nn.Linear(in_features=hidden_units*7*7, 
                      out_features=output_shape)
        )
    
    def forward(self, x: torch.Tensor):
        x = self.block_1(x)
        x = self.block_2(x)
        x = self.classifier(x)
        return x


    
torch.manual_seed(42)
model_2 = FashionMNISTModelV2(input_shape=1,
                              hidden_units=10,
                              output_shape=len(class_names)).to(device)

print(f"\nModel 2: {model_2}\n")

"""
nn.Conv2d(), also known as a convolutional layer.
    The 2d is for 2-dimensional data. As in, our images have two dimensions: 
    height and width. Yes, there's color channel dimension but each of the color 
    channel dimensions have two dimensions too: height and width.
        For other dimensional data (such as 1D for text or 3D for 3D objects) 
        there's also nn.Conv1d() and nn.Conv3d().
nn.MaxPool2d(), also known as a max pooling layer.

To test the layers out, let's create some toy data just like the data used on CNN Explainer.
"""

torch.manual_seed(42)
# Create sample batch of random numbers with same size as image batch
images = torch.randn(size=(32, 3, 64, 64)) # [batch_size, color_channels, height, width]
test_image = images[0] # get a single image for testing
print(f"Image batch shape: {images.shape} -> [batch_size, color_channels, height, width]")
print(f"Single image shape: {test_image.shape} -> [color_channels, height, width]") 
print(f"Single image pixel values:\n{test_image}")

"""
Let's create an example nn.Conv2d() with various parameters:
    in_channels (int) - Number of channels in the input image.
    out_channels (int) - Number of channels produced by the convolution.
    kernel_size (int or tuple) - Size of the convolving kernel/filter.
    stride (int or tuple, optional) - How big of a step the convolving kernel takes at a time. Default: 1.
    padding (int, tuple, str) - Padding added to all four sides of input. Default: 0.
"""

torch.manual_seed(42)

# Create a convolutional layer with same dimensions as TinyVGG
conv_layer = nn.Conv2d(in_channels=3,
                       out_channels=10,
                       kernel_size=3,
                       stride=1,
                       padding=0)

# Pass data through the convolutional layer
print(f"\nConvolution layer through test image: {conv_layer(test_image)}\n")

"""
If we try to pass a single image in, we get a shape mismatch error

This is because our nn.Conv2d() layer expects a 4-dimensional tensor 
    as input with size (N, C, H, W) or [batch_size, color_channels, height, width].
        Right now our single image test_image only has a shape of 
            [color_channels, height, width] or [3, 64, 64].
                We can fix this for a single image using test_image.unsqueeze(dim=0) 
                to add an extra dimension for N.
"""
# Add extra dimenstion to test image
print(f"New Test Image dimension: {test_image.unsqueeze(dim=0).shape}")

# Pass test image with extra dimension through conv_layer
print(f"Pass extra dimension image through conv_layer: {conv_layer(test_image.unsqueeze(dim=0)).shape}")

"""
What if we changed the values of conv_layer?
"""

torch.manual_seed(42)
# Create a new conv_layer with different values 
conv_layer_2 = nn.Conv2d(in_channels=3, # same number of color channels as our input image
                         out_channels=10,
                         kernel_size=(5,5), # kernal is usually a square so a tuple
                         stride=2,
                         padding=0)

# Pass single image through new conv_layer_2 (this calls nn.Conv2d()'s forward() method on the input)
print(f"\nPass extra dimension image through conv_layer 2: {conv_layer_2(test_image.unsqueeze(dim=0)).shape}\n")

"""
Another shape change.

nn.Conv2d() is compressing the information stored in the image by performing
    operations on the input (our test image) against its internal parameters
        The goal is, as data goes in and the layers try to update their internal 
        parameters, to lower the loss function thanks to some help of the optimizer.
        The only difference is how different layers calulate their parameter updates
"""

# Check out the conv_layer_2 internal parameters
print(f"\nConv_layer_2 Internal parameters: {conv_layer_2.state_dict()}\n")

"""
A bunch of random numbers for a weight and bias tensor.

The shapes of these are manipulated by the inputs we 
    passed to nn.Conv2d() when we set it up.
"""

# Get shapes of weight and bias tensors within conv_layer_2
print(f"\nconv_layer_2 weight shape: \n{conv_layer_2.weight.shape} -> [out_channels=10, in_channels=3, kernel_size=5, kernel_size=5]\n")
print(f"\nconv_layer_2 bias shape: \n{conv_layer_2.bias.shape} -> [out_channels=10]\n")

"""
What should we set the parameters of our nn.Conv2d() layers?
    That's a good one. But similar to many other things in machine learning, 
    the values of these aren't set in stone (and recall, because these values 
    are ones we can set ourselves, they're referred to as "hyperparameters").
        The best way to find out is to try out different values and see how 
        they effect your model's performance.
        Or even better, find a working example on a problem similar to 
        yours (like we've done with TinyVGG) and copy it.
"""

"""
Now we can check an see what happens as we move data through nn.MaxPool2d()
"""

# Print out original image shape without and with unsqueezed dimension
print(f"\nTest image original shape: {test_image.shape}\n")
print(f"\nTest image with unsqueezed dimension: {test_image.unsqueeze(dim=0).shape}\n")

# Create a sample nn.MaxPoo2d() layer
max_pool_layer = nn.MaxPool2d(kernel_size=2)

# Pass data through just the conv_layer
test_image_through_conv = conv_layer(test_image.unsqueeze(dim=0))
print(f"\nShape after going through conv_layer(): {test_image_through_conv.shape}\n")

# Pass data through the max pool layer
test_image_through_conv_and_max_pool = max_pool_layer(test_image_through_conv)
print(f"\nShape after going through conv_layer() and max_pool_layer(): {test_image_through_conv_and_max_pool.shape}\n")

"""
Notice the change in the shapes of what's happening in 
    and out of a nn.MaxPool2d() layer.
        The kernel_size of the nn.MaxPool2d() layer will 
        affect the size of the output shape.
In our case, the shape halves from a 62x62 image to 31x31 image.
Let's see this work with a smaller tensor.
"""

torch.manual_seed(42)
# Create a random tensor with a similar number of dimensions to images
random_tensor = torch.randn(size=(1, 1, 2, 2))
print(f"\nRandom tensor: {random_tensor}\n")
print(f"\nRandom tensor shape: {random_tensor.shape}\n")

# Create a max pool layer
max_pool_layer = nn.MaxPool2d(kernel_size=2) # see what happens when you change the kernal_size value

# Pass the random tensor through the max pool layer
max_pool_tensor = max_pool_layer(random_tensor)
print(f"\nMax pool tensor: {max_pool_layer} <- this is the maximum value from random_tensor\n")
print(f"\nMax pool tensor shape: {max_pool_tensor.shape}\n")

"""
Notice the final two dimensions between random_tensor and max_pool_tensor, 
    they go from [2, 2] to [1, 1]. In essence, they get halved.
        And the change would be different for different values of 
        kernel_size for nn.MaxPool2d().
Also notice the value leftover in max_pool_tensor is the maximum value from random_tensor.
    This is another important piece of the puzzle of neural networks.
        Essentially, every layer in a neural network is trying to compress data 
        from higher dimensional space to lower dimensional space.
            In other words, take a lot of numbers (raw data) and learn patterns in those numbers, 
            patterns that are predictive whilst also being smaller in size than the original values.
            From an artificial intelligence perspective, you could consider the whole goal of a neural 
            network to compress information.
    This is the idea of the use of a nn.MaxPool2d() layer: take the maximum value from a portion of a 
    tensor and disregard the rest.
        In essence, lowering the dimensionality of a tensor whilst still retaining a 
        (hopefully) significant portion of the information.

It is the same story for a nn.Conv2d() layer.
    Except instead of just taking the maximum, the nn.Conv2d() performs a convolutional operation 
      the data (see this in action on the CNN Explainer webpage)
"""

"""
Now we will create a loss function and optimizer
"""

# Setup loss and optimizer
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(params=model_2.parameters(),
                            lr=0.1)

torch.manual_seed(42)
train_time_start_model_2 = timer()

# Train and test model
epochs = 3
for epoch in tqdm(range(epochs)):
    print(f"Epoch: {epoch}\n-------")
    train_step(data_loader=train_dataloader,
               model=model_2,
               loss_fn=loss_fn,
               optimizer=optimizer,
               accuracy_fn=accuracy_fn,
               device=device)
    test_step(data_loader=test_dataloader,
              model=model_2,
              loss_fn=loss_fn,
              accuracy_fn=accuracy_fn,
              device=device)
    
train_time_end_model_2 = timer()

total_train_time_model_2 = print_train_time(start=train_time_start_model_2, 
                                            end=train_time_end_model_2, 
                                            device=device)

# Get model_2 results
model_2_results = eval_model(
    model=model_2,
    data_loader=test_dataloader,
    loss_fn=loss_fn,
    accuracy_fn=accuracy_fn
)

print(f"\nModel 2 Results: {model_2_results}\n")

print(f"\n\n\nCompare and evaluate models\n\n\n")
compare_results = pd.DataFrame([model_0_results, model_1_results, model_2_results])
compare_results["training_time"] = [total_train_time_model_0,
                                    total_train_time_model_1,
                                    total_train_time_model_2]
print(compare_results)

"""
Something to be aware of in machine learning is the performance-speed tradeoff.
Generally, you get better performance out of a larger, more complex model 
    (like we did with model_2).
However, this performance increase often comes at a sacrifice of training 
    speed and inference speed.
"""

# Visualize our model results
compare_results.set_index("model_name")["model_acc"].plot(kind="barh")
plt.xlabel("accuracy (%)")
plt.ylabel("model")
plt.show(block=True)

"""
let's further evaluate our best performing model, model_2.

To do so, let's create a function make_predictions() where 
    we can pass the model and some data for it to predict on.
"""

def make_predictions(model: torch.nn.Module, data: list, device: torch.device = device):
    pred_probs = []
    model.eval()
    with torch.inference_mode():
        for sample in data:
            # Prepare sample 
            sample = torch.unsqueeze(sample, dim=0).to(device) # Add an extra dimension and send sample to device

            # Forward pass (model outpus raw logit)
            pred_logit = model(sample)

            # Get prediction probability (logit -> prediction probability)
            pred_prob = torch.softmax(pred_logit.squeeze(), dim=0) # note: perform softmax on the "logits" dimension, not "batch" dimension 
                                                                   # (in this case we have a batch size of 1, so can perform on dim=0)
            
            pred_probs.append(pred_prob.cpu())

    # Stack the pred_probs to turn list into a tensor
    return torch.stack(pred_probs)

random.seed(42)
test_samples = []
test_labels = []

for sample, label in random.sample(list(test_data), k=9):
    test_samples.append(sample)
    test_labels.append(label)

# View the first test sample shape and label
print(f"\nTest sample image shape: {test_samples[0].shape}\nTest sample label: {test_labels[0]} ({class_names[test_labels[0]]})\n")

pred_probs=make_predictions(model=model_2, 
                            data=test_samples)

# View first two prediction probabilities list
print(f"\nFirst two prediction probabilities: {pred_probs[:2]}\n")

"""
Now we can go from prediction probabilities to prediction labels by 
    taking the torch.argmax() of the output of the torch.softmax() activation function.
"""

pred_classes = pred_probs.argmax(dim=1)
print(f"\nPrediction classes: {pred_classes}\nTest Classes: {test_labels}\n")

# Plot predictions
plt.figure(figsize=(9, 9))
nrows = 3
ncols = 3
for i, sample in enumerate(test_samples):
    # Create a subplot
    plt.subplot(nrows, ncols, i+1)

    # Plot the target image
    plt.imshow(sample.squeeze(), cmap="gray")

    # Find the prediction label (in text form, e.g. "Sandal")
    pred_label = class_names[pred_classes[i]]

    # Get the truth label (in text form, e.g. "T-shirt")
    truth_label = class_names[test_labels[i]]

    # Create the title text of plot
    title_text = f"Pred: {pred_label} | Truth: {truth_label}"

    # Check for equality and change title colour accordingly 
    if pred_label == truth_label:
        plt.title(title_text, fontsize=10, c="g") # green text if correct
    else:
        plt.title(title_text, fontsize=10, c="r") # red text if wrong
    plt.axis(False)

    plt.show(block=True)

"""
There are many different evaluation metrics we can use for classification problems.
One of the most visual is a confusion matrix.
    A confusion matrix shows you where your classification model 
    got confused between predictions and true labels.
To make a confusion matrix, we'll go through three steps:
    Make predictions with our trained model, model_2 (a confusion matrix compares predictions to true labels).
    Make a confusion matrix using torchmetrics.ConfusionMatrix.
    Plot the confusion matrix using mlxtend.plotting.plot_confusion_matrix().
Let's start by making predictions with our trained model.
"""
# 1. Make predictions with trained model
y_preds = []
model_2.eval()
with torch.inference_mode():
    for X, y in tqdm(test_dataloader, desc="Making predictions"):
        # Send data and targets to target device
        X, y = X.to(device), y.to(device)
        # Do the forward pass
        y_logit = model_2(X)
        # Turn predictions from logits -> prediction probabilities -> predictions labels
        y_pred = torch.softmax(y_logit, dim=1).argmax(dim=1) # note: perform softmax on the "logits" dimension, not "batch" dimension 
                                                             # (in this case we have a batch size of 32, so can perform on dim=1)
        y_preds.append(y_pred.cpu())
    # Concatenate list of predictions into a tensor
    y_pred_tensor = torch.cat(y_preds)
    
"""
Now we can create the confusion matrix

First we'll create a torchmetrics.ConfusionMatrix instance 
    telling it how many classes we're dealing with by setting 
    num_classes=len(class_names).
Then we'll create a confusion matrix (in tensor format) by passing 
    our instance our model's predictions (preds=y_pred_tensor) and 
    targets (target=test_data.targets).
Finally we can plot our confusion matrix using the plot_confusion_matrix() 
    function from mlxtend.plotting.
"""

# 2. Setup confusion matrix instance and compare predictions to targets
confmat = ConfusionMatrix(num_classes=len(class_names), task='multiclass')
confmat_tensor = confmat(preds=y_pred_tensor,
                         target=test_data.targets)

#3. Plot the confusion matric
fig, ax = plot_confusion_matrix(
    conf_mat=confmat_tensor.numpy(), # matplotlib likes working with NumPy
    class_names=class_names, # turn the row and column labels into class names
    figsize=(10, 7)
)

"""
We can see our model does fairly well since most of the dark squares are 
    down the diagonal from top left to bottom right (and ideal model will 
    have only values in these squares and 0 everywhere else).
The model gets most "confused" on classes that are similar, 
    for example predicting "Pullover" for images that are actually labelled "Shirt".
        And the same for predicting "Shirt" for classes that are actually labelled "T-shirt/top".

This kind of information is often more helpful than a single accuracy metric because it tells 
    use where a model is getting things wrong.
        It also hints at why the model may be getting certain things wrong.
We can use this kind of information to further inspect our models and data to see how it could be improved.
"""

print(f"\n\n\nSave and load the model\n\n\n")

# Create models directory (if it doesn't exist)
MODEL_PATH = Path("models")
MODEL_PATH.mkdir(parents=True, # create parent directories if needed
                 exist_ok=True # if models directory already exists, don't error
)

# Create model save path
MODEL_NAME = "03_pytorch_computer_vision_model_2.pth"
MODEL_SAVE_PATH = MODEL_PATH / MODEL_NAME

# Save the model state dict
print(f"Saving model to: {MODEL_SAVE_PATH}")
torch.save(obj=model_2.state_dict(), # only saving the state_dict() only saves the learned parameters
           f=MODEL_SAVE_PATH)

"""
Since we're using load_state_dict(), we'll need to create a new instance of 
    FashionMNISTModelV2() with the same input parameters as our saved model state_dict().
"""

# Create a new instance of FashionMNISTModelV2 (the same class as our saved state_dict())
# Note: loading model will error if the shapes here aren't the same as the saved version
loaded_model_2 = FashionMNISTModelV2(input_shape=1,
                                     hidden_units=10,
                                     output_shape=10)

# Load in the saved state_dict()
loaded_model_2.load_state_dict(torch.load(f=MODEL_SAVE_PATH))

# Send model to GPU
loaded_model_2 = loaded_model_2.to(device)

# Evaluate loaded model
torch.manual_seed(42)

loaded_model_2_results = eval_model(
    model=loaded_model_2,
    data_loader=test_dataloader,
    loss_fn=loss_fn,
    accuracy_fn=accuracy_fn
)

print(f"\nLoaded model 2 results: {loaded_model_2_results}\nModel 2 Results: {model_2_results}\n")

"""
We can find out if two tensors are close to each other using torch.isclose() 
    and passing in a tolerance level of closeness via the parameters atol (absolute tolerance) 
    and rtol (relative tolerance).

If our model's results are close, the output of torch.isclose() should be true.
"""
# Check to see if results are close to each other (if they are very far away, there may be an error)
tensorsClose = torch.isclose(torch.tensor(model_2_results["model_loss"]),
                             torch.tensor(loaded_model_2_results["model_loss"]),
                             atol=1e-08, # absolute tolerance
                             rtol=0.0001) # Relative tolerance
print(f"\nTensors close: {tensorsClose}\n")