from sklearn.datasets import make_circles, make_blobs
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pandas as pd
import torch
from torch import nn
from torchmetrics import Accuracy
device = "cuda" if torch.cuda.is_available() else "cpu"

import requests
from pathlib import Path 

# Download helper functions from Learn PyTorch repo (if not already downloaded)
if Path("helper_functions.py").is_file():
  print("helper_functions.py already exists, skipping download")
else:
  print("Downloading helper_functions.py")
  request = requests.get("https://raw.githubusercontent.com/mrdbourke/pytorch-deep-learning/main/helper_functions.py")
  with open("helper_functions.py", "wb") as f:
    f.write(request.content)

from helper_functions import plot_decision_boundary

# Set the hyperparameters for data creation
NUM_CLASSES = 4
NUM_FEATURES = 2
RANDOM_SEED = 42

"""
Architecture of a classification neural network

Binary classification problem deals with classifying something as 
one of two options (e.g. a photo as a cat photo or a dog photo) 
where as a multi-class classification problem deals with classifying 
something from a list of more than two options (e.g. classifying a 
photo as a cat a dog or a chicken).

Structure:
Hyperparameter
    Binary classification
    Multiclass classification

Input layer shape (in_features) 	
    Same as number of features (e.g. 5 for age, sex, height, weight, 
        smoking status in heart disease prediction) 	
    Same as binary classification

Hidden layer(s) 	
    Problem specific, minimum = 1, maximum = unlimited 	
    Same as binary classification

Neurons per hidden layer 	
    Problem specific, generally 10 to 512 	
    Same as binary classification

Output layer shape (out_features) 	
    1 (one class or the other) 	
    1 per class (e.g. 3 for food, 
        person or dog photo)

Hidden layer activation 	
    Usually ReLU (rectified linear unit) but can be many others 	
    Same as binary classification

Output activation 	
    Sigmoid (torch.sigmoid in PyTorch) 	
    Softmax (torch.softmax in PyTorch)

Loss function 	
    Binary crossentropy (torch.nn.BCELoss in PyTorch) 	
    Cross entropy (torch.nn.CrossEntropyLoss in PyTorch)

Optimizer 	
    SGD (stochastic gradient descent), 
        Adam (see torch.optim for more options) 	
    Same as binary classification
"""

print("\n\n\n====Make Classification data===\n\n\n")
# Make 1000 samples
n_samples = 1000

# Create circles
X, y = make_circles(n_samples, noise = 0.03, # small amount of noise to the dots
                    random_state=42) # keep random state to get same values

print(f"First 5 X Features:\n{X[:5]}")
print(f"\nFirst 5 y Features:\n{y[:5]}\n")


# Make DataFrame of circle data
circles = pd.DataFrame({"X1": X[:, 0],
    "X2": X[:, 1],
    "label": y
})
print("\n\n",circles.head(10))


# Check different labels
print("\n\n",circles.label.value_counts())

# Visualize with a plot
plt.scatter(x=X[:, 0], 
            y=X[:, 1], 
            c=y, 
            cmap=plt.cm.RdYlBu);
plt.show(block=True)

# Turn data into tensors
X = torch.from_numpy(X).type(torch.float)
y = torch.from_numpy(y).type(torch.float)

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, # 20% test, 80% train
                                                    random_state=42) # make the random split reproducible

print("\n\n\nMake the Binary classfication Model\n\n\n")


"""
The only major change is what's happening between self.layer_1 and self.layer_2.

self.layer_1 takes 2 input features in_features=2 and produces 5 output features 
out_features=5. This is known as having 5 hidden units or neurons. 
    This layer turns the input data from having 2 features to 5 features.

This allows the model to learn patterns from 5 numbers rather than just 2 numbers, 
potentially leading to better outputs (I say potentially because sometimes it doesn't work.)
    The number of hidden units you can use in neural network layers is a 
    hyperparameter (a value you can set yourself) and there's no set in stone value you have to use.
        Generally more is better but there's also such a thing as too much. The amount you 
        choose will depend on your model type and dataset you're working with.

The only rule with hidden units is that the next layer, in our case, self.layer_2 has to take the 
same in_features as the previous layer out_features.
    That's why self.layer_2 has in_features=5, it takes the out_features=5 from self.layer_1 
    and performs a linear computation on them, turning them into out_features=1 (the same shape as y).
"""

class CircleModelV0(nn.Module):
    def __init__(self):
        super().__init__()
        # 2. Create 2 nn.Linear layers of handling X and y input and output shapes
        self.layer_1 = nn.Linear(in_features=2, out_features=5) # takes in two features (x), produces 5
        self.layer_2 = nn.Linear(in_features=5, out_features=1) # takes in 5 features, produces 1 (y)
    
    # 3. Define a forward method containing the forward pass computation
    def forward(self, x):
        # Return the output of layer_2, a single feature, the same shape as y
        return self.layer_2(self.layer_1(x)) # computation goes through layer_1 first then the output of layer_1
                                             # goes through layer_2

# 4. Create an instance of the model and send it to target device
model_0 = CircleModelV0().to(device)

print("\n\n",model_0)

# Replicate CircleModelV0 with nn.Sequential
"""
nn.Sequential performs a forward pass computation of 
the input data through the layers in the order they appear.

nn.Sequential is fantastic for straight-forward computations, 
however, as the namespace says, it always runs in sequential order.
    So if you'd like something else to happen (rather than just 
    straight-forward sequential computation) you'll want to define 
    your own custom nn.Module subclass.
"""
model_0 = nn.Sequential(
    nn.Linear(in_features=2, out_features=5),
    nn.Linear(in_features=5, out_features=1)
).to(device)

print("\n\n",model_0, "\n\n")

# make predicitons with the model
untrained_preds = model_0(X_test.to(device))
print(f"Length of predictions: {len(untrained_preds)}, Shape: {untrained_preds.shape}")
print(f"Length of test samples: {len(y_test)}, Shape: {y_test.shape}")
print(f"\nFirst 10 predictions (Untrained):\n{untrained_preds[:10]}")
print(f"\nFirst 10 test labels:\n{y_test[:10]}")

"""
Loss functions and Optimizers

Structure:
Loss function/Optimizer
    Problem Type
    Pytorch Code

Stochastic Gradient Descent (SGD) optimizer 	
    Classification, regression, many others. 	
    torch.optim.SGD()

Adam Optimizer 	
    Classification, regression, many others. 	
    torch.optim.Adam()

Binary cross entropy loss 	
    Binary classification 	
    torch.nn.BCELossWithLogits
       Creates a loss function that measures 
       the binary cross entropy between the 
       target (label) and input (features).
    or torch.nn.BCELoss
        This is the same as above except it 
        has a sigmoid layer (nn.Sigmoid) built-in
            more numerically stable than using 
            torch.nn.BCELoss() after a nn.Sigmoid layer.

Cross entropy loss 	
    Multi-class classification 	
    torch.nn.CrossEntropyLoss

Mean absolute error (MAE) or L1 Loss 	
    Regression 	
    torch.nn.L1Loss

Mean squared error (MSE) or L2 Loss 	
    Regression 	
    torch.nn.MSELoss 
"""

# Create a loss function
# loss_fn = nn.BCE.LOSS() # BCELoss = no sigmoid built-in
loss_fn = nn.BCEWithLogitsLoss() # sigmoid built-in

# Create an optimizer
optimizer = torch.optim.SGD(params=model_0.parameters(), lr=0.1)

# Calculate accuracy (a classificaition and evaluation metric)
def accuracy_fn(y_true, y_pred):
    correct= torch.eq(y_true, y_pred).sum().item() # torch.eq() calculates where two tensors are equal
    acc = (correct / len(y_pred)) * 100
    return acc

# View the first 5 outputs of the forward pass on the test data
"""
These outputs come from the forward method implementing two layers of nn.Linear()

The raw outputs (unmodified) of this equation and in turn, the raw outputs of our
model are often referred to as logits.
""" 
y_logits = model_0(X_test.to(device))[:5] # untrained model so these outputs are random
print(f"\nLogits: {y_logits}\n")
"""
However, these numbers are hard to interpret.
    We'd like some numbers that are comparable to our truth labels.
        To get our model's raw outputs (logits) into such a form, 
        we can use the sigmoid activation function.
            From GeeksForGeeks: "An activation function is applied to the
              weighted sum of inputs before producing the final output of 
              a neuron. It introduces non-linearity, allowing the network 
              to learn complex patterns... Real-world data is rarely linearly 
              separable. Non-linear functions allow neural networks to form 
              curved decision boundaries, making them capable of handling 
              complex patterns (e.g., classifying apples vs. bananas under 
              varying colors and shapes)."
            From GeeksForGeeks: "Sigmoid Activation Function is characterized 
                by 'S' shape. It is mathematically defined as A  = 1/(1+e^(-x))
                This formula ensures a smooth and continuous output that is essential 
                for gradient-based optimization methods.
"""

# Use sigmoid on model logits
"""
The outputs are now in the form of prediciton probibilities, in other words, the values 
are now how much the model thinks the data point belongs to one class or another.
    In our case, since we're dealing with binary classification, 
    our ideal outputs are 0 or 1.

If y_pred_probs >= 0.5, y=1 (class 1)
If y_pred_probs < 0.5, y=0 (class 0)
"""
y_pred_probs = torch.sigmoid(y_logits)
print(f"\n Prediction Probabilities: {y_pred_probs}\n")

"""
To turn our prediction probabilities into prediction labels, 
we can round the outputs of the sigmoid activation function.
"""

# Find the predicted labels (round the prediction probabilities)
y_preds = torch.round(y_pred_probs)

# Or the Model output to labels conversion in full
y_pred_labels = torch.round(torch.sigmoid(model_0(X_test.to(device))[:5]))

# Check for equality
print("Check for equality:", torch.eq(y_preds.squeeze(), y_pred_labels.squeeze()))

# Get rid of extra dimension
y_preds.squeeze()

torch.manual_seed(42)

# Set the number of epochs
epochs = 100

# Put data to target device
X_train, y_train = X_train.to(device), y_train.to(device)
X_test, y_test = X_test.to(device), y_test.to(device)

print("\nnModel 0 Testing\n")

# Build training and evaluation loop
for epoch in range(epochs):
    # Training
    model_0.train()

    # 1. Forward pass (model output raw logits)
    y_logits = model_0(X_train).squeeze() # squeeze to remove extra 1 dimensions 
                                          # won't work unless model & data are on same device
    y_pred = torch.round(torch.sigmoid(y_logits)) # turn logits -> pred probs -> pred labels

    # 2. Calculate loss/accuracy
    loss = loss_fn(y_logits, # Using nn.BCEWithLogitsLoss works with raw logits, nn.BCELoss would need torch.sigmoid()
                   y_train)
    acc = accuracy_fn(y_true=y_train,
                      y_pred=y_pred)
    
    # 3. Optimizer zero grad
    optimizer.zero_grad()

    # 4. Loss backwards
    loss.backward()

    # 5. Optimizer sstep
    optimizer.step()

    # Testing
    model_0.eval()
    with torch.inference_mode():
        # 1. Forward pass
        test_logits = model_0(X_test).squeeze()
        test_pred = torch.round(torch.sigmoid(test_logits))

        # 2. Calculate loss/accuracy
        test_loss = loss_fn(test_logits, y_test)
        test_acc = accuracy_fn(y_true=y_test, y_pred=test_pred)
    
    
    if epoch % 10 == 0:
        print(f"Epoch: {epoch} | Loss: {loss:.5f}, Accuracy: {acc:.2f}% | Test loss: {test_loss:.5f}, Test acc: {test_acc:.2f}%")

"""
In the above training and testing, the accuracy stays around 50%, we need to check if the model is guessing
"""

# Plot decision boundaries for training and test sets
"""
In the plots we see the model is trying to split the red and blue dots using a straight line
Since the data is circular, a straight line can at best cut down the middle

This shows the model is underfitting (not learning predictive patterns from the data)
"""
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.title("Train (Model 0)")
plot_decision_boundary(model_0, X_train, y_train)
plt.show(block=True)
plt.subplot(1, 2, 2)
plt.title("Test (Model 0)")
plot_decision_boundary(model_0, X_test, y_test)
plt.show(block=True)

print("\n\n\nImproving the Model\n\n\n")

"""
Structure:
Model Improvement technique
    What it does

Add more layers 	
    Each layer potentially increases the learning capabilities 
    of the model with each layer being able to learn some kind 
    of new pattern in the data. 
        More layers are often referred 
        to as making your neural network deeper.

Add more hidden units 	
    Similar to the above, more hidden units per layer means a 
    potential increase in learning capabilities of the model. 
        More hidden units are often referred to as making your 
        neural network wider.

Fitting for longer (more epochs) 	
    Your model might learn more if it had more opportunities 
    to look at the data.

Changing the activation functions 	
    Some data just can't be fit with only straight 
    lines (like what we've seen), using non-linear 
    activation functions can help with this (hint, hint).

Change the learning rate 	
    Less model specific, but still related, the learning 
    rate of the optimizer decides how much a model should 
    change its parameters each step, too much and the model 
    overcorrects, too little and it doesn't learn enough.

Change the loss function 	
    Again, less model specific but still important, different 
    problems require different loss functions. For example, 
    a binary cross entropy loss function won't work with a multi-class 
    classification problem.

Use transfer learning 	
    Take a pretrained model from a problem domain similar to yours and 
    adjust it to your own problem. We cover transfer learning in notebook 06.

We will try to add an extra layer, fit for longer and increase the hidden units to 10
"""

class CircleModelV1(nn.Module):
    def __init__(self):
      super().__init__()
      self.layer_1 = nn.Linear(in_features=2, out_features=10)
      self.layer_2 = nn.Linear(in_features=10, out_features=10) # extra layer
      self.layer_3 = nn.Linear(in_features=10, out_features=1)
    def forward(self, x):
      return self.layer_3(self.layer_2(self.layer_1(x)))

model_1 = CircleModelV1().to(device)
print(f"Model 1: {model_1}")

# Updating the optimizer (loss fn stays the same)
optimizer = torch.optim.SGD(model_1.parameters(), lr=0.1)

# Increase the number of epochs
epochs = 1000

# Put data to target device
X_train, y_train = X_train.to(device), y_train.to(device)
X_test, y_test = X_test.to(device), y_test.to(device)

print("\nnModel 1 Testing\n")

# Build training and evaluation loop
for epoch in range(epochs):
    # Training
    model_1.train()

    # 1. Forward pass (model output raw logits)
    y_logits = model_1(X_train).squeeze() # squeeze to remove extra 1 dimensions 
                                          # won't work unless model & data are on same device
    y_pred = torch.round(torch.sigmoid(y_logits)) # turn logits -> pred probs -> pred labels

    # 2. Calculate loss/accuracy
    loss = loss_fn(y_logits, # Using nn.BCEWithLogitsLoss works with raw logits, nn.BCELoss would need torch.sigmoid()
                   y_train)
    acc = accuracy_fn(y_true=y_train,
                      y_pred=y_pred)
    
    # 3. Optimizer zero grad
    optimizer.zero_grad()

    # 4. Loss backwards
    loss.backward()

    # 5. Optimizer sstep
    optimizer.step()

    # Testing
    model_1.eval()
    with torch.inference_mode():
        # 1. Forward pass
        test_logits = model_1(X_test).squeeze()
        test_pred = torch.round(torch.sigmoid(test_logits))

        # 2. Calculate loss/accuracy
        test_loss = loss_fn(test_logits, y_test)
        test_acc = accuracy_fn(y_true=y_test, y_pred=test_pred)
    
    
    if epoch % 100 == 0:
        print(f"Epoch: {epoch} | Loss: {loss:.5f}, Accuracy: {acc:.2f}% | Test loss: {test_loss:.5f}, Test acc: {test_acc:.2f}%")

plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.title("Train (Model 1)")
plot_decision_boundary(model_1, X_train, y_train)
plt.show(block=True)
plt.subplot(1, 2, 2)
plt.title("Test (Model 1)")
plot_decision_boundary(model_1, X_test, y_test)
plt.show(block=True)

"""
The model is still inaccurate and drawing a straight line, we shoule check to see if
it can model linear data or if the entire model is junk mand can't learn anything
"""

weight = 0.7
bias = 0.3
start = 0
end = 1
step = 0.01

# Create data
X_regression = torch.arange(start, end, step).unsqueeze(dim=1)
y_regression = weight * X_regression + bias # linear regression formula

# Check the data
print(len(X_regression))
X_regression[:5], y_regression[:5]

# Create train and test splits
train_split = int(0.8 * len(X_regression)) # 80% of data used for training set
X_train_regression, y_train_regression = X_regression[:train_split], y_regression[:train_split]
X_test_regression, y_test_regression = X_regression[train_split:], y_regression[train_split:]

# Same architecture as model_1 (but using nn.Sequential)
model_2 = nn.Sequential(
    nn.Linear(in_features=1, out_features=10),
    nn.Linear(in_features=10, out_features=10),
    nn.Linear(in_features=10, out_features=1)
).to(device)

# Loss and optimizer
loss_fn = nn.L1Loss()
optimizer = torch.optim.SGD(model_2.parameters(), lr=0.1)

# Train the model
torch.manual_seed(42)

# Set the number of epochs
epochs = 1000

# Put data to target device
X_train_regression, y_train_regression = X_train_regression.to(device), y_train_regression.to(device)
X_test_regression, y_test_regression = X_test_regression.to(device), y_test_regression.to(device)

print("\nnModel 2 (Linear data) Testing\n")
for epoch in range(epochs):
    ### Training 
    # 1. Forward pass
    y_pred = model_2(X_train_regression)
    
    # 2. Calculate loss (no accuracy since it's a regression problem, not classification)
    loss = loss_fn(y_pred, y_train_regression)

    # 3. Optimizer zero grad
    optimizer.zero_grad()

    # 4. Loss backwards
    loss.backward()

    # 5. Optimizer step
    optimizer.step()

    ### Testing
    model_2.eval()
    with torch.inference_mode():
      # 1. Forward pass
      test_pred = model_2(X_test_regression)
      # 2. Calculate the loss 
      test_loss = loss_fn(test_pred, y_test_regression)

    # Print out what's happening
    if epoch % 100 == 0: 
        print(f"Epoch: {epoch} | Train loss: {loss:.5f}, Test loss: {test_loss:.5f}")

# Turn on evaluation mode
model_2.eval()

# Make predictions (inference)
with torch.inference_mode():
    y_preds = model_2(X_test_regression)

"""
Here the loss decreases and plot shows good precictions so our model can learn

Now that we ave established a baseline, we can investigate why the model fails for 
non linear data.

The neural network currently only uses linear line functions but we use non linear data,
we can try using non linear activation functions by putting a common non linear activation function
ReLU (rectified linear-unit, torch.nn.ReLU()) between the hidden layers in the forward pass
"""



class CircleModelV2(nn.Module):
    def __init__(self):
      super().__init__()
      self.layer_1 = nn.Linear(in_features=2, out_features=10)
      self.layer_2 = nn.Linear(in_features=10, out_features=10)
      self.layer_3 = nn.Linear(in_features=10, out_features=1)
      self.relu = nn.ReLU() # add ReLU activation function
      """
      Can also put sigmoid (self.sigmoid = nn.Sigmoid()) and therfore
      wouldn't need it on the predictions
      """
    def forward(self, x):
      return self.layer_3(self.relu(self.layer_2(self.relu(self.layer_1(x)))))

model_3 = CircleModelV2().to(device)
print(f"Model 3: {model_3}")

"""
Where should I put the non-linear activation functions when constructing a neural network?
    A rule of thumb is to put them in between hidden layers and just after the output layer, 
    however, there is no set in stone option.
"""

# Setup loss and optimizer 
loss_fn = nn.BCEWithLogitsLoss()
optimizer = torch.optim.SGD(model_3.parameters(), lr=0.1)

torch.manual_seed(42)
epochs = 1000

# Put all data on target device
X_train, y_train = X_train.to(device), y_train.to(device)
X_test, y_test = X_test.to(device), y_test.to(device)

print("\n\n\nModel 3 Testing\n\n\n")
for epoch in range(epochs):
    # 1. Forward pass
    y_logits = model_3(X_train).squeeze()
    y_pred = torch.round(torch.sigmoid(y_logits))

    # 2. Calculate loss and accuracy
    loss = loss_fn(y_logits, y_train) # BCEWithLogitsLoss calculates loss with logits
    acc = accuracy_fn(y_true=y_train, y_pred=y_pred)

    # 3. Optimizer zero grad
    optimizer.zero_grad()

    # 4. Loss backward
    loss.backward()

    #5. Optimizer step
    optimizer.step()

    # Testing
    model_3.eval()
    with torch.inference_mode():
      # 1. Forward pass
      test_logits = model_3(X_test).squeeze()
      test_pred = torch.round(torch.sigmoid(test_logits)) # logits -> prediction probabilities -> prediction labels
    # 2. Calculate loss and accuracy
      test_loss = loss_fn(test_logits, y_test)
      test_acc = accuracy_fn(y_true=y_test, y_pred=test_pred)
    
    if epoch % 100 == 0:
       print(f"Epoch: {epoch} | Loss: {loss:.5f}, Accuracy: {acc:.2f}% | Test Loss: {test_loss:.5f}, Test Accuracy: {test_acc:.2f}%")

"""
The accuracy is much better. Lets see how the models compare (linear vs non linear)
"""

# Make predictions
model_3.eval()
with torch.inference_mode():
    y_preds = torch.round(torch.sigmoid(model_3(X_test))).squeeze()
y_preds[:10], y[:10] # want preds in same format as truth labels

# Plot decision boundaries for training and test sets
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.title("Train (Model 1)")
plot_decision_boundary(model_1, X_train, y_train) # model_1 = no non-linearity
plt.show(block=True)
plt.subplot(1, 2, 2)
plt.title("Test (model 3)")
plot_decision_boundary(model_3, X_test, y_test) # model_3 = has non-linearity
plt.show(block=True)

print("\n\n\nReplicating Non-linear activation functions\n\n\n")
"""
We can see how the activation functions ReLU and the sigmoid function effect
outputs by plotting them

We can create a straight line at a tensor of inputs
"""
A = torch.arange(-10, 10, 1, dtype=torch.float32)
print("Straight line of inputs:",A)
plt.plot(A)
plt.show(block=True)

"""
We can now plot the ReLU activation function to see how it influeces the inputs 
"""

def relu(x):
   return torch.maximum(torch.tensor(0), x)
print("ReLU activation function:", relu(A))
plt.plot(relu(A))
plt.show(block=True)

"""
We can do the same for the sigmoid activation function
"""

def sigmoid(x):
   return 1 / (1 + torch.exp(-x))
print("Sigmoid activation function:", sigmoid(A))
plt.plot(sigmoid(A))
plt.show(block=True)

print("\n\n\nMulti Class classification data\n\n\n")
"""
To begin a multi-class classification problem, let's create some multi-class data.

To do so, we can leverage Scikit-Learn's make_blobs() method.

This method will create however many classes (using the centers parameter) we want.

We can also check if this data sets requires non linear layers
"""
# 1. Create multi-class data
X_blob, y_blob = make_blobs(n_samples=1000,
                            n_features=NUM_FEATURES, # X features
                            centers=NUM_CLASSES, # y labels
                            cluster_std=1.5, # give the clusters a little shake
                            random_state=RANDOM_SEED)

# 2. Turn data into tensors
X_blob = torch.from_numpy(X_blob).type(torch.float)
y_blob = torch.from_numpy(y_blob).type(torch.LongTensor)
print(X_blob[:5], y_blob[:5])

# 3. Split into train and test sets 
X_blob_train, X_blob_test, y_blob_train, y_blob_test = train_test_split(X_blob,
                                                                        y_blob,
                                                                        test_size=0.2,
                                                                        random_state=RANDOM_SEED)

# 4. Plot data
plt.figure(figsize=(10, 7))
plt.scatter(X_blob[:, 0], X_blob[:, 1], c=y_blob, cmap=plt.cm.RdYlBu)
plt.show(block=True)

# Build model
class BlobModel(nn.Module):
    def __init__(self, input_features, output_features, hidden_units=8):
      """Initializes all required hyperparameters for a multi-class classification model.

        Args:
            input_features (int): Number of input features to the model.
            out_features (int): Number of output features of the model
              (how many classes there are).
            hidden_units (int): Number of hidden units between layers, default 8.
        """
      super().__init__()
      self.linear_layer_stack = nn.Sequential(
         nn.Linear(in_features=input_features, out_features=hidden_units),
         # nn.ReLU(), # Adds non-linear layers if necessary
         nn.Linear(in_features=hidden_units, out_features=hidden_units),
         # nn.ReLU(), # Adds non-linear layers if necessary
         nn.Linear(in_features=hidden_units, out_features=output_features)
      )

    def forward(self, x):
      return self.linear_layer_stack(x)
    
model_4 = BlobModel(input_features=NUM_FEATURES, output_features=NUM_CLASSES, hidden_units=8).to(device)

print(f"\nModel 4: {model_4}\n")

# Loss fn and optimizer
loss_fn = nn.CrossEntropyLoss() # Good for multi class classification
optimizer = torch.optim.SGD(model_4.parameters(), lr=0.1)

# Perform a single forward pass on the data
print(f"Forward Pass: {model_4(X_blob_train.to(device))[:5]}")

"""
The output of this pass is logits, we need to convert them to prediction labels
    We can do this using the softmax activation function
        It calculates the probability of each predicition class being the actual 
        predicteded class compared to all other clases
"""

# Make predicition logits with model
y_logits = model_4(X_blob_test.to(device))

# Perform softmax calculation on logits across dimension 1 to get prediction possibilities
y_pred_probs = torch.softmax(y_logits, dim=1)
print(f"\nY Logits: {y_logits[:5]}\n")
print(f"\nY Prediciton Probablities: {y_pred_probs[:5]}\n")

"""
It may still look like the outputs of the softmax function are jumbled numbers 
(and they are, since our model hasn't been trained and is predicting using random patterns) 
but there's a very specific thing different about each sample.

After passing the logits through the softmax function, each individual sample now adds to 1 (or very close to).
"""

# Sum the first sample output of the softmax activation function 
print(f"Sum of row 0 of the prediction probabilities: {torch.sum(y_pred_probs[0])}")

"""
These prediction probabilities are essentially saying how much the model thinks the target X
sample (the input) maps to each class.
    Since there's one value for each class in y_pred_probs, the index of the highest 
    value is the class the model thinks the specific data sample most belongs to.   
        We can check which index has the highest value using torch.argmax().
"""
# Which class does the model think is *most* likely at the index 0 sample?
print(f"\nIndex 0 sample: {y_pred_probs[0]}\n")
print(f"The Class the model believes index 0 has the highest probability of being in is: {torch.argmax(y_pred_probs[0])}")

"""
You can see the output of torch.argmax() returns 3, so for the features (X) of the sample 
index 0, the model is predicting that the most likely class value (y) is 3.

Of course, right now this is just random guessing so it's got a 25% chance of being right 
(since there's four classes). But we can improve those chances by training the model.


To summarize the above, a model's raw output is referred to as logits.

For a multi-class classification problem, to turn the logits into prediction probabilities, 
you use the softmax activation function (torch.softmax).
    The index of the value with the highest prediction probability is the class number the
    model thinks is most likely given the input features for that sample (although this is a 
    prediction, it doesn't mean it will be correct).

Now we can train the model
"""  

torch.manual_seed(42)

# Set number of epochs
epochs = 100

# Put data to target device
X_blob_train, y_blob_train = X_blob_train.to(device), y_blob_train.to(device)
X_blob_test, y_blob_test = X_blob_test.to(device), y_blob_test.to(device)

for epoch in range(epochs):
    # Training
    model_4.train()

    # 1. Forward pass
    y_logits = model_4(X_blob_train) # model outputs raw logits
    y_pred = torch.softmax(y_logits, dim=1).argmax(dim=1) # go from logits -> prediction probabilities -> prediction labels

    # 2. Calculate loss and accuracy
    loss = loss_fn(y_logits, y_blob_train)
    acc = accuracy_fn(y_true=y_blob_train,
                        y_pred=y_pred)

    # 3. Optimizer zero grad
    optimizer.zero_grad()

    # 4. Loss backwards
    loss.backward()

    # 5. Optimizer Step
    optimizer.step()

    # Testing
    model_4.eval()
    with torch.inference_mode():
      # 1. Forward pass
      test_logits = model_4(X_blob_test)
      test_pred = torch.softmax(test_logits, dim=1).argmax(dim=1)

      # 2. Calculate test loss and accuracy
      test_loss = loss_fn(test_logits, y_blob_test)
      test_acc = accuracy_fn(y_true=y_blob_test,
                     y_pred=test_pred)
    if epoch % 10 == 0:
        print(f"Epoch: {epoch} | Loss: {loss:.5f}, Acc: {acc:.2f}% | Test Loss: {test_loss:.5f}, Test Acc: {test_acc:.2f}%")

    
# Make predictions
model_4.eval()
with torch.inference_mode():
    y_logits = model_4(X_blob_test)

# View the first 10 predictions
print(f"First 10 predicitions: {y_logits[:10]}")

# Turn predicted logits in prediction probabilities
y_pred_probs = torch.softmax(y_logits, dim=1)

# Turn prediction probabilities into prediction labels
y_preds = y_pred_probs.argmax(dim=1)

# Compare first 10 model preds and test labels
print(f"Predictions: {y_preds[:10]}\nLabels: {y_blob_test[:10]}")
print(f"Test accuracy: {accuracy_fn(y_true=y_blob_test, y_pred=y_preds)}%")


# Plot model 4 train and test result boundaries
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.title("Train (Model 4)")
plot_decision_boundary(model_4, X_blob_train, y_blob_train)
plt.show(block=True)
plt.subplot(1, 2, 2)
plt.title("Test (Model 4)")
plot_decision_boundary(model_4, X_blob_test, y_blob_test)
plt.show(block=True)

"""
We can see that the non linear activation function was not necessary as a series of 
linear functions was sufficient in accurately defining the boundaries between classes
    this was not possible for the circle data but in this case it was
"""

"""
More Classification evaluation metrics

Structure:
Metric Name
    Definition
    Code

Accuracy 	
    Out of 100 predictions, how many does your model get correct? 
        E.g. 95% accuracy means it gets 95/100 predictions correct. 	
    torchmetrics.Accuracy() or sklearn.metrics.accuracy_score()
Precision 	
    Proportion of true positives over total number of samples. Higher 
        precision leads to less false positives (model predicts 1 when 
        it should've been 0). 	
    torchmetrics.Precision() or sklearn.metrics.precision_score()
Recall 	
    Proportion of true positives over total number of true positives 
        and false negatives (model predicts 0 when it should've been 1). 
        Higher recall leads to less false negatives. 	
    torchmetrics.Recall() or sklearn.metrics.recall_score()
F1-score 	
    Combines precision and recall into one metric. 1 is best, 0 is worst. 	
    torchmetrics.F1Score() or sklearn.metrics.f1_score()
Confusion matrix 	
    Compares the predicted values with the true values in a tabular way, 
        if 100% correct, all values in the matrix will be top left to bottom right 
        (diagonal line). 	
    torchmetrics.ConfusionMatrix or sklearn.metrics.plot_confusion_matrix()
Classification report 	
    Collection of some of the main classification metrics such as precision, 
        recall and f1-score. 	
    sklearn.metrics.classification_report()

Scikit-Learn (a popular and world-class machine learning library) has many 
    implementations of the above metrics and you're looking for a PyTorch-like 
    version, check out TorchMetrics, especially the TorchMetrics classification 
    section.

Let's try the torchmetrics.Accuracy metric out.
"""

# Setup metric and make sure it's on the target device
torchmetrics_accuracy = Accuracy(task='multiclass', num_classes=4).to(device)

# Calculate accuracy
print(f"\nTorch Metrics Accuracy: {torchmetrics_accuracy(y_preds, y_blob_test)}\n")