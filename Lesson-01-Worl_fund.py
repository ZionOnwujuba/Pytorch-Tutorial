import torch
from torch import nn
import matplotlib.pyplot as plt
from pathlib import Path
from pprint import pprint

# Setup device agnostic code
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}\n\n")

# known params
weight = 0.7
bias = 0.3

# create data
start = 0
end = 1
step = 0.02
X = torch.arange(start, end, step).unsqueeze(dim=1)
y = weight * X + bias

print(X[:10], y[:10]) 

# Create train/test split
train_split = int(0.8 * len(X)) # 80% of data used for training set, 20% for testing 
X_train, y_train = X[:train_split], y[:train_split]
X_test, y_test = X[train_split:], y[train_split:]

print("\n\n\n===Data and Model Creation===\n\n\n")

# print("\n", len(X_train), len(y_train), len(X_test), len(y_test))

def plot_predictions(plot_title, train_data=X_train, train_labels=y_train, 
                     test_data=X_test, test_labels=y_test, predictions=None):
    # Plots training & test data and compares the predictions

    plt.figure(figsize=(10,7))

    # Plot training data in blue
    plt.scatter(train_data, train_labels, c="b", s=4, label="Training data")

    # Plot test data in blue
    plt.scatter(test_data, test_labels, c="g", s=4, label="Testing data")

    if predictions is not None:
        # Plot predictions on test data in red
        plt.scatter(test_data, predictions, c="r", s=4, label="Predictions")
    
    plt.legend(prop={"size": 14})
    plt.title(plot_title)

    plt.show(block=True)

plot_predictions("Plot Predictions")

class LinearRegressionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.weights = nn.Parameter(torch.rand(1, # <- start with a random weight that will be adjusted as the model learns
                                               dtype=torch.float),  # default float32
                                               requires_grad=True) # <- can we update this value with gradient descent
        
        self.bias = nn.Parameter(torch.rand(1, # <- start with a random bias that will be adjusted as the model learns
                                               dtype=torch.float),  # default float32
                                               requires_grad=True) # <- can we update this value with gradient descent
        
    def forward(self, x: torch.Tensor) -> torch.Tensor: # "x" is the input data (training/testing features)
        return self.weights * x + self.bias # <- linear regression formula

# manual seed since nn.Paramenter are randomly initialized   
torch.manual_seed(42)

# Create an instance of the model
model_0 = LinearRegressionModel()

# check the nn.Parameters within the nn.Module subclass
print(list(model_0.parameters()))
print("\n", model_0.state_dict())

# Make predictions with model no training
with torch.inference_mode(): 
    """
    As the name suggests, torch.inference_mode() is used when using a model for inference (making predictions).

    torch.inference_mode() turns off a bunch of things (like gradient tracking, which is necessary for training 
    but not for inference) to make forward-passes (data going through the forward() method) faster.
    """
    y_preds = model_0(X_test)

# Check the predictions (random guessing)
print(f"Number of testing samples: {len(X_test)}") 
print(f"Number of predictions made: {len(y_preds)}")
print(f"Predicted values:\n{y_preds}")


# plot_predictions("Random guess Predictions", predictions=y_preds)

print("\n\n\n===Model Testing and Training===\n\n\n")

# Create loss function
"""
Loss func: Measures how wrong your model's predictions (e.g. y_preds) are compared to the truth labels (e.g. y_test). Lower the better.

Mean absolute error (MAE, in PyTorch: torch.nn.L1Loss) measures the absolute difference between two points (predictions and 
labels) and then takes the mean across all examples.
"""

loss_fn = nn.L1Loss()

# Create Optimizer

"""
Optimizer: Tells your model how to update its internal parameters to best lower the loss.

Stochastic gradient descent: SGD, torch.optim.SGD(params, lr) where:

    params is the target model parameters you'd like to optimize (e.g. the weights and bias values we randomly set before).
    lr is the learning rate you'd like the optimizer to update the parameters at, higher means the optimizer will try larger 
    updates (these can sometimes be too large and the optimizer will fail to work), lower means the optimizer will try smaller 
    updates (these can sometimes be too small and the optimizer will take too long to find the ideal values). 
    
        The learning rate is considered a hyperparameter (because it's set by a machine learning engineer). 
        Common starting values for the learning rate are 0.01, 0.001, 0.0001, however, these can also be adjusted over time 
        (this is called learning rate scheduling).

"""
optimizer = torch.optim.SGD(params=model_0.parameters(), # parameters of target model to optimize
                            lr=0.01) # learning rate (how much the optimizer should change params)

"""
Training Loop Steps

Forward Pass: The model goes through all of the training data once, performing its forward() function calculations

Calculate the loss: The model's outputs (predictions) are compared to the ground truth and evaluated to see how wrong they are.

Zero gradients: The optimizers gradients are set to zero (they are accumulated by default) so they can be recalculated for
                the specific training step.


Perform backpropagation on the loss: Computes the gradient of the loss with respect for every model parameter to be updated 
                                    (each parameter with requires_grad=True). This is known as backpropagation, hence "backwards".

Update the optimizer (gradient descent): Update the parameters with requires_grad=True with respect to the loss gradients in order 
                                        to improve them.

<---->

Testing Loop Steps

Forward Pass: The model goes through all of the training data once, performing its forward() function calculations

Calculate the loss: The model's outputs (predictions) are compared to the ground truth and evaluated to see how wrong they are.

Calulate evaluation metrics (optional): Alongside the loss value you may want to calculate other evaluation metrics such as 
                                        accuracy on the test set.

~ Notice the testing loop doesn't contain performing backpropagation (loss.backward()) or stepping the optimizer (optimizer.step()), 
this is because no parameters in the model are being changed during testing, they've already been calculated. For testing, we're only 
interested in the output of the forward pass through the model. ~
"""

torch.manual_seed(42)

epochs = 100 # epoch count (# of times the model will pass over the training data)

# Create empty loss lists to track values
train_loss_values = []
test_loss_values = []
epoch_count = []

for epoch in range(epochs):
    ### Training

    # Put model in training mode (default state of a model)
    model_0.train()

    # 1. Forward pass on train data using forward() method inside
    y_pred = model_0(X_train)

    # 2. Calculate the loss 
    loss = loss_fn(y_pred, y_train)

    # 3. Zero grad od the optimizer
    optimizer.zero_grad()

    #4. Loss backwards
    loss.backward()

    #5. Progress the optimizer
    optimizer.step()

    ### Testing
    
    # Put mdoel in evaulation mode
    model_0.eval()

    with torch.inference_mode():
        #1. Forward pass on test data
        test_pred = model_0(X_test)

        #2. Calcualte loss on test data
        test_loss = loss_fn(test_pred, y_test.type(torch.float))

        # Print the test results every 10 epochs
        if epoch % 10 == 0:
                epoch_count.append(epoch)
                train_loss_values.append(loss.detach().numpy())
                test_loss_values.append(test_loss.detach().numpy())
                print(f"Epoch: {epoch} | MAE Train Loss: {loss} | MAE Test Loss: {test_loss}")

def plot_traintest_loss(epoch_count=epoch_count, train_loss_values=train_loss_values, test_loss_values=test_loss_values):
    plt.plot(epoch_count, train_loss_values, label="Train loss")
    plt.plot(epoch_count, test_loss_values, label="Test loss")
    plt.title("Training and test loss curves")
    plt.ylabel("Loss")
    plt.xlabel("Epochs")
    plt.legend();
    plt.show(block=True)

plot_traintest_loss()

print("\n\n\n===Inference===\n\n\n")

"""
There are three things to remember when making predictions (also called performing inference) with a PyTorch model:

    Set the model in evaluation mode (model.eval()).
    Make the predictions using the inference mode context manager (with torch.inference_mode(): ...).
    All predictions should be made with objects on the same device (e.g. data and model on GPU only or 
    data and model on CPU only), ensuring that you won't run into cross-device errors.

"""

# 1. Set the model in evaluation mode
model_0.eval()

#2. Setup the inference mode context manager
with torch.inference_mode():
    # 3. Make sure the calculations are done with the model and data on the same device
    # in our case, we haven't setup device-agnostic code yet so our data and model are
    # on the CPU by default.
    # model_0.to(device)
    # X_test = X_test.to(device)
    y_preds = model_0(X_test)
print("\nPredictions:\n", y_preds)
print("\nTesting Data:\n", y_test)
plot_predictions("Model 0 Predictions", predictions=y_preds)

print("\n\n\n===Saving and Loading the model===\n\n\n")

"""
Methods to save and load models

torch.save: Saves a serialized object to disk using Python's 
            pickle utility. Models, tensors and various other 
            Python objects like dictionaries can be saved using 
            torch.save. 

torch.load: Uses pickle's unpickling features to deserialize 
            and load pickled Python object files (like models, 
            tensors or dictionaries) into memory. You can also 
            set which device to load the object to (CPU, GPU etc).

torch.nn.Module.load_state_dict: Loads a model's parameter dictionary 
                                (model.state_dict()) using a saved 
                                state_dict() object.

The recommended way for saving and loading a model for inference 
(making predictions) is by saving and loading a model's state_dict().

Let's see how we can do that in a few steps:

    1. We'll create a directory for saving models to called models 
        using Python's pathlib module.
    2. We'll create a file path to save the model to.
    3. We'll call torch.save(obj, f) where obj is the target model's
         state_dict() and f is the filename of where to save the model.

Saving the entire model rather than just the state_dict() is more intuitive, 
however:

    The disadvantage of this approach (saving the whole model) is that the 
    serialized data is bound to the specific classes and the exact directory structure 
    used when the model is saved.

    Because of this, your code can break in various ways when used in other projects 
    or after refactors.

"""

# 1. Create models directory
MODEL_PATH = Path("models")
MODEL_PATH.mkdir(parents=True, exist_ok=True)

MODEL_NAME = "01_pytorch_workflow_model_0.pth"
MODEL_SAVE_PATH = MODEL_PATH / MODEL_NAME

# 3. Save the model state dict
print(f"Saving model to: {MODEL_SAVE_PATH}")
torch.save(obj=model_0.state_dict(), # only saving the state_dict() only saves the modes learned parameters
           f=MODEL_SAVE_PATH)

# Instantiate a new instante of our model (this will be instantiated with random weights)
loaded_model_0 = LinearRegressionModel()

# Load the state_dict of our saved model (this will update the new instance of our model with trained weights)
loaded_model_0.load_state_dict(torch.load(f=MODEL_SAVE_PATH))

# 1. Put the loaded model into evaluation mode
loaded_model_0.eval()

# 2. Use the inference mode context manager to make predictions
with torch.inference_mode():
    loaded_model_preds = loaded_model_0(X_test) # perform a forward pass on the test data with the loaded model

# Compare previous model predictions with loaded model predictions (these should be the same)
print("Compare model 0 with loaded model 0:", y_preds == loaded_model_preds)


print("\n\n\n===Model V2===\n\n\n")

"""
We'll create the same style of model as before except this time, instead of defining the weight 
and bias parameters of our model manually using nn.Parameter(), we'll use nn.Linear(in_features, out_features) 
to do it for us.

Where in_features is the number of dimensions your input data has and out_features is the number of dimensions 
you'd like it to be output to.

In our case, both of these are 1 since our data has 1 input feature (X) per label (y).
"""
class LinearRegressionModelV2(nn.Module):
    def __init__(self):
        super().__init__()
        # Use nn.Linear() for creating the model parameters
        self.linear_layer = nn.Linear(in_features=1,
                                      out_features=1)
        
    # Define the forward computation (input data x flows through nn.Linear())
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear_layer(x)

torch.manual_seed(42)
model_1 = LinearRegressionModelV2()
print(model_1, model_1.state_dict())

# Set model to GPU if avaliable otherwise default to CPU
model_1.to(device) # device var was set to cuda if avaliable
next(model_1.parameters()).device

# Using the original loss function, create new optimizer
optimizer = torch.optim.SGD(params=model_1.parameters(), # optimize newly created model's params
                            lr=0.01)

torch.manual_seed(42)

# Set the number of epochs 
epochs = 1000 

# Put data on the available device
# Without this, error will happen (not all model/data on device)
X_train = X_train.to(device)
X_test = X_test.to(device)
y_train = y_train.to(device)
y_test = y_test.to(device)

for epoch in range(epochs):
    ### Training
    model_1.train() # train mode is on by default after construction

    # 1. Forward pass
    y_pred = model_1(X_train)

    # 2. Calculate loss
    loss = loss_fn(y_pred, y_train)

    # 3. Zero grad optimizer
    optimizer.zero_grad()

    # 4. Loss backward
    loss.backward()

    # 5. Step the optimizer
    optimizer.step()

    ### Testing
    model_1.eval() # put the model in evaluation mode for testing (inference)
    # 1. Forward pass
    with torch.inference_mode():
        test_pred = model_1(X_test)
    
        # 2. Calculate the loss
        test_loss = loss_fn(test_pred, y_test)

    if epoch % 100 == 0:
        print(f"Epoch: {epoch} | Train loss: {loss} | Test loss: {test_loss}")

print("\nThe model learned the following values for weights and bias:")
pprint(model_1.state_dict())
print("\nAnd the original vals for weights and bias are:")
print(f"weights: {weight}, bias: {bias}")

# Turn model into evaluation mode
model_1.eval()

# Make predictions on test data
with torch.inference_mode():
    y_preds = model_1(X_test)
print("Predictions:", y_preds)

"""
Many data science libraries such as pandas, matplotlib and NumPy aren't capable 
of using data that is stored on GPU. So you might run into some issues when trying 
to use a function from one of these libraries with tensor data not stored on the CPU. 

To fix this, you can call .cpu() on your target tensor to return a copy of your 
target tensor on the CPU.
"""
plot_predictions("Model 1 Predictions", predictions=y_preds.cpu())

# Create model 1 save path 
MODEL_NAME = "01_pytorch_workflow_model_1.pth"
MODEL_SAVE_PATH = MODEL_PATH / MODEL_NAME

# Save the model 1 state dict 
print(f"Saving model to: {MODEL_SAVE_PATH}")
torch.save(obj=model_1.state_dict(), # only saving the state_dict() only saves the models learned parameters
           f=MODEL_SAVE_PATH)

# Instantiate a fresh instance of LinearRegressionModelV2
loaded_model_1 = LinearRegressionModelV2()

# Load model state dict 
loaded_model_1.load_state_dict(torch.load(MODEL_SAVE_PATH))

# Put model to target device (if your data is on GPU, model will have to be on GPU to make predictions)
loaded_model_1.to(device)

print(f"Loaded model:\n{loaded_model_1}")
print(f"Model on device:\n{next(loaded_model_1.parameters()).device}")

# Evaluate loaded model
loaded_model_1.eval()
with torch.inference_mode():
    loaded_model_1_preds = loaded_model_1(X_test)
print("Compare model 1 with loaded model 1:", y_preds == loaded_model_1_preds)