import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
import mlxtend 
from sklearn.metrics import roc_curve, auc
print(mlxtend.__version__)
assert int(mlxtend.__version__.split(".")[1]) >= 19 # should be version 0.19.0 or higher
from torchmetrics import ConfusionMatrix
from mlxtend.plotting import plot_confusion_matrix

"""
Tutorial from https://www.codegenes.net/blog/multivariate-lstm-pytorch/ 
and https://www.codegenes.net/blog/lstm-for-multple-output-pytorch/

Dataset from http://archive.ics.uci.edu/dataset/357/occupancy+detection

Multivariate LSTM is an extension of the basic LSTM architecture to 
    handle multiple input variables. In a univariate time-series, 
    we have only one variable changing over time. For example, 
    the daily temperature. 
        But in a multivariate time-series, we have multiple variables 
        that are related and changing over time. For instance, in 
        weather forecasting, we might have temperature, humidity, and 
        wind speed as different variables. A multivariate LSTM can 
        learn the relationships between these variables and make 
        predictions based on their combined information.

LSTM is a type of recurrent neural network (RNN) designed to overcome 
    the vanishing gradient problem of traditional RNNs. It has a 
    memory cell and three gates: the input gate, the forget gate, 
    and the output gate.
        Forget Gate: Decides what information should be thrown away 
            from the cell state.
        Input Gate: Decides which values from the input will update 
            the cell state.
        Output Gate: Decides what the next hidden state will be, 
            which is also used for the output of the LSTM.
"""

# Load data from CSV file
data = pd.read_csv(r"/mnt/c/Users/Ziono/OneDrive/Documents/Garden IoT project/Pytorch Tutorial/data/OccupancyDetection/OccupancyDetection.csv")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Extract relevant columns
#"date","Temperature","Humidity","Light","CO2","HumidityRatio","Occupancy"
input_features = data[['Temperature','Humidity','Light','HumidityRatio']].values
target = data['Occupancy'].values

"""
Normalize the input data to a common scale, such as between 0 and 1 
    or with a mean of 0 and a standard deviation of 1. This helps 
    the model converge faster.
"""
scaler = MinMaxScaler(feature_range=(0,1))
input_features = scaler.fit_transform(input_features)

# Split the data into training and testing sets
train_size = int(len(input_features) * 0.8)
train_input = input_features[:train_size]
test_input = input_features[train_size:]
train_target = target[:train_size]
test_target = target[train_size:]


print(f"\nTrain Input: {train_input}\n Test Input: {test_input}\n Train Shape: {train_input.shape}\n")

def create_sequences(input_data, target_data, seq_length):
    inp_seq = []
    tgt_seq = []
    for i in range(len(input_data) - seq_length):
        inp_seq.append(input_data[i:i + seq_length])
        tgt_seq.append(target_data[i + seq_length])
    return torch.tensor(np.array(inp_seq), dtype=torch.float32), torch.tensor(np.array(tgt_seq), dtype=torch.float32)

seq_length = 50
train_inp_seq, train_tgt_seq = create_sequences(train_input, train_target, seq_length)
test_inp_seq, test_tgt_seq = create_sequences(test_input, test_target, seq_length)

print(f"\nTrain Input Sequence: {train_inp_seq}\n, Train Input Sequence Shape: {train_inp_seq.shape}\n")

class MultivariateLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout = 0.2):
        """
        From GeeksForGeeks: The super() function in Python is used to 
            refer to the parent class. When used in conjunction with 
            the __init__() method, it allows the child class to invoke 
            the constructor of its parent class. This is especially 
            useful when you want to add functionality to the child 
            class's constructor without completely overriding the 
            parent class's constructor.
        """
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size=input_size,
                            hidden_size=hidden_size,
                            num_layers=num_layers,
                            batch_first=True)
        self.dropout = nn.Dropout(0.5)
        self.sigmoid = nn.Sigmoid()
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.sigmoid(self.fc(self.dropout(out[:, -1, :])))
        return out
def accuracy_fn(y_true, y_pred):
    correct= torch.eq(y_true, y_pred).sum().item() # torch.eq() calculates where two tensors are equal
    acc = (correct / len(y_pred)) * 100
    return acc

input_size = train_inp_seq.shape[2]
hidden_size = 64 # Prev 8 then 50* then 64 then 128* then 64
num_layers = 2 # Prev 8 then 1* then 1 then 3 then 2
output_size = 1

batch_size = 128  # Adjusted batch size
train_dataset = TensorDataset(train_inp_seq, train_tgt_seq)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_dataset = TensorDataset(test_inp_seq, test_tgt_seq)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

model = MultivariateLSTM(input_size, hidden_size, num_layers, output_size).to(device)

loss_fn = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

train_loss_set = []
test_loss_set = []
total_train_loss = 0
total_test_loss = 0
train_acc = 0
test_acc = 0
# Training
num_epochs = 100
results = {"train_loss": [],
               "avg_train_acc": [],
               "test_loss": [],
               "avg_test_acc": []
    }
for epoch in tqdm(range(num_epochs)):
    total_train_loss = 0
    train_acc = 0
    model.train()
    for batch_X, batch_y in train_loader:
        batch_X, batch_y = batch_X.to(device), batch_y.to(device)
        outputs = model(batch_X)
        train_pred = torch.round(outputs)
        optimizer.zero_grad()
        loss = loss_fn(outputs, batch_y.unsqueeze(1))
        train_acc += accuracy_fn(batch_y, train_pred.argmax(dim=1))
        
        loss.backward()
        optimizer.step()
        total_train_loss += loss.item()
    average_train_loss = total_train_loss / len(train_loader)
    avg_train_acc = train_acc / len(train_loader)
    ### Testing
    
    # Put mdoel in evaulation mode
    model.eval()

    with torch.inference_mode():
        total_test_loss = 0
        test_acc = 0
        for batch_X_test, batch_y_test in test_loader:
            batch_X_test, batch_y_test = batch_X_test.to(device), batch_y_test.to(device)
            test_outputs = model(batch_X_test)
            test_pred = torch.round(test_outputs)

            test_loss = loss_fn(test_outputs, batch_y_test.unsqueeze(dim=1))
            total_test_loss += test_loss.item()
            test_acc += accuracy_fn(batch_y_test, test_pred.argmax(dim=1))
    average_test_loss = total_test_loss / len(test_loader)
    avg_test_acc = test_acc / len(test_loader)
    results['train_loss'].append(average_train_loss)
    results['test_loss'].append(average_test_loss)
    results['avg_train_acc'].append(avg_train_acc)
    results['avg_test_acc'].append(avg_test_acc)
        # print(f"\nTest Loss: {test_loss}\n")

    # Print the test results every 10 epochs
    if epoch % 10 == 0:
        print(f"Epoch: {epoch} | Train Loss: {average_train_loss} | Test Loss: {average_test_loss:.5f}")

x = np.linspace(1, num_epochs, num_epochs)
plt.plot(x,results['train_loss'], label='Train Loss')
plt.plot(x,results['test_loss'], label='Test Loss')
plt.legend()
plt.show(block=True)
model.eval()
with torch.inference_mode():
    test_predictions = []
    for batch_X_pred in test_inp_seq:
        batch_X_pred = batch_X_pred.to(device).unsqueeze(0)
        pred = torch.round(model(batch_X_pred))
        test_predictions.append(torch.Tensor(pred))

# Plot the results
plt.plot(test_target[seq_length:], label='Actual')
plt.plot(np.array(test_predictions).flatten(), label='Predicted')
plt.legend()
plt.show(block=True)

confmat_target_tensor = torch.Tensor(test_target[seq_length:]).unsqueeze(1)

# Setup confusion matrix instance and compare predictions to targets
confmat = ConfusionMatrix(num_classes=2, task='binary')
confmat_tensor = confmat(preds=torch.cat(test_predictions),
                         target=confmat_target_tensor)

#3. Plot the confusion matric
fig, ax = plot_confusion_matrix(
    conf_mat=confmat_tensor.numpy(), # matplotlib likes working with NumPy
    class_names=["0","1"], # turn the row and column labels into class names
    figsize=(10, 7)
)
plt.show(block=True)

test_predictions = np.array(test_predictions).flatten()

"""
From GeeksForGeeks:
The ROC curve stands for Receiver Operating Characteristics Curve and is an evaluation 
    metric for classification tasks and it is a probability curve that plots sensitivity 
    and specificity. So, we can say that the ROC Curve can also be defined as the evaluation 
    metric that plots the sensitivity against the false positive rate. The ROC curve plots 
    two different parameters given below:
        True positive rate
        False positive rate 
The ROC Curve can also defined as a graphical representation that shows the performance or 
    behavior of a classification model at all different threshold levels. The ROC Curve is a 
    tool used for binary classification in machine learning. While learning about the ROC Curve 
    we need to be familiar with the terms specificity and sensitivity.
        Specificity: It is defined as the proportion of negative instances that were predicted 
            correctly as negative values. In other terms, the true negative is also called the 
            specificity. The false positive rate can be found using the specificity by subtracting 
            one from it.
        Sensitivity: The true positive rate is defined as the rate of positive instances that were 
            predicted correctly to be positive. The true positive rate is a synonym for 
            "True positive rate".The sensitivity is also called recall and these terms 
            are often interchangeable. 

An ideal ROC curve would be as close as possible to the upper left corner of the plot, 
    indicating high TPR (correctly identifying true positives) with low FPR 
    (incorrectly identifying false positives). The closer the curve is to the 
    diagonal baseline, the worse the classifier's performance.
The AUC score provides a quantitative measure of the classifier's performance, 
    with a value of 1 indicating perfect classification and a value of 0.5 
    indicating no better than random guessing. 
"""


# Calculate ROC curve
fpr, tpr, thresholds = roc_curve(test_target[seq_length:], test_predictions) 
roc_auc = auc(fpr, tpr)
# Plot the ROC curve
plt.figure()  
plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], 'k--', label='No Skill')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve for Occupancy Detection Classification')
plt.legend()
plt.show()


pred_results = {
    "Target": test_target[seq_length:],
    "Predicitions": test_predictions
}
model_df = pd.DataFrame(results)
print(f"\n{model_df}\n")

pred_results_df = pd.DataFrame(pred_results)
print(f"\n{pred_results_df}\n")



