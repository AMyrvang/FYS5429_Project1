import numpy as np
import pandas as pd
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
#from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset


torch.manual_seed(2024)

class simple_RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(simple_RNN, self).__init__()
        self.hidden_size = hidden_size
        self.in_to_hidden = nn.Linear(input_size + hidden_size, hidden_size)
        self.in_to_output = nn.Linear(input_size + hidden_size, output_size)


    def forward(self, x, hidden_state):
        combined = torch.cat((x, hidden_state), 1)
        hidden = torch.sigmoid(self.in_to_hidden(combined))
        output = self.in_to_output(combined)
        return output, hidden
    
    def init_hidden(self):
        return nn.init.kaiming_uniform_(torch.empty(1, self.hidden_size))
    
hidden_size = 0
learning_rate = 0 

model = simple_RNN()
criterion = nn.CrossEntropyLoss()
optimizer = "Adam?"

def train():
    pass
def predict_temp():
    pass











"""
#Making a simple RNN with pytorch. 
hidden_size = 128
num_classes = 10
num_epochs = 2
batch_size = 100
learning_rate = 0.001

input_size = 28
sequence_length = 28
num_layers = 2

# tanh = (exp(x) - exp(-x))/(exp(x) + exp(-x)) (hyperbolik tangent)(kinda looks like a sigmoid function)

class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNN, self).__init__()

        self.hidden_size = hidden_size

        self.i2h = nn.Linear(input_size + hidden_size, hidden_size)
        self.h2o = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
        combined = torch.cat((input, hidden), 1)
        hidden = self.i2h(combined)
        output = self.h2o(hidden)
        output = self.softmax(output)
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, self.hidden_size)
"""



