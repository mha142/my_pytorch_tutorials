import torch
import torch.nn as nn

class LinearegressionModel(nn.Module):
    def __init__(self, input_size, output_size):
        super(LinearegressionModel, self).__init__()
        self.linear = nn.Linear(input_size, output_size) # one input feature and one output feature 

    def forward(self, x):
        return self.linear(x)
        