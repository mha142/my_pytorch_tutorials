import torch 
#import torch.nn as nn # to make the weights and biases a part of the neural network (NN)
from torch import nn
import torch.nn.functional as F # to use to the activatio function 
from torch.optim import SGD # to use Stochastic Gradient Descent to fit NN to the data 
# to drwa nice looking graphs 
import matplotlib.pyplot as plt
import seaborn as sns 

class BasicNN(nn.Module):#creating a new NN means reating a new class #BasicNN will inherit from a pytorch class 
                            #called module
    
    #create and initialize weights w and biases b
    def __init__(self):
        super().__init__()# we call the initialization method of the parent class nn.module
        #make the weight w00 a parameter for the NN
        self.w00 = nn.Parameter(torch.tensor(1.7), requires_grad=False)#initilize w00=1.7, requires_grad means we don't need to optimize w00, optimize means making its value better 
        self.b00 = nn.Parameter(torch.tensor(-0.85), requires_grad=False)
        self.w01 = nn.Parameter(torch.tensor(-40.8), requires_grad=False)

        self.w10 = nn.Parameter(torch.tensor(12.6), requires_grad=False)
        self.b10 = nn.Parameter(torch.tensor(0.0), requires_grad=False)
        self.w11 = nn.Parameter(torch.tensor(2.7), requires_grad=False)

        self.final_bias = nn.Parameter(torch.tensor(-16.), requires_grad=False)

    #forward pass through the NN
    def forward(self, input):#connects the parts of the NN (input, activation function, w, b, output)
        #connecting the input to the activation function 
        input_to_top_relu = input * self.w00 + self.b00
        top_relu_output = F.relu(input_to_top_relu) 
        scaled_top_relu_output = top_relu_output * self.w01

        input_to_bottom_relu = input * self.w10 + self.b10
        bottom_relu_output = F.relu(input_to_bottom_relu) 
        scaled_bottom_relu_output = bottom_relu_output * self.w11

        input_to_final_relu = scaled_top_relu_output + scaled_bottom_relu_output + self.final_bias

        output = F.relu(input_to_final_relu)

        return output
    
    
    
if __name__ == "__main__":
    #check that the NN is working correctly and has no bugs
    #working correctly means that the output results a bent shape that fits the training data 
    #create a sequence of doses to get multiple outputs 
    input_doses = torch.linspace(start=0, end=1, steps=11) #create a tensor with a sequence of 11 values between and including 0 and 1 
    print(input_doses) 

    model = BasicNN()
    
    output_values = model(input_doses)

    sns.set(style="whitegrid")

    sns.lineplot(x=input_doses,
                 y=output_values, 
                 color="green", 
                 linewidth=2.5)
    
    plt.ylabel("Effictivness")
    plt.xlabel("Dose")
    plt.show()
