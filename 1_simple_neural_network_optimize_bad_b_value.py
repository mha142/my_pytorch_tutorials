import torch 
#import torch.nn as nn # to make the weights and biases a part of the neural network (NN)
from torch import nn
import torch.nn.functional as F # to use to the activatio function 
from torch.optim import SGD # to use Stochastic Gradient Descent to fit NN to the data 
# to drwa nice looking graphs 
import matplotlib.pyplot as plt
import seaborn as sns 

class BasicNN_train(nn.Module):#creating a new NN means reating a new class #BasicNN will inherit from a pytorch class 
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

        self.final_bias = nn.Parameter(torch.tensor(0.), requires_grad=True)# requires_grad=True meanse that this prameter (final_bias) should be optimized 

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

    model = BasicNN_train()
    
    output_values = model(input_doses)

    sns.set(style="whitegrid")

    sns.lineplot(x=input_doses,
                 y=output_values.detach(), #output_values has the gradients and the final output values, and by using detach when are only using the final output values and ignoring the gradients
                 color="green", 
                 linewidth=2.5)
    
    plt.ylabel("Effictivness")
    plt.xlabel("Dose")
    plt.show()


    inputs = torch.tensor([0., 0.5, 1.])
    labels = torch.tensor([0., 1., 0.])

    #lr means the learning rate
    optimizer = SGD(model.parameters(), lr=0.1)#model.parameters() will optimize every parameter that we set requires_grad=True

    print("Final bias, before optimization: " + str(model.final_bias.data) + "\n")

    # each time our optimization code sees all of the training data is called an EPOCH 
    #so everytime we run all 3 points from our training data through the model, we call that an epoch 
    #we run all 3 data points from the training data through the model up to 100 times 
    for epoch in range (100):
        total_loss = 0 # a measure of how well the model fits the data 

        for iteration in range(len(inputs)): #this nested for loop runs each data point from the training data through the model and calculates the total loss
            input_i = inputs[iteration] # dose 
            label_i = labels[iteration] # effectivness

            output_i = model(input_i)

            #calculating the squared residual 
            loss = (output_i - label_i) ** 2 #calculate the residual between the predectide value (output_i) and the original known value (label_i)
            #or use MSELoss() or CrossEntropyLoss() from Pytorch

            loss.backward()# calculate the derivetive of the loss function with respect to the paramter or parametere we want to optimize
                                #in this code we calculating the derivetive of the squared residuals with repect to b_final

            total_loss += float(loss) #add the squared residual to the total loss, to keep track how well the model fits all the data 

        if(total_loss < 0.0001): #if the total_loss is very small this means that the model fits the training data really well and we can stop training
            print("Num steps: " + str(epoch))
            break #print the number of epochs we've gone through so far and stop training becuase the error is small enough

        optimizer.step() #if the error is not small, then we take a small step towards a better value for b_final 
        optimizer.zero_grad() #zero out the derivarives that we're storing in model 

        #after finishing the 100 loops (epochs)
        print("Step: " + str(epoch) + " Final Bias:" + str(model.final_bias.data) + "\n") #print the current epoch, and the current value of final_bias, to see how fina_bias changes each time at the end of each loop 
        

    #verify that the optimized model fits the training data by graphing it with this code 
        
    output_values = model(input_doses)

    sns.set(style="whitegrid")

    sns.lineplot(x=input_doses,
                 y=output_values.detach(), #output_values has the gradients and the final output values, and by using detach when are only using the final output values and ignoring the gradients
                 color="green", 
                 linewidth=2.5)
    
    plt.ylabel("Effictivness")
    plt.xlabel("Dose")
    plt.show()