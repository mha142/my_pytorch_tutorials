#1- desgin model (input, output size, forward pass)
#2- construct loss and optimizer 
#3- training loop:
#   -forward pass: compute predection and loss 
#   -backward pass: gradients
#   -update weights



import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt 
import numpy as np 
from sklearn import datasets
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

import models.linear_model_file as linearModel
import models.mobilenet_v1 as model_mobilenet
import models.cascaded_regression as cr

from torch.utils.tensorboard import SummaryWriter
#to run tensorboard
#tensorboard --logdir=runs/linear


def train_synthesized_data():
    #0- prepare datas
    #Generate synthetic data OR load data

  
    # Example usage
    input_size = 10  # Assume 10 input features
    output_size = 2  # For example, predicting 2D coordinates

    X = torch.randn(32, input_size)  # 32 samples, each with 10 features
    y = torch.randn(32, output_size) 

    #define linear regression model        
    #it was defined outside the function 

    #instantiate the model
    #model = linearModel.LinearegressionModel(input_size, output_size)
    model= cr.CascadedRegressor(input_size, output_size, num_stages=3)
    
    #2- loss function, and optimizer 
    criterion = nn.MSELoss()
    #optimizer = optim.SGD(model.parameters(), lr= 0.01)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)


    #training loop 
    num_epochs = 100
    writer = SummaryWriter('./runs/cascaded')
    for epoch in range(num_epochs):
        step_loss = []
        model.train()
        #calling the model
        #forward pass 
        y_pred = model(X)

        #compute_loss 
        loss = criterion(y_pred, y)
        step_loss.append(loss.item())
        #backward pass
        optimizer.zero_grad()
        loss.backward()
        #update the weights by doing optimization 
        optimizer.step()

        #empty the gradients 
        optimizer.zero_grad()

        #print the progress every 100 epochs 
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')
        loss_per_epoch = np.array(step_loss).mean()
        writer.add_scalar('Loss per epoch', loss_per_epoch, epoch)


    #plot results
    predicted = model(X).detach().numpy()
    #plt.plot(x_numpy, y_numpy, 'ro', label='Original data')
    #plt.plot(x_numpy, predicted, 'b', label= 'Fitted line')

    ## plt.scatter(X.numpy(), y.numpy(), label='Original data')
    ## plt.plot(X.numpy(), y_pred.detach().numpy(), 'r-', label= 'Fitted line')
    # plt.legend()#this to allow the labels to show up in the picture 
    # plt.xlabel('X')#this will put a name to the axes 
    # plt.ylabel('y')#this will put a name to the axes 
    # plt.title('Cascaded Linear Regression using Pytorch')
    # plt.show()
    #plt.savefig('synthesized_data_training.png')
    
    #test the model 
    #item = 1
   
    #prediction  = model(torch.tensor(item)) 
    #print(f'prediction of {item} is {prediction}')
          


    #save the model 
    #torch.save(model.state_dict(), './liner_regression_for_syntheized_data')

    #reload the model
    #the_model = TheModelClass(*args, **kwargs)
    #the_model.load_state_dict(torch.load(PATH))
    #test the model 

if __name__ == "__main__":
    #print("Linear Model Class in another file")
    train_synthesized_data()
