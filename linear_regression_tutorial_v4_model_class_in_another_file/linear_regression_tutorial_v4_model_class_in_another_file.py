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

import linear_regression_model_class_for_v4_tutorial as linearModel


        


def train_synthesized_data():
    #0- prepare datas
    #Generate synthetic data OR load data
    x_numpy, y_numpy = datasets.make_regression(n_samples=100, n_features=1, noise=20, random_state=1) 

    X = torch.from_numpy(x_numpy.astype(np.float32))
    y = torch.from_numpy(y_numpy.astype(np.float32))
    y =y.view(y.shape[0], 1)#veiw does the same thing as np.reshape

    n_samples, n_features = X.shape
    
    #1- design the model 
   
    input_size = n_features
    output_size = 1
    #model = nn.Linear(input_size, output_size) #instantiate the model

    #define linear regression model        
    #it was defined outside the function 

    #instantiate the model
    model = linearModel.LinearegressionModel(input_size, output_size)
    #2- loss function, and optimizer 
    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr= 0.01)

    #training loop 
    num_epochs = 50

    for epoch in range(num_epochs):
        #calling the model
        #forward pass 
        y_pred = model(X)

        #compute_loss 
        loss = criterion(y_pred, y)

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
        
    #plot results
    predicted = model(X).detach().numpy()
    plt.plot(x_numpy, y_numpy, 'ro', label='Original data')
    plt.plot(x_numpy, predicted, 'b', label= 'Fitted line')
    

    # plt.scatter(X.numpy(), y.numpy(), label='Original data')
    # plt.plot(X.numpy(), y_pred.detach().numpy(), 'r-', label= 'Fitted line')
    plt.legend()#this to allow the labels to show up in the picture 
    plt.xlabel('X')#this will put a name to the axes 
    plt.ylabel('y')#this will put a name to the axes 
    plt.title('Linear Regression using Pytorch')
    plt.show()
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
    print("Linear Model Class in another file")
    train_synthesized_data()