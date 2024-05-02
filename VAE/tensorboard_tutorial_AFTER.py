#From YOutube video : https://www.youtube.com/watch?v=VJW9wU-1n18

import torch 
import torch.nn as nn
import torchvision 
import torchvision.transforms as transforms 
import matplotlib.pyplot as plt 
import torch.nn.functional as F # for precision recall curve
import sys
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

from torch.utils.tensorboard import SummaryWriter 
writer = SummaryWriter('./runs/classify_mnist')
 

#device configuration 
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#hyper parameters 
input_size = 784 #28x28
hidden_size = 500
num_classes = 10
num_epochs = 1
batch_size = 64
learning_rate = 0.001 


#MNIST dataset 
train_dataset = torchvision.datasets.MNIST(root='./dataset',
                                           train=True, 
                                           transform=transforms.ToTensor(), 
                                           download=True)

test_dataset = torchvision.datasets.MNIST(root='./dataset', 
                                          train=False, 
                                          transform=transforms.ToTensor())

#Data Loader 
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, 
                                           batch_size=batch_size, 
                                           shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset, 
                                           batch_size=batch_size, 
                                           shuffle=False)

examples = iter(test_loader)
example_data, example_targets = examples.__next__()

for i in range(6):
    plt.subplot(2, 3, i+1)
    plt.imshow(example_data[i][0], cmap='gray')
#plt.show()
img_grid = torchvision.utils.make_grid(example_data)
writer.add_image('mnist_images', img_grid)
writer.close()
#sys.exit()

#Fully connected nueral network with one hidden layer 
class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNet, self).__init__()
        self.input_size = input_size
        self.l1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.l2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out = self.l1(x)
        out = self.relu(out)
        out = self.l2(out)
        #no activation and no softmax at the end 
        return out
    
model = NeuralNet(input_size, hidden_size, num_classes).to(device)

#loss and optimizer 
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)
#writer.add_graph(model, example_data.reshape(-1, 28*28).to(device))
writer.add_graph(model, example_data.reshape(-1, 28*28))
writer.close()
#sys.exit()

#train the model 
n_total_steps = len(train_loader)

running_loss = 0.0

running_correct = 0

for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):#i is the current batch iteration
        #origin shape: [100, 1, 28, 28]
        #resized: [100, 784]
        images = images.reshape(-1, 28*28).to(device)
        labels = labels.to(device)

        #forward pass 
        outputs = model(images)
        loss = criterion(outputs, labels)

        #backward and optimize 
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        running_correct += (predicted == labels).sum().item()
        
        if(i+1) % 100 ==0:#print the loss every 100 step 
            print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{n_total_steps}], Loss:{loss.item():.4f}')
            writer.add_scalar('training loss', running_loss / 100, epoch * n_total_steps + i) #global step = epoch * n_total_steps + i
            writer.add_scalar('accuracy', running_correct / 100, epoch * n_total_steps + i) #global step = epoch * n_total_steps + i
            running_loss = 0.0
            running_correct = 0


#Test the model 
#in test phase we don't need to compute gradients (for memory efficincy)

labels = [] # for precision recall curve 
preds = []  # for precision recall curve

with torch.no_grad():
    n_correct = 0
    n_samples = 0
    for images, labels1 in test_loader:
        images = images.reshape(-1, 28*28).to(device)
        labels1 = labels1.to(device)
        outputs = model(images)
        #max returns (value, index)
        _, predicted = torch.max(outputs.data, 1)
        n_samples += labels1.size(0)
        n_correct += (predicted == labels1).sum().item()
        class_predictions = [F.softmax(output, dim=0) for output in outputs]# for precision recall curve #list comprehension
        preds.append(class_predictions)# for precision recall curve
        labels.append(predicted) # for precision recall curve

    preds = torch.cat([torch.stack(batch) for batch in preds])# for precision recall curve
    labels = torch.cat(labels) # for precision recall curve



    acc = 100.0 * n_correct / n_samples
    print(f'Accuracy of the network on the test images: {acc} %')

    # for precision recall curve
    classes = range(10)
    for i in classes:
        labels_i = labels == i 
        preds_i = preds[:, i]
        writer.add_pr_curve(str(i), labels_i, preds_i, global_step = 0)
        writer.close()

