# from youtube video: https://www.youtube.com/watch?v=afNuE5z2CQ8&t=764s
import torch 
import torchvision.datasets as datasets #standards datasets 
#from tqdm import tqdm # for progress bar 
from torch import nn, optim
from torch.nn import functional as F
from model_autoencoder_512 import ConvolutionalVariationalAutoEncoder
from torchvision import transforms # for image augmentation stuff 
from torchvision.utils import save_image
from torch.utils.data import DataLoader, random_split #gives easier dataset management by creating mini batches etc 

#import matplotlib.pyplot as plt
import numpy as np 
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter 

from data_autoencoder_512 import FFHQ_Data
from torchvision.utils import save_image

import time
from datetime import timedelta



#configuration 
#DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DEVICE = torch.device("cuda")
NUM_EPOCHS = 1000  #NUM_EPOCHS >= start_epoch
PRINT_EVERY = 200
print('Device is: ', DEVICE, '___ Epochs: ', NUM_EPOCHS, '___ Print every: ', PRINT_EVERY )
#INPUT_DIM = 128 * 128


BATCH_SIZE = 40
LR_RATE = 0.0001    #1e-3 #3e-4 #learning rate #we use this number cuz its the karpathy constant 

#dataset loading 
p = Path('./ffhq52000')
ffhq_data = FFHQ_Data(data_dir= p, transform= transforms.ToTensor())
data_length = ffhq_data.__len__() 
print(f'no. of objs in the entire dataset is:', data_length)
# for i in range(data_length):
#     ffhq_data.__getitem__(i)
#     break

# Split the dataset into training and validation sets
train_size = int(0.8 *len(ffhq_data))#0.4, 0.5, 0.8
test_size = len(ffhq_data) - train_size
#generate a random number 
gen = torch.Generator()
gen.manual_seed(0) # this to make sure that the data is being splitted randomly in the same way each time we run the code 

train_dataset, test_dataset = random_split(ffhq_data, [train_size, test_size], generator=gen)#if we omit "generator=gen" then each time we run the code the random split will be different, and this will produce different training results 

print('no of training samples:', len(train_dataset))
print('no of testing samples:', len(test_dataset))

train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, num_workers=30, shuffle=True) #num_workers is 30 beacuse the number of cpu cores is 56 
test_loader = DataLoader(dataset=test_dataset, batch_size= BATCH_SIZE, num_workers=30, shuffle =True)

Images = next(iter(train_loader))
#print(f"Shape of image [N, C, H, W]: {Images.shape}") # Batch, View, Channel, Height, Width


model = ConvolutionalVariationalAutoEncoder().to(DEVICE)
optimizer = optim.Adam(model.parameters(), lr=LR_RATE)
#loss_fn = nn.BCELoss(reduction="sum")# you can use MSE loss or absolute error loss 
#loss_fn = nn.MSELoss(reduction="mean")




def save_checkpoint(model, optimizer, epoch, filename):
    torch.save({
    'epoch' : epoch,
    'optimizer': optimizer.state_dict(),
    'model': model.state_dict(),
}, filename)
    print(f"Epoch {epoch+1} | Training checkpoint saved at {filename}")
    
def resume(model, optimizer, filename):#restart training from a specific epoch 
#resume the state of the model   
    checkpoint = torch.load(filename)
    model.load_state_dict(checkpoint['model'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    epoch = checkpoint['epoch']
    print(f'Resume training from Epoch {epoch+1}')


def VAE_loss_func(x, x_reconstructed, z_mean, z_log_var):
    
    reconstruction_loss = loss_fn(x_reconstructed, x)# the inputs is labels of the images # this will push the model to reconstruct the image 
    kl_div = -0.5 * torch.sum(1 + z_log_var - z_mean.pow(2) - torch.exp(z_log_var), axis=1)# sum over latent dimension 
    kl_div = kl_div.mean() #average over batch dimension
    loss =  reconstruction_loss + kl_div
    return loss
    


start_epoch = 0
if start_epoch > 0:
    filename = f"./autoencoder/trained_ffhq_512_autoencoder_epoch_{start_epoch}.pth"
    resume(model, optimizer, filename) 
#start training 
writer = SummaryWriter('./autoencoder512/conv_vae_ffhq_512')

#trainingEpoch_loss_list = []
#validationEpoch_loss_list = []
for currrent_epoch in range(start_epoch, NUM_EPOCHS+1):
    start_time = time.monotonic()
    print(f"Start training for epoch # {currrent_epoch+1} ......... ")
#for currrent_epoch in range(NUM_EPOCHS):
    model.train()
    train_epoch_loss = 0
    
    for x in train_loader: #training loop 
        #forward pass 
        
        
        x = x.to(DEVICE) # x are the features 
        optimizer.zero_grad()#no accumulated gradients from before 
        #x_reconstructed, z_mean, z_log_var = model(x) #model will excute the forward function in the conculotional VAE model class 
        x_reconstructed = model(x) #([40, 3, 512, 512])
        x_reconstructed = torch.flatten(x_reconstructed, start_dim=2) #[40, 3, 262144]
        print(x_reconstructed.shape)
        x_hat =  torch.flatten(x, start_dim=2)

        #loss = VAE_loss_func(x, x_reconstructed, z_mean, z_log_var)
        #loss = loss_fn(x_reconstructed, x)
        loss = torch.mean(torch.sum((x_hat - x_reconstructed)**2, dim=1), dim=-1).mean()#torch.mean (takes the mean over each pixel), .mean (takes the mean over all the images in the batch)
        #dim=1 means "3" channels r, g, b, so take the sum (r-r^)squared + (g-g^)squared + (b-b^)squared
        #dim = -1 means "262144" so take the mean over all the pixels in the image 
       
        train_epoch_loss += loss.item()
        
        #backpropagation
        
        loss.backward()#compute a new gradient 
        optimizer.step()

    train_epoch_loss = train_epoch_loss / len(train_loader)
    
    writer.add_scalars("Epoch Loss", {'Training Loss': train_epoch_loss}, currrent_epoch)
    #print epoch loss and save check point after ___ epochs:
    print(f"Train loss for epoch {currrent_epoch+1} is {train_epoch_loss}")  
    if currrent_epoch % PRINT_EVERY == 0:#will print after every time the epoch is a multiple of 100 (if the remainder is 0 then it is a multiple).
        save_checkpoint(model, optimizer, currrent_epoch, f"./autoencoder/trained_ffhq_512_autoencoder_epoch_{currrent_epoch+1}.pth")

    #validation loop 
    model.eval()
    valid_epoch_loss = 0
    with torch.no_grad():
        for x in test_loader:
            x = x.to(DEVICE)
            #image = image.to(DEVICE)
        
            #x_reconstructed, z_mean, z_log_var = model(x)
            x_reconstructed = model(x)
            
            if currrent_epoch % PRINT_EVERY == 0:
                save_image(x[0].detach().cpu(), f"./autoencoder/GT_image_512_e{currrent_epoch+1}.png")#you need to go to the cpu to save the image 
                #print("x_reconstructed[0]: ", x_reconstructed[0].shape)
                save_image(x_reconstructed[0].detach().cpu(), f"./autoencoder/reconst_image_512_e{currrent_epoch+1}.png")

            x_reconstructed = torch.flatten(x_reconstructed, start_dim=2)
            x_hat =  torch.flatten(x, start_dim=2)

            #loss = VAE_loss_func(x, x_reconstructed, z_mean, z_log_var)
            loss = torch.mean(torch.sum((x_hat - x_reconstructed)**2, dim=1), dim=-1).mean()
            valid_epoch_loss += loss.item()
        valid_epoch_loss = valid_epoch_loss / len(test_loader)
        #validationEpoch_loss_list.append(valid_epoch_loss)
        writer.add_scalars("Epoch Loss", {'Validation Loss': valid_epoch_loss}, currrent_epoch)

    end_time = time.monotonic()
    print("training and validating one epoch took:", timedelta(seconds= end_time - start_time))

writer.close()
print("Done training ....!")





