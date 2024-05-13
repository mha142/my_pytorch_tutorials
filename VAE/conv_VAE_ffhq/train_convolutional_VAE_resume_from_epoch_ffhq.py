
# from youtube video: https://www.youtube.com/watch?v=afNuE5z2CQ8&t=764s
import torch 
import torchvision.datasets as datasets #standards datasets 
from tqdm import tqdm # for progress bar 
from torch import nn, optim
from model_convolutional_VAE_ffhq import ConvolutionalVariationalAutoEncoder
from torchvision import transforms # for image augmentation stuff 
from torchvision.utils import save_image
from torch.utils.data import DataLoader, random_split #gives easier dataset management by creating mini batches etc 
import matplotlib.pyplot as plt
import numpy as np 
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter 

from data import FFHQ_Data
from torchvision.utils import save_image




#configuration 
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
INPUT_DIM = 128 * 128
H_DIM = 200
Z_DIM = 20 

BATCH_SIZE = 32#256#32 
LR_RATE = 0.0005 #1e-3 #3e-4 #learning rate #we use this number cuz its the karpathy constant 

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

train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size= BATCH_SIZE, shuffle =True)

Images = next(iter(train_loader))
#print(f"Shape of image [N, C, H, W]: {Images.shape}") # Batch, View, Channel, Height, Width


model = ConvolutionalVariationalAutoEncoder().to(DEVICE)
optimizer = optim.Adam(model.parameters(), lr=LR_RATE)
loss_fn = nn.BCELoss(reduction="sum")# you can use MSE loss or absolute error loss 


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
    

NUM_EPOCHS = 4  #NUM_EPOCHS >= start_epoch
start_epoch = 2
if start_epoch > 0:
    filename = f"trained_ffhq_conv_vae_epoch_{start_epoch}.pth"
    resume(model, optimizer, filename) 
#start training 
writer = SummaryWriter('./runs/conv_vae_ffhq')

#trainingEpoch_loss_list = []
#validationEpoch_loss_list = []
for currrent_epoch in range(start_epoch, NUM_EPOCHS+1):
    print(f"Start training for epoch # {currrent_epoch+1} ......... ")
#for currrent_epoch in range(NUM_EPOCHS):
    model.train()
    train_epoch_loss = 0
    #loop = tqdm(enumerate(train_loader))
    #loop = tqdm(train_loader)
    #for batch_index, (features, _) in enumerate(train_loader):
    #for batch_index, x in loop:
    #for x, _ in loop:
    for x in train_loader: #training loop 
        #forward pass 
        
        #print(x.shape)
        x = x.to(DEVICE) # x are the features 
        #x = x.to(DEVICE).view(x.shape[0], 3, 128, 128)
        #x_reconstructed, mu, sigma = model(x) 
        x_reconstructed, z_mean, z_log_var = model(x) #model will excute the forward function in the conculotional VAE model class 
        # print("x_reconstructed: ", x_reconstructed.shape)
        # save_image(x[0].detach().cpu(), "image.png")
        # print("x_reconstructed[0]: ", x_reconstructed[0].shape)
        # save_image(x_reconstructed[0].detach().cpu(), "image.png")
        # exit()


        #compute loss
        # reconstruction_loss = loss_fn(x_reconstructed, x)# the inputs is labels of the images # this will push the model to reconstruct the image 
        # kl_div = -0.5 * torch.sum(1 + z_log_var - z_mean.pow(2) - torch.exp(z_log_var), axis=1)# sum over latent dimension 

        # kl_div = kl_div.mean() #average over batch dimension
        # loss =  reconstruction_loss + kl_div

        loss = VAE_loss_func(x, x_reconstructed, z_mean, z_log_var)

        #loop.set_postfix(loss=loss.item())
        train_epoch_loss += loss.item()
        
        #backpropagation
        optimizer.zero_grad()#no accumulated gradients from before 
        loss.backward()#compute a new gradient 
        optimizer.step()

    train_epoch_loss = train_epoch_loss / len(train_loader)
    #trainingEpoch_loss_list.append(train_epoch_loss)
    #writer.add_scalar('Training loss per epoch', train_epoch_loss, currrent_epoch)
    writer.add_scalars("Epoch Loss", {'Training Loss': train_epoch_loss}, currrent_epoch)
    #print epoch loss and save check point after ___ epochs:
    if currrent_epoch % 1 == 0:#will print after every time the epoch is a multiple of 100 (if the remainder is 0 then it is a multiple).
        print(f"Train loss for epoch {currrent_epoch+1} is {train_epoch_loss}")    
        save_checkpoint(model, optimizer, currrent_epoch, f"trained_ffhq_conv_vae_epoch_{currrent_epoch+1}.pth")
    #save_checkpoint(model, optimizer, currrent_epoch, f"trained_mnist_conv_vae_batchsize_{BATCH_SIZE}_lr_{LR_RATE}_epoch_{currrent_epoch+1}.pth")

    #validation loop 
    model.eval()
    valid_epoch_loss = 0
    with torch.no_grad():
        for x in test_loader:
            x = x.to(DEVICE)
            #image = image.to(DEVICE)
            x_reconstructed, z_mean, z_log_var = model(x)
            
            save_image(x[0].detach().cpu(), "GTimage.png")#you need to go to the cpu to save the image 
            #print("x_reconstructed[0]: ", x_reconstructed[0].shape)
            save_image(x_reconstructed[0].detach().cpu(), "reconst_image.png")

            #cv2.imwrite("groundtruth.png", x.detach().cpu().numpy())
            #cv2.imwrite("reconstructed.png", x_reconstructed.detach().cpu().numpy())

            loss = VAE_loss_func(x, x_reconstructed, z_mean, z_log_var)
            valid_epoch_loss += loss.item()
        valid_epoch_loss = valid_epoch_loss / len(test_loader)
        #validationEpoch_loss_list.append(valid_epoch_loss)
        writer.add_scalars("Epoch Loss", {'Validation Loss': valid_epoch_loss}, currrent_epoch)


writer.close()
print("Done training ....!")


exit()
#get random images from the dataset
n_imgs = 10
idxs = torch.randint(0, len(train_dataset)-1, (n_imgs, ))
#convert tha images to a shape that the model expects
#so, we convert imags to tensors and stack them 
imgs = torch.cat([train_dataset[idx][0][None, :] for idx in idxs]).float()
print("imgs.shape: ", imgs.shape)
#imgs = imgs.view(imgs.shape[0], 3, 128, 128)
#imgs = imgs.view(n_imgs, 3, 128, 128)

generated_imgs, _, _ = model(imgs) #pass images to the model 

#save both ground truth and generated image 
for i in range(n_imgs):
    ground_truth = imgs[i].view(3, 128, 128)
    save_image(ground_truth, f"ground_truth_{i}.png")
    out = generated_imgs[i].view(3, 128, 128)
    save_image(out, f"generated_{i}.png")





