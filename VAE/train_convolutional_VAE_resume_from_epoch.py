
# from youtube video: https://www.youtube.com/watch?v=afNuE5z2CQ8&t=764s
import torch 
import torchvision.datasets as datasets #standards datasets 
from tqdm import tqdm # for progress bar 
from torch import nn, optim
from model_convolutional_VAE import ConvolutionalVariationalAutoEncoder
from torchvision import transforms # for image augmentation stuff 
from torchvision.utils import save_image
from torch.utils.data import DataLoader #gives easier dataset management by creating mini batches etc 
import matplotlib.pyplot as plt
import numpy as np 
from torch.utils.tensorboard import SummaryWriter 

#configuration 
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
INPUT_DIM = 784
H_DIM = 200
Z_DIM = 20 

BATCH_SIZE = 32#256#32 
LR_RATE = 0.0005 #1e-3 #3e-4 #learning rate #we use this number cuz its the karpathy constant 

#dataset loading 
dataset = datasets.MNIST(root="dataset/", train=True, transform=transforms.ToTensor(), download=True)
#train_size = int(0.9 *len(dataset))
#val_size = len(dataset) - train_size
#train_dataset, test_set = torch.utils.data.random_split(dataset, [train_size, val_size])
train_loader = DataLoader(dataset=dataset, batch_size=BATCH_SIZE, shuffle=True)
#test_loader = DataLoader(dataset=test_set, batch_size= batch_size, shuffle =True)

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
#resum the state of the model   
    checkpoint = torch.load(filename)
    model.load_state_dict(checkpoint['model'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    epoch = checkpoint['epoch']
    print(f'Resume training from Epoch {epoch+1}')

NUM_EPOCHS = 2  #NUM_EPOCHS >= start_epoch
start_epoch = 1
if start_epoch > 0:
    filename = f"trained_mnist_conv_vae_epoch_{start_epoch}.pth"
    resume(model, optimizer, filename) 
#start training 
writer = SummaryWriter('./runs/vae_mnist')
for currrent_epoch in range(start_epoch, NUM_EPOCHS+1):
#for currrent_epoch in range(NUM_EPOCHS):
    model.train()
    train_epoch_loss = 0
    loop = tqdm(enumerate(train_loader))
    #loop = tqdm(train_loader)
    #for batch_index, (features, _) in enumerate(train_loader):
    for batch_index, (x, _) in loop:
    #for x, _ in loop:
        #forward pass 
        #x = x.to(DEVICE).view(x.shape[0], INPUT_DIM)
        x = x.to(DEVICE) # x are the features 
        #x_reconstructed, mu, sigma = model(x) #model will excute the forward function in the conculotional VAE model class 
        x_reconstructed, z_mean, z_log_var = model(x) #model will excute the forward function in the conculotional VAE model class 

        #compute loss
        reconstruction_loss = loss_fn(x_reconstructed, x)# the inputs is labels of the images # this will push the model to reconstruct the image 
        kl_div = -0.5 * torch.sum(1 + z_log_var - z_mean.pow(2) - torch.exp(z_log_var), axis=1)# sum over latent dimension 

        batchsize= kl_div.size(0)
        kl_div = kl_div.mean() #average over batch dimension

        # pixelwise = loss_fn(x_reconstructed, x,  reduction='none')
        # pixelwise = pixelwise.view(batchsize, -1).sum(axis=1)
        # pixelwise = pixelwise.mean() # average over batch 

 
        #backprobagation
        loss =  reconstruction_loss + kl_div
        #loop.set_postfix(loss=loss.item())
        train_epoch_loss += loss.item()
        
        #backpropagation
        optimizer.zero_grad()#no accumulated gradients from before 
        loss.backward()#compute a new gradient 
        optimizer.step()

    train_epoch_loss = train_epoch_loss / len(train_loader)
    writer.add_scalar('Loss per epoch', train_epoch_loss, currrent_epoch)
    print(f"Train loss for epoch {currrent_epoch+1} is {train_epoch_loss}")

    
        


        
        
        
    save_checkpoint(model, optimizer, currrent_epoch, f"trained_mnist_conv_vae_epoch_{currrent_epoch+1}.pth")
    #save_checkpoint(model, optimizer, currrent_epoch, f"trained_mnist_conv_vae_batchsize_{BATCH_SIZE}_lr_{LR_RATE}_epoch_{currrent_epoch+1}.pth")

writer.close()
print("Done training ....!")
model = model.to(DEVICE)


#get random images from the dataset
n_imgs = 10
idxs = torch.randint(0, len(dataset)-1, (n_imgs, ))
#convert tha images to a shape that the model expects
#so, we convert imags to tensors and stack them 
imgs = torch.cat([dataset[idx][0][None, :] for idx in idxs]).float()

generated_imgs, _, _ = model(imgs) #pass images to the model 

#save both ground truth and generated image 
for i in range(n_imgs):
    ground_truth = imgs[i].view(-1, 1, 28, 28)
    save_image(ground_truth, f"ground_truth_{i}.png")
    out = generated_imgs[i].view(-1, 1, 28, 28)
    save_image(out, f"generated_{i}.png")

exit()

 #x_reconstructed, z_mean, z_log_var = model(x)



########### way 1 
#
# Generate new images
# with torch.no_grad():
#     model.eval()
#     z = torch.randn(64, 2).to(DEVICE)
#     generated_images = model.decode(z).cpu()

# for i in range(64):
#     out = generated_images.view(-1, 1, 28, 28)
#     save_image(out, f"generated_ex{i}.png")



########################################
########### way 1 
def plot_generated_images(data_loader, model, device, 
                          unnormalizer=None,
                          figsize=(20, 2.5), n_images=15):

    fig, axes = plt.subplots(nrows=2, ncols=n_images, 
                             sharex=True, sharey=True, figsize=figsize)
    
    for batch_idx, (features, _) in enumerate(data_loader):
        
        features = features.to(device)

        color_channels = features.shape[1]
        image_height = features.shape[2]
        image_width = features.shape[3]
        
        with torch.no_grad():
            decoded_images, z_mean, z_log_var = model(features)[:n_images]
            
        orig_images = features[:n_images]
        break

    for i in range(n_images):
        for ax, img in zip(axes, [orig_images, decoded_images]):
            curr_img = img[i].detach().to(torch.device(DEVICE)) #"cpu"       
            #if unnormalizer is not None:
            #    curr_img = unnormalizer(curr_img)

            if color_channels > 1:
                curr_img = np.transpose(curr_img, (1, 2, 0))
                save_image(curr_img, f"generated_{img[i]}.png")

                #ax[i].imshow(curr_img)
            else:
                save_image(curr_img.view((image_height, image_width)), f"generated_{img[i]}.png")
                

plot_generated_images(data_loader=train_loader, model=model, device=DEVICE )


#################################################

######## not working here 

# def inference(digit, num_examples=1):

#     #get all the images and add them to a list 
#     images = []
#     idx = 0
#     for x, y in dataset:
#         if y == idx:
#             images.append(x)
#             idx +=1
#         if idx == 10:
#             break

#     #get the mu and sigma for each digit 
#     encodings_digits = []
#     for d in range(10):
#         with torch.no_grad():
#             mu, sigma = model.encode(images[d].view(1, 784)) # view is the same as reshape 
#         encodings_digits.append((mu, sigma))

#     mu, sigma = encodings_digits[digit]
#     for example in range (num_examples):
#         epsilon = torch.randn_like(sigma)
#         z = mu + sigma * epsilon 
#         out = model.decode(z)
#         out = out.view(-1, 1, 28, 28)
#         save_image(out, f"generated_{digit}_ex{example}.png")


# for idx in range(10):
#     inference(idx, num_examples=1)





