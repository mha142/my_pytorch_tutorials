import torch 
import torchvision.datasets as datasets #standards datasets 
from tqdm import tqdm # for progress bar 
from torch import nn, optim
from model import VariationalAutoEncoder
from torchvision import transforms # for image augmentation stuff 
from torchvision.utils import save_image
from torch.utils.data import DataLoader #gives easier dataset management by creating mini batches etc 
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter()

#configuration 
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
INPUT_DIM = 784
H_DIM = 200
Z_DIM = 20 
NUM_EPOCHS = 10
BATCH_SIZE = 32 
LR_RATE = 1e-3 #3e-4 #learning rate #we use this number cuz its the karpathy constant 

#dataset loading 
dataset = datasets.MNIST(root="dataset/", train=True, transform=transforms.ToTensor(), download=True)
train_loader = DataLoader(dataset=dataset, batch_size=BATCH_SIZE, shuffle=True)
model = VariationalAutoEncoder(INPUT_DIM, H_DIM, Z_DIM).to(DEVICE)
optimizer = optim.Adam(model.parameters(), lr=LR_RATE)
loss_fn = nn.BCELoss(reduction="sum")# you can use MSE loss or absolute error loss 

#start training 
for epoch in range(NUM_EPOCHS):
    loop = tqdm(enumerate(train_loader))
    for i, (x, _) in loop:
        #forward pass 
        x = x.to(DEVICE).view(x.shape[0], INPUT_DIM)
        x_reconstructed, mu, sigma = model(x)

        #compute loss
        reconstruction_loss = loss_fn(x_reconstructed, x)# the inputs is labels of the images # this will push the model to reconstruct the image 
        kl_div = -torch.sum(1 + torch.log(sigma.pow(2)) - mu.pow(2) - sigma.pow(2))#this will push the model to standard gaussian 

        #backprobagation
        loss = reconstruction_loss + kl_div
        writer.add_scalar("Loss/train", loss, epoch)
        optimizer.zero_grad()#no accumulated gradients from before 
        loss.backward()#compute a new gradient 
        optimizer.step()
        loop.set_postfix(loss=loss.item())

writer.close()

model = model.to("cpu")
def inference(digit, num_examples=1):

    #get all the images and add them to a list 
    images = []
    idx = 0
    for x, y in dataset:
        if y == idx:
            images.append(x)
            idx +=1
        if idx == 10:
            break

    #get the mu and sigma for each digit 
    encodings_digits = []
    for d in range(10):
        with torch.no_grad():
            mu, sigma = model.encode(images[d].view(1, 784)) # view isthe same as reshape 
        encodings_digits.append((mu, sigma))

    mu, sigma = encodings_digits[digit]
    for example in range (num_examples):
        epsilon = torch.randn_like(sigma)
        z = mu + sigma * epsilon 
        out = model.decode(z)
        out = out.view(-1, 1, 28, 28)
        save_image(out, f"generated_{digit}_ex{example}.png")


for idx in range(10):
    inference(idx, num_examples=1)





