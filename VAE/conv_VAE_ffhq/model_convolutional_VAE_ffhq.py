import torch 
import torch.nn.functional as F 
from torch import nn



#pipeline:
#input image -> idden dim -> mean, std -> parametrization trick -> decoder -> output image 
class Reshape(nn.Module):
    def __init__(self, *args):
        super().__init__()
        self.shape = args

    def forward(self, x):
        return x.view(self.shape)


class Trim(nn.Module):
    def __init__(self, *args):
        super().__init__()

    def forward(self, x):
        return x[:, :, :128, :128]
    
class ConvolutionalVariationalAutoEncoder(nn.Module):
    #def __init__(self, input_dim, h_dim = 200, z_dim = 20): #h_dim = hidden dimension 
    def __init__(self):
        super().__init__()
        #nn.Conv2d(3, 64, 4, 2, 1) #input channels, output channels, kernel size, stride. padding
        self.encoder = nn.Sequential( # ((n+2p-f)/s ) + 1 >> n = square images dimension, p = padding, f = square filter size, s = stride  
            nn.Conv2d(3, 32,  kernel_size=(3, 3), stride=(1, 1), padding=1),# 28x28x32 >> 128x128x32
            nn.LeakyReLU(0.01), 
            nn.Conv2d(32, 64,  kernel_size=(3, 3), stride=(2, 2), padding=1),# 14x14x64 >> 64x64x64
            nn.LeakyReLU(0.01),
            nn.Conv2d(64, 64,  kernel_size=(3, 3), stride=(2, 2),padding=1), # 7x7x64 >> 32x32x64
            nn.LeakyReLU(0.01),
            nn.Conv2d(64, 64,  kernel_size=(3, 3), stride=(1, 1),padding=1), # 7x7x64 >> 32x32x64
            nn.Flatten()
            )
        
        
        self.decoder = nn.Sequential(
            torch.nn.Linear(128, 32*32*64), #this is a fully connected layer (fc)
            Reshape(-1, 64, 32, 32),
            nn.ConvTranspose2d(64, 64, kernel_size=(3, 3), stride=(1, 1),  padding=1), # 32x32x64
            nn.LeakyReLU(0.01), 
            nn.ConvTranspose2d(64, 64, kernel_size=(3, 3),  stride=(2, 2),padding=1), 
            nn.LeakyReLU(0.01),
            nn.ConvTranspose2d(64, 32,  kernel_size=(3, 3), stride=(2, 2),padding=0),
            nn.LeakyReLU(0.01),
            nn.ConvTranspose2d(32, 3,  kernel_size=(3, 3), stride=(1, 1), padding=0), 
            Trim(), #trim from 1x29x29 to -> 1x28x28
            nn.Sigmoid()#sigmoid works for images between 0 and 1 
            #nn.tanh() #works for images between -1 and 1 
            )



        self.z_mean = torch.nn.Linear(32*32*64, 128) # you pick the latent space dimensions # here I picked 128 
        self.z_log_var = torch.nn.Linear(32*32*64, 128)



    def forward(self, x):
        #print("shape BEFORE encoder: ", x.shape)
        x= self.encoder(x) #regular auto encoder line #1
        #print("shape AFTER encoder: ", x.shape) 
        z_mean = self.z_mean(x)
        z_log_var = self.z_log_var(x)
        # print("z_mean", z_mean.shape)
        # print("z_log_var", z_log_var.shape)
        
        #reparametarization
        epsilon = torch.randn(z_mean.size(0), z_mean.size(1)).to(z_mean.device)# z_mean means mu #sampling epsilon from the normal distributon 
        z_reparametrized = z_mean + epsilon * torch.exp(z_log_var/2.) # torch.exp(z_log_var/2.)  is the standard deviation 
        #print("z_reparametrized", z_reparametrized.shape)
    
        x_reconstructed = self.decoder(z_reparametrized) ##regular auto encoder line #2   -> x_reconstructed = self.decoder(x)
        return x_reconstructed, z_mean, z_log_var # x_reconstructed -> decoded,  z_mean -> mu,  z_log_var -> sigma # in regular auto encoder there is no z_mean, z_log_var


    def encode(self, x):
        x = self.encoder(x)
        return self.z_mean(x), self.z_log_var(x)
    
    def decode(self, z):
        return self.decoder(z)
    
    

if __name__=="__main__":
    pass
    #This is NOT working #
    # # 128x128 = 16384
    # x = torch.randn(4, 16384) #1 is the number of examples (images) # input image dimension = 28 * 28 = 784
    # #x = x.view(4, 3, 128, 128)
    # #vae = ConvolutionalVariationalAutoEncoder(input_dim = 784)
    # vae = ConvolutionalVariationalAutoEncoder()
    # x_reconstructed, mu, sigma = vae(x)
    # print("hello")
    # print("x_reconstructed: ", x_reconstructed.shape)
    # print("mu:              ", mu.shape)
    # print("sigma:           ", sigma.shape)

