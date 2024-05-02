
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
        return x[:, :, :28, :28]
    
class ConvolutionalVariationalAutoEncoder(nn.Module):
    #def __init__(self, input_dim, h_dim = 200, z_dim = 20): #h_dim = hidden dimension 
    def __init__(self):
        super().__init__()
        #nn.Conv2d(3, 64, 4, 2, 1) #input channels, output channels, kernel size, stride. padding
        self.encoder = nn.Sequential( # (n+2p-f) + 1
            nn.Conv2d(1, 32,  kernel_size=(3, 3), stride=(1, 1), padding=1),# 28x28x32
            nn.LeakyReLU(0.01), 
            nn.Conv2d(32, 64,  kernel_size=(3, 3), stride=(2, 2), padding=1),# 14x14x64
            nn.LeakyReLU(0.01),
            nn.Conv2d(64, 64,  kernel_size=(3, 3), stride=(2, 2),padding=1), # 7x7x64
            nn.LeakyReLU(0.01),
            nn.Conv2d(64, 64,  kernel_size=(3, 3), stride=(1, 1),padding=1), # 7x7x64
            nn.Flatten()
            )
        
        
        self.decoder = nn.Sequential(
            torch.nn.Linear(2, 3136), #this is a fully connected layer (fc)
            Reshape(-1, 64, 7, 7),
            nn.ConvTranspose2d(64, 64, kernel_size=(3, 3), stride=(1, 1),  padding=1), 
            nn.LeakyReLU(0.01), 
            nn.ConvTranspose2d(64, 64, kernel_size=(3, 3),  stride=(2, 2),padding=1), 
            nn.LeakyReLU(0.01),
            nn.ConvTranspose2d(64, 32,  kernel_size=(3, 3), stride=(2, 2),padding=0), 
            nn.LeakyReLU(0.01),
            nn.ConvTranspose2d(32, 1,  kernel_size=(3, 3), stride=(1, 1), padding=0), 
            Trim(), #trim from 1x29x29 to -> 1x28x28
            nn.Sigmoid()#sigmoid works for images between 0 and 1 
            #nn.tanh() #works for images between -1 and 1 
            )



        self.z_mean = torch.nn.Linear(3136, 2)
        self.z_log_var = torch.nn.Linear(3136, 2)



    def forward(self, x):
        x= self.encoder(x) #regular auto encoder line #1 
        z_mean = self.z_mean(x)
        z_log_var = self.z_log_var(x)
        #reparametarization
        epsilon = torch.randn(z_mean.size(0), z_mean.size(1)).to(z_mean.device)# z_mean means mu #sampling epsilon from the normal distributon 
        z_reparametrized = z_mean + epsilon * torch.exp(z_log_var/2.) # torch.exp(z_log_var/2.)  is the standard deviation 
        x_reconstructed = self.decoder(z_reparametrized) ##regular auto encoder line #2   -> x_reconstructed = self.decoder(x)
        return x_reconstructed, z_mean, z_log_var # x_reconstructed -> decoded,  z_mean -> mu,  z_log_var -> sigma # in regular auto encoder there is no z_mean, z_log_var


    def encode(self, x):
        x = self.encoder(x)
        return self.z_mean(x), self.z_log_var(x)
    
    def decode(self, z):
        return self.decoder(z)
    
    

if __name__=="__main__":
    x = torch.randn(4, 784) #1 is the number of examples (images) # input image dimension = 28 * 28 = 784
    #vae = ConvolutionalVariationalAutoEncoder(input_dim = 784)
    vae = ConvolutionalVariationalAutoEncoder()
    x_reconstructed, mu, sigma = vae(x)
    print("hello")
    print("x_reconstructed: ", x_reconstructed.shape)
    print("mu:              ", mu.shape)
    print("sigma:           ", sigma.shape)

