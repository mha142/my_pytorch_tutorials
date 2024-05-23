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
        return x[:, :, :512, :512]
    
class ConvolutionalVariationalAutoEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        #nn.Conv2d(3, 64, 4, 2, 1) #input channels, output channels, kernel size, stride. padding
        self.encoder = nn.Sequential( # ((n+2p-f)/s ) + 1 >> n = square images dimension, p = padding, f = square filter size, s = stride  
            #input is 512x512 
            nn.Conv2d(3, 32,  kernel_size=(3, 3), stride=(1, 1), padding=1),#output #> 512x512x32
            nn.LeakyReLU(0.01), 
            nn.Conv2d(32, 64,  kernel_size=(3, 3), stride=(2, 2), padding=1),#  256x256x64 
            nn.LeakyReLU(0.01),
            nn.Conv2d(64, 64,  kernel_size=(3, 3), stride=(2, 2),padding=1), #  128x128x64 
            nn.LeakyReLU(0.01),
            nn.Conv2d(64, 64,  kernel_size=(3, 3), stride=(1, 1),padding=1), #  128x128x64
            nn.Flatten()
            )
        
        self.z_mean = torch.nn.Linear(128*128*64, 128) # you pick the latent space dimensions # here I picked 128 
        self.z_log_var = torch.nn.Linear(128*128*64, 128)
        self.z =  torch.nn.Linear(128*128*64, 512)
        
        self.decoder = nn.Sequential(
            #torch.nn.Linear(128, 128*128*64), #this is a fully connected layer (fc)
            torch.nn.Linear(512, 128*128*64),
            Reshape(-1, 64, 128, 128),
            #input: 7x7x64 >> 32x32x64 >>  128x128x64
            #(n-1) x s - (2p) + f
            #ask chat gpt:   n=128 and I have this  nn.ConvTranspose2d(64, 64, kernel_size=(3, 3), stride=(1, 1),  padding=1) what is the output size?
            nn.ConvTranspose2d(64, 64, kernel_size=(3, 3), stride=(1, 1),  padding=1), #7x7x64 >> 128x128x64
            nn.LeakyReLU(0.01), 
            nn.ConvTranspose2d(64, 64, kernel_size=(3, 3),  stride=(2, 2),padding=1),  #13x13x64 >> 255x255x64
            nn.LeakyReLU(0.01),
            nn.ConvTranspose2d(64, 32,  kernel_size=(3, 3), stride=(2, 2),padding=0), #          >> 511x511xx32
            nn.LeakyReLU(0.01),
            nn.ConvTranspose2d(32, 3,  kernel_size=(3, 3), stride=(1, 1), padding=0), #           >>  513x513x3
            Trim(), #trim from 513x513 to 512x512
            nn.Sigmoid()#sigmoid works for images between 0 and 1 
            #nn.tanh() #works for images between -1 and 1 
            )



    def forward(self, x):
        #print("shape BEFORE encoder: ", x.shape)
        x= self.encoder(x) #regular auto encoder line #1
        #print("shape AFTER encoder: ", x.shape) 
        #z_mean = self.z_mean(x)
        #z_log_var = self.z_log_var(x)
        x = self.z(x)
        
        #reparametarization
        #epsilon = torch.randn(z_mean.size(0), z_mean.size(1)).to(z_mean.device)# z_mean means mu #sampling epsilon from the normal distributon 
        #z_reparametrized = z_mean + epsilon * torch.exp(z_log_var/2.) # torch.exp(z_log_var/2.)  is the standard deviation 
        #print("z_reparametrized", z_reparametrized.shape)
    
        #x_reconstructed = self.decoder(z_reparametrized) ##regular auto encoder line #2   -> x_reconstructed = self.decoder(x)
        x_reconstructed = self.decoder(x)
        return x_reconstructed#, z_mean, z_log_var # x_reconstructed -> decoded,  z_mean -> mu,  z_log_var -> sigma # in regular auto encoder there is no z_mean, z_log_var


    def encode(self, x):
        x = self.encoder(x)
        return self.z_mean(x), self.z_log_var(x)
    
    def decode(self, z):
        return self.decoder(z)
    
    

if __name__=="__main__":
    x = torch.randn(4, 512*512*3) #1 is the number of examples (images) # input image dimension = 128 x 128 x 3 (number of channels is 3  for RGB)
    x = x.view(4, 3, 512, 512)
    #vae = ConvolutionalVariationalAutoEncoder(input_dim = 784)
    vae = ConvolutionalVariationalAutoEncoder()
    x_reconstructed, mu, sigma = vae(x)
    print("hello")
    print("x_reconstructed: ", x_reconstructed.shape)
    print("mu:              ", mu.shape)
    print("sigma:           ", sigma.shape)

