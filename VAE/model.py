import torch 
import torch.nn.functional as F 
from torch import nn

#pipeline:
#input image -> idden dim -> mean, std -> parametrization trick -> decoder -> output image 
class VariationalAutoEncoder(nn.Module):
    def __init__(self, input_dim, h_dim = 200, z_dim = 20): #h_dim = hidden dimension 
        super().__init__()
        self.img_2hid = nn.Linear(input_dim, h_dim)
        self.hid_2mu = nn.Linear(h_dim, z_dim)
        self.hid_2sigma = nn.Linear(h_dim, z_dim)

        #decoder 
        self.z_2hid = nn.Linear(z_dim, h_dim)
        self.hid_2img = nn.Linear(h_dim, input_dim)

        self.relu = nn.ReLU()

    def encode(self, x):
        #h = F.relu(self.img_2hid(x))
        h = self.relu(self.img_2hid(x))
        mu, sigma = self.hid_2mu(h), self.hid_2sigma(h)
        return mu, sigma

    def decode(self, z):
        h = self.relu(self.z_2hid(z))
        return torch.sigmoid(self.hid_2img(h)) #sigmoid makes the value between 0 abnd 1 


    def forward(self, x):
        mu, sigma = self.encode(x)
        epsilon = torch.randn_like(sigma)
        z_reparametrized = mu + sigma * epsilon 
        x_reconstructed = self.decode(z_reparametrized)
        return x_reconstructed, mu, sigma


if __name__=="__main__":
    x = torch.randn(4, 784) #1 is the number of examples (images) # input image dimension = 28 * 28 = 784
    vae = VariationalAutoEncoder(input_dim = 784)
    x_reconstructed, mu, sigma = vae(x)
    print("x_reconstructed: ", x_reconstructed.shape)
    print("mu:              ", mu.shape)
    print("sigma:           ", sigma.shape)

