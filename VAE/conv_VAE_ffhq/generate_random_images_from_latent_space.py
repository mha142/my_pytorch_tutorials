import torch
import torchvision
from torchvision.utils import save_image
from model_convolutional_VAE_ffhq import ConvolutionalVariationalAutoEncoder


def resume(model, filename):#restart training from a specific epoch 
#resume the state of the model   
    checkpoint = torch.load(filename)
    model.load_state_dict(checkpoint['model'])


model = ConvolutionalVariationalAutoEncoder()

epoch = 3
filename = f"trained_ffhq_conv_vae_epoch_{epoch}.pth"
resume(model, filename) 

# Number of images to generate
num_images = 10

# Set the model to evaluation mode
model.eval()

# Generate random latent vectors
latent_vectors = torch.randn(num_images, 128)  # latent_dim is the dimensionality of the latent space

# Decode the latent vectors into images
with torch.no_grad():
    generated_images = model.decoder(latent_vectors)

# Save or visualize the generated images
for i in range(num_images):
    save_image(generated_images[i], f"generated_image_{i}.png")  # Save each generated image to a file