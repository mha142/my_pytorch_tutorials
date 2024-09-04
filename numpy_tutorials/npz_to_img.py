import numpy as np 
from PIL import Image 
import zipfile 

path = "./expr/gpu_0_samples_162.npz"
data = np.load(path)

print("Arrays in the .npz file:", data.files)

array = data[data.files[0]]
print("shape:", array.shape)


print("shape2:", array[0].shape)
x = array[0].shape
x = array[1].transpose(1, 2, 0)

# Convert to uint8 
if x.dtype != np.uint8:
    x = (x * 255).astype(np.uint8)  # Scale if the values are in range [0, 1]
    
im = Image.fromarray(x)
im.save('162_0.png')

