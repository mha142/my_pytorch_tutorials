import numpy as np 
from PIL import Image 
import zipfile 

for j in range (10):
    path = f"./expr/gpu_0_samples_{j}.npz"
    data = np.load(path)


    print("Arrays in the .npz file:", data.files)

    array = data[data.files[0]]
    print("Array shape:", array.shape)

    #loop through each image and save it 
    for i in range(array.shape[0]):
        print(i)
        print("Image shape:", array[i].shape)
        x = array[i].transpose(1, 2, 0)

        # Convert to uint8 
        if x.dtype != np.uint8:
            x = (x * 255).astype(np.uint8)  # Scale if the values are in range [0, 1]
            
        im = Image.fromarray(x)
        im.save(f'./samples/sample_{j}_image_{i}.png')

