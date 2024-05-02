import sys 
sys.path.append('..') 
import cv2 
from FaceBoxes import FaceBoxes 
from FaceBoxes.models.faceboxes import FaceBoxesNet

import numpy as np
import torch 
import torch.nn as nn
import torch.optim as optim
import torchvision 
import torchvision.transforms as transforms
from torchvision.transforms import Compose

import os 
import pandas as pd 
from torch.utils.data import Dataset, DataLoader, random_split
from skimage import io 

import matplotlib.pyplot as plt 

from pathlib import Path
from typing import List, Tuple, Optional, Callable, Any
from NumpyIO import readFile

from torch import Tensor
from PIL import Image
import models
from models import maha_mobilenet_v1 #mobilenet_v1
from TDDFA import TDDFA
from utils.tddfa_util import (
    load_model, _parse_param, similar_transform,
    ToTensorGjz, NormalizeGjz
)
from utils.functions import (
    crop_img, parse_roi_box_from_bbox, parse_roi_box_from_landmark,
)

import pickle


class Facekit_Processed(Dataset):
    #def __init__(self, data_dir: Path=Path('./facekit_data'), transform:Optional[Callable]=None) -> None:
    def __init__(self, data_dir: Path, transform:Optional[Callable]=None) -> None:
        super().__init__()
        self.image_labels =[]
        self.data_folder = data_dir
        self.transform = transform 
        self.sub_dir = self.__get_dir(data_dir)
      
    def __get_dir(self, data_dir: Path) -> List[Path]:
        return [x for x in data_dir.iterdir() if x.is_dir()] # if x.is_dir() means if we have a subdirectory 
    
    def __len__(self)-> int:
        return len(self.sub_dir)

    def __getitem__(self, idx: int) -> Tuple[List[Any], List[Any], Tensor, Tensor, str]:
        face_path = self.sub_dir[idx]
        label = face_path.parts[-1] 
        
        #print('face_path:', face_path)
        #print('label', label)

        # Image.open read images as PILImage
        # Always call ToTensor() to convert them to Tensors
        img_left  = Image.open(face_path.joinpath(label+'_leftViewShape_cropped_facebox.jpg'))#'leftViewShape_g.jpg')) 
        img_front = Image.open(face_path.joinpath(label+'_frontViewShape_cropped_facebox.jpg')) #image are between 0-1
        img_right = Image.open(face_path.joinpath(label+'_rightViewShape_cropped_facebox.jpg'))#'rightViewShape_g.jpg'))
        #print('image_front:', img_front) Does this give the same path as image_path in line 820?


        if self.transform is not None:
            img_left = self.transform(img_left)
            img_front = self.transform(img_front)
            img_right = self.transform(img_right)

        #left_visible  = torch.tensor(readFile(face_path.joinpath('left_visible.npy')), dtype=torch.float32) 
        #front_visible = torch.tensor(readFile(face_path.joinpath('front_visible.npy')), dtype=torch.float32)
        #right_visible = torch.tensor(readFile(face_path.joinpath('right_visible.npy')), dtype=torch.float32)
         
        PointParam = torch.tensor(readFile(face_path.joinpath(label+'_PointParam.npy')), dtype=torch.float32).view(-1)
        #PointParam = PointParam[:-1]

        #modelMat = torch.tensor(readFile(face_path.joinpath('ModelMat.npy')), dtype=torch.float32)

        #return [img_left, img_front, img_right], [left_visible, front_visible, right_visible], modelMat,  PointParam, lable
        #return img_front, PointParam, label
        return  [img_left, img_front, img_right], PointParam, label
        
def checkpoint(model, optimizer, filename):
    #torch.save(model.state_dict(), filename)
    torch.save({
    'optimizer': optimizer.state_dict(),
    'model': model.state_dict(),
}, filename)
    
def resume(model, optimizer, filename):
    #model.load_state_dict(torch.load(filename))
    checkpoint = torch.load(filename)
    model.load_state_dict(checkpoint['model'])
    optimizer.load_state_dict(checkpoint['optimizer'])

def NMELoss(predicted,target):
    #define loss function and optimizer
    # https://stackoverflow.com/questions/55955059/python-coding-of-the-normalized-mean-error
    xdiff = predicted - target
    NME=((np.sum(abs(xdiff)))/(np.sum(abs(np.mean(target)-(target)))))
    return torch.from_numpy(NME)

def train_processed_images():
    #set device 
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    #load data 
    data_dir = './facekit_data'  #'./facekit_4'
    p = Path(data_dir)
    #load data     
    transform_normalize = NormalizeGjz(mean=127.5, std=128)
    transform_to_tensor = ToTensorGjz()
    transform = Compose([transform_to_tensor, transform_normalize])
    facekit_data = Facekit_Processed(data_dir= p, transform= transforms.ToTensor())#transforms.ToTensor()) #use this to pass the path as an argument (this means that the path is NOT specified in the class itself)
    data_length = facekit_data.__len__() 
    print(f'no. of objs in the entire dataset is:', data_length)

    #RetopologyNN/remesh.py
    # Split the dataset into training and validation sets
    train_size = int(0.8 *len(facekit_data))#0.4, 0.5, 0.8
    test_size = len(facekit_data) - train_size
    #generate a random number 
    gen = torch.Generator()
    gen.manual_seed(0) # this to make sure that the data is being splitted randomly in the same way each time we run the code 
    
    train_dataset, test_dataset = random_split(facekit_data, [train_size, test_size], generator=gen)#if we omit "generator=gen" then each time we run the code the random split will be different, and this will produce different training results 
    
    print('no of training samples:', len(train_dataset))
    print('train data:', train_dataset)
    #print('no of testing samples:', len(test_dataset))


    batch_size = 1
    train_loader = DataLoader(dataset=train_dataset, batch_size= batch_size, shuffle =False)
    #valid_loader = DataLoader(dataset=val_dataset, batch_size= batch_size, shuffle =True)
    test_loader = DataLoader(dataset=test_dataset, batch_size= batch_size, shuffle =False)

    #face_boxes = FaceBoxes() #to detect the face bounding box and landmarks 
    
    #param_mean = 127.5
    #param_std = 128

    #define model
    model = maha_mobilenet_v1.MobileNet(widen_factor=1.0, num_classes=48)
    model.to(device)
   
    criterion = nn.MSELoss()# the same as ((input-target)**2).mean()
    #criterion =  nn.CrossEntropyLoss()
    #criterion = NMELoss
    optimizer = optim.SGD(model.parameters(), lr=10e-5, momentum=0.9, weight_decay= 0.0005, nesterov=True)
    
    n_epochs = 100
    trainingEpoch_loss = []
    validationEpoch_loss = []
    n_total_steps = len(train_dataset)
    for epoch in range(n_epochs):  # loop over the dataset multiple times
        step_loss = []
        model.train()
        #get an image from the train loader
        for batch_index, data in enumerate(train_loader):#, 0):#loop over all the data one time
            #if batch_index ==0: #uncomment this line if you want to train and test one image only 
                img_front, PointParam, label = data
                img_front = img_front.to(device = device)
                PointParam = PointParam.to(device = device)
                #load the image 
                
                #image_path = data_dir  + '/' + label[0]+'/'+label[0]+'_frontViewShape.jpg'
                #print(image_path)
                #img = cv2.imread(image_path)
                #boxes = face_boxes(img) 
                #n = len(boxes)
                #print(f'image: {image_path}, >>>>>> has {n} face(s)')
                #print('boxes: ', boxes)

                #input_param, roi_box_lst = preprocess_image(img, boxes)
                #input_param = input_param.to(device = device)

                #print(roi_box_lst)
                #print('input_param.shape', input_param.shape)

                #Clear the gradients to be all zeros 
                #optimizer.zero_grad()
                
                #forward pass
                param = model(img_front)
                #print('pridected param shape', param.shape)
                #print('Ground truth param shape', PointParam.shape)

                #print('pridected param: ', param)
                #print('Ground truth param: ', PointParam)
                # calculating the loss between original and predicted data points
                loss = criterion(param, PointParam) #(prediected, target or ground truth )
                #diffrent kind of computing loss (needs)
                #reonstruct the vertices 
                #predicted_verts = convert param to obj #NOW
                #ground_truth_verts = convert param to obj 
                #loss = criterion(predicted_verts, ground_truth_verts) 
                #print("loss:", loss)
                step_loss.append(loss.item())
    
                #Clear the gradients to be all zeros 
                #optimizer.zero_grad()
                #backward pass to calculate gradients
                #loss.requires_grad = True
                loss.backward()
                #optimize means Update the weights
                optimizer.step()
                optimizer.zero_grad()
                #https://discuss.pytorch.org/t/how-to-draw-loss-per-epoch/22333
                #running_loss += param.shape[0] * loss.item()#The loss is averaged by the batch_size, which is the first dimension similar to (np.array(step_loss).mean()
                
                #type 2 of prinintng stats 
                #https://stackoverflow.com/questions/74754493/plot-training-and-validation-loss-in-pytorch
                if (batch_index+1) % 1 == 0:#print the loss for every image in the dataset
                    print (f'Epoch [{epoch+1}/{n_epochs}], Step [{batch_index+1}/{n_total_steps}], Loss: {loss.item():.4f}')
                
        #https://stackoverflow.com/questions/74754493/plot-training-and-validation-loss-in-pytorch
        trainingEpoch_loss.append(np.array(step_loss).mean())
        # print epoch loss
        #print(f'training loss after epoch #{epoch+1}: {trainingEpoch_loss / len(train_dataset)}')#Fix
        print(f'training loss after epoch #{epoch+1}: {np.array(step_loss).mean()}')
        #save the checkpoint 
        checkpoint(model, optimizer, f"facekit_3views_3ddfa_v2_trained_model_epoch_{epoch}.pth")
        
        #validation 
        
        print(f'validating the model now after epoch #{epoch+1}............... ')
        #test the model after finishing the current epoch 
        model.eval() # Optional when not using Model Specific layer
        for batch_index, data in enumerate(test_loader):#loop over all the data one time
            validationStep_loss = []

            img_front, PointParam, label = data
            img_front = img_front.to(device = device)
            PointParam = PointParam.to(device = device)

            #image_path = './facekit_data/'+label[0]+'/'+label[0]+'_frontViewShape.jpg'
            #img = cv2.imread(image_path)
            #boxes = face_boxes(img) 
            #input_param, roi_box_lst = preprocess_image(img, boxes)
            #input_param = input_param.to(device = device)
            #with torch.no_grad():
            #forward pass
            param = model(img_front)
            #compute loss 
            validation_loss = criterion(param, PointParam) #(prediected model output, target or ground truth)
            validationStep_loss.append(validation_loss.item())
        print(f'validation accuracy after epoch #{epoch+1}: {np.array(validationStep_loss).mean()}')
        validationEpoch_loss.append(np.array(validationStep_loss).mean())
        
    print('**********Final Results**********')
    print('trainingEpoch_loss',trainingEpoch_loss)
    #print('validationEpoch_loss',validationEpoch_loss)
    #plot
    plt.plot(trainingEpoch_loss, label='train_loss')
    plt.plot(validationEpoch_loss,label='val_loss')
    #plt.legend()
    plt.savefig('training_validation_curves_3views.png')

    print('Finished Training')

#https://saturncloud.io/blog/how-to-normalize-image-dataset-using-pytorch/
    
def get_mean_std():
    # Compute the mean and standard deviation of all pixels in the dataset
    transform = Compose([transforms.ToTensor()])
    data_dir = './facekit_data'  #'./facekit_4'
    p = Path(data_dir)
    dataset = Facekit_Processed(data_dir= p, transform= transform)
    batch_size = 32
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    num_pixels = 0
    mean = 0.0
    std = 0.0
    for images, _ in loader:
        batch_size, num_channels, height, width = images.shape
        num_pixels += batch_size * height * width
        mean += images.mean(axis=(0, 2, 3)).sum()
        std += images.std(axis=(0, 2, 3)).sum()
    mean /= num_pixels
    std /= num_pixels
    print('mean', mean)
    print('std', std)


def train_processed_images_plot_after_every_epoch():
    #set device 
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    #load data 
    data_dir = './facekit_data'  #'./facekit_4'
    p = Path(data_dir)
    #load data     
    mean=127.5
    std=128
    #transform.ToTensor() will make the pixel values to be between [0, 1].
    transform = Compose([ transforms.ToTensor(),  transforms.Normalize(mean, std)])# normalize = (x-mean)/std
    facekit_data = Facekit_Processed(data_dir= p, transform= transform)#transforms.ToTensor()) #use this to pass the path as an argument (this means that the path is NOT specified in the class itself)
    data_length = facekit_data.__len__() 
    print(f'no. of objs in the entire dataset is:', data_length)

    #RetopologyNN/remesh.py
    # Split the dataset into training and validation sets
    train_size = int(0.8 *len(facekit_data))#0.4, 0.5, 0.8
    test_size = len(facekit_data) - train_size
    #generate a random number 
    gen = torch.Generator()
    gen.manual_seed(0) # this to make sure that the data is being splitted randomly in the same way each time we run the code 
    
    train_dataset, test_dataset = random_split(facekit_data, [train_size, test_size], generator=gen)#if we omit "generator=gen" then each time we run the code the random split will be different, and this will produce different training results 
    
    print('no of training samples:', len(train_dataset))
    print('train data:', train_dataset)
    #print('no of testing samples:', len(test_dataset))


    batch_size = 1
    train_loader = DataLoader(dataset=train_dataset, batch_size= batch_size, shuffle =False)
    #valid_loader = DataLoader(dataset=val_dataset, batch_size= batch_size, shuffle =True)
    test_loader = DataLoader(dataset=test_dataset, batch_size= batch_size, shuffle =False)

    #face_boxes = FaceBoxes() #to detect the face bounding box and landmarks 
    
    #param_mean = 127.5
    #param_std = 128

    #define model
    model = maha_mobilenet_v1.MobileNet(widen_factor=1.0, num_classes=48)
    model.to(device)
   
    criterion = nn.MSELoss()# the same as ((input-target)**2).mean()
    #criterion =  nn.CrossEntropyLoss()
    #criterion = NMELoss
    optimizer = optim.SGD(model.parameters(), lr=10e-5, momentum=0.9, weight_decay= 0.0005, nesterov=True)
    
    n_epochs = 200
    trainingEpoch_loss = []
    validationEpoch_loss = []
    n_total_steps = len(train_dataset)
    for epoch in range(n_epochs):  # loop over the dataset multiple times
        step_loss = []
        model.train()
        #get an image from the train loader
        for batch_index, data in enumerate(train_loader):#, 0):#loop over all the data one time
            #if batch_index ==0: #uncomment this line if you want to train and test one image only 
                [img_left, img_front, img_right], PointParam, label = data
                list_of_tensors = [img_left, img_front, img_right]
                input_tensor = torch.cat(list_of_tensors, dim=0) #concatenate tensors in the list along a new dimension called zero 
                input_tensor = input_tensor.to(device = device)

                PointParam = PointParam.to(device = device)
                
                #plot the range of pixels (we see that the range is between 0 and 1)
                #img_np = np.array(img_front)
                #plt.hist(img_np.ravel(), bins=50, density=True)
                #plt.xlabel("pixel values")
                #plt.ylabel("relative frequency")
                #plt.title("distribution of pixels") 
                #plt.savefig(f'distribution of pixels in image tensor after normalization.png')
                #exit()

             
                #load the image 
                
                #image_path = data_dir  + '/' + label[0]+'/'+label[0]+'_frontViewShape.jpg'
                #print(image_path)
                #img = cv2.imread(image_path)
                #boxes = face_boxes(img) 
                #n = len(boxes)
                #print(f'image: {image_path}, >>>>>> has {n} face(s)')
                #print('boxes: ', boxes)

                #input_param, roi_box_lst = preprocess_image(img, boxes)
                #input_param = input_param.to(device = device)

                #print(roi_box_lst)
                #print('input_param.shape', input_param.shape)

                #Clear the gradients to be all zeros 
                #optimizer.zero_grad()
                
                #forward pass
                param = model(input_tensor)
                #param = model([img_left, img_front, img_right])
                #print('pridected param shape', param.shape)
                #print('Ground truth param shape', PointParam.shape)

                #print('pridected param: ', param)
                #print('Ground truth param: ', PointParam)
                # calculating the loss between original and predicted data points
                loss = criterion(param, PointParam) #(prediected, target or ground truth )
                #diffrent kind of computing loss (needs)
                #reonstruct the vertices 
                #predicted_verts = convert param to obj #NOW
                #ground_truth_verts = convert param to obj 
                #loss = criterion(predicted_verts, ground_truth_verts) 
                #print("loss:", loss)
                step_loss.append(loss.item())
    
                #Clear the gradients to be all zeros 
                #optimizer.zero_grad()
                #backward pass to calculate gradients
                #loss.requires_grad = True
                loss.backward()
                #optimize means Update the weights
                optimizer.step()
                optimizer.zero_grad()
                #https://discuss.pytorch.org/t/how-to-draw-loss-per-epoch/22333
                #running_loss += param.shape[0] * loss.item()#The loss is averaged by the batch_size, which is the first dimension similar to (np.array(step_loss).mean()
                
                #type 2 of prinintng stats 
                #https://stackoverflow.com/questions/74754493/plot-training-and-validation-loss-in-pytorch
                if (batch_index+1) % 1 == 0:#print the loss for every image in the dataset
                    print (f'Epoch [{epoch+1}/{n_epochs}], Step [{batch_index+1}/{n_total_steps}], Loss: {loss.item():.4f}')
                
        #https://stackoverflow.com/questions/74754493/plot-training-and-validation-loss-in-pytorch
        trainingEpoch_loss.append(np.array(step_loss).mean())
        # print epoch loss
        #print(f'training loss after epoch #{epoch+1}: {trainingEpoch_loss / len(train_dataset)}')#Fix
        print(f'training loss after epoch #{epoch+1}: {np.array(step_loss).mean()}')
        #save the checkpoint 
        checkpoint(model, optimizer, f"trained_model_epoch_{epoch+1}_3views_facekit_3ddfa_v2.pth")
        
        #validation     
        print(f'validating the model now after epoch #{epoch+1}............... ')
        #test the model after finishing the current epoch 
        model.eval() # Optional when not using Model Specific layer
        for batch_index, data in enumerate(test_loader):#loop over all the data one time
            validationStep_loss = []

            [img_left, img_front, img_right],PointParam, label = data
            list_of_tensors = [img_left, img_front, img_right]
            input_tensor = torch.cat(list_of_tensors, dim=0)
            input_tensor = input_tensor.to(device = device)

            PointParam = PointParam.to(device = device)

            #image_path = './facekit_data/'+label[0]+'/'+label[0]+'_frontViewShape.jpg'
            #img = cv2.imread(image_path)
            #boxes = face_boxes(img) 
            #input_param, roi_box_lst = preprocess_image(img, boxes)
            #input_param = input_param.to(device = device)
            #with torch.no_grad():
            #forward pass
            param = model(input_tensor)
            #compute loss 
            validation_loss = criterion(param, PointParam) #(prediected model output, target or ground truth)
            validationStep_loss.append(validation_loss.item())
        print(f'validation accuracy after epoch #{epoch+1}: {np.array(validationStep_loss).mean()}')
        validationEpoch_loss.append(np.array(validationStep_loss).mean())
        #save the losses after every epoch
        with open(f'trainingEpoch_loss_epoch_{epoch+1}_3views_3DDFA_v2', "wb") as fp:   #Pickling
            pickle.dump(trainingEpoch_loss, fp)
        with open(f'validationEpoch_loss_epoch_{epoch+1}_3views_3DDFA_v2', "wb") as fp:   #Pickling
            pickle.dump(validationEpoch_loss, fp)

        #plot after every epoch 
        plt.plot(trainingEpoch_loss, label='train_loss')
        plt.savefig(f'training_curve_epoch_{epoch+1}.png')
        #plt.plot(validationEpoch_loss,label='val_loss')
        #plt.legend()
        plt.savefig(f'training_curves_3views_epoch_{epoch+1}.png')
        
    #print('**********Final Results**********')
    print('trainingEpoch_loss',trainingEpoch_loss)
    #print('validationEpoch_loss',validationEpoch_loss)
 

    print('Finished Training')



def test():
    image_path = 'examples/inputs/Face0000_frontViewShape.jpg'
    img = cv2.imread(image_path)
    face_boxes = FaceBoxes()
    boxes = face_boxes(img)
    n = len(boxes)
    print(f'image: {image_path}, >>>>>> has {n} face(s)')
    #reload the model
    #the_model = TheModelClass(*args, **kwargs)
    #the_model.load_state_dict(torch.load(PATH))
    #test the model 





if __name__ == "__main__":
    #test() #working
    #load_image() #working
    #train_synthesized_data() #working with synthesized data 
    #load_train_validate() #training loss is the same # train 1600 images 
    #load_train_validate_2() #training loss is the same # train a small subset of images 
    #load_train_validate_3() #ERROR in creating a tensor from  the dataset  at line 554 
    #train_processed_images()#imagers were ropped and resized
    train_processed_images_plot_after_every_epoch()#save the model, the losses, and the plots after every epoch 
    #get_mean_std() #not Working 





