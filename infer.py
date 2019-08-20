import numpy as np
import torch
from dataset import PennFudanDataset as data_set
from engine import train_one_epoch, evaluate
from model import *
import utils
import transforms as T
import cv2
from skimage import transform as sktsf


def get_transform(train):
    transforms = []
    transforms.append(T.ToTensor())
    return T.Compose(transforms)

def resize(self,image,min_size= 600,max_size=1024):

        height,width, _ = image.shape
        # width,height = image.size 

        min_dimension = min(height, width)
        upscale_factor = max(min_size / min_dimension, 1.)
        max_dimension = max(height, width)
        downscale_factor = min(max_size / max_dimension, 1.)
        scale_factor = upscale_factor * downscale_factor

        new_height = height * scale_factor
        new_width = width * scale_factor


        dim = (int(new_width),int(new_height))
        # image = image.resize(dim)

        image = cv2.resize(image, dim)
        return image

path = '/home/haroonrashid/Table_Identification/AsadsWorkspace/torch/Table-Detection-PyTorch/data'
dataset_test = data_set(path, get_transform(train=False))
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
num_classes = 2
# model = get_instance_segmentation_model(num_classes)
# model.load_state_dict(torch.load('/home/haroonrashid/Umaid/checkpoints/checkpoint_5.pth'))
# # move model to the right device
# model.to(device)
# print('Model loaded')
# # evaluation mode ON
# model.eval()

img, _ = dataset_test[0]
img = resize(img)
img = np.asarray(img).astype('float32')
img = img/255.
img = img.transpose((2,0,1))
# Normalizing image
normalize = tvtsf.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
img = normalize(torch.from_numpy(img))
img = img.numpy()
img = img.transpose((1,2,0))
# put the model in evaluation mode
# model.eval()
# with torch.no_grad():
#     prediction = model([img.to(device)])
# img = img.numpy()
# img = img.transpose(2,1,0)
cv2.imwrite('/home/haroonrashid/Umaid/checkpoints/img.jpg',img)
