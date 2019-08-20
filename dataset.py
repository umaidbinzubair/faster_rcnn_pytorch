import pandas as pd
import torch.utils.data
import os
from PIL import Image
import cv2
import numpy as np
from torchvision import transforms as tvtsf
import utils
from skimage import transform as sktsf



class PennFudanDataset(torch.utils.data.Dataset):
    def __init__(self, root, transforms=None):
        self.root = root
        self.transforms = transforms
        self.imgs = list(sorted(os.listdir(os.path.join(root, "train"))))
        self.csv = os.path.join(self.root, "train.csv")

        # load all image files, sorting them to
        # ensure that they are aligned
        # self.imgs = list(sorted(os.listdir(os.path.join(root, "PNGImages"))))
        # self.masks = list(sorted(os.listdir(os.path.join(root, "PedMasks"))))
    
    def resize(self,image, bboxes,min_size= 600,max_size=1024):

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
        bboxes = self.adjust_bboxes(bboxes,old_height=height, old_width=width,new_height=new_height, new_width=new_width)
        
        return image,bboxes

    def adjust_bboxes(self,bboxes, old_height, old_width, new_height, new_width):
        """Adjusts the bboxes of an image that has been resized.
        Args:
            bboxes: Tensor with shape (num_bboxes, 5). Last element is the label.
            old_height: Float. Height of the original image.
            old_width: Float. Width of the original image.
            new_height: Float. Height of the image after resizing.
            new_width: Float. Width of the image after resizing.
        Returns:
            Tensor with shape (num_bboxes, 5), with the adjusted bboxes.
        """
        # We normalize bounding boxes points.
        # print(bboxes)
        x_min, y_min, x_max, y_max= np.split(np.asarray(bboxes),4,axis =1)
        x_min = x_min / old_width
        y_min = y_min / old_height
        x_max = x_max / old_width
        y_max = y_max / old_height

        # Use new size to scale back the bboxes points to absolute values.
        x_min = x_min * new_width
        y_min = y_min * new_height
        x_max = x_max * new_width
        y_max = y_max * new_height

        # Concat points and label to return a [num_bboxes, 5] tensor.
        out = np.stack([x_min,y_min,x_max,y_max],axis=1)
        out = np.squeeze(out)
        return out

    def __getitem__(self, idx):
        # load images ad masks
        file = self.imgs[idx]
        img_path = os.path.join(self.root, "train", file)

#         mask_path = os.path.join(self.root, "PedMasks", self.masks[idx])
        # Image.MAX_IMAGE_PIXELS = None
        # img = Image.open(img_path).convert("RGB")
        img = cv2.imread(img_path)
        a = img
        img = img.astype('float32')
        csv_file = pd.read_csv(self.csv)
        data_csv = csv_file.groupby('image_id')
        filenames = data_csv.groups
        # a = cv2.imread(img_path)
        for key in filenames.keys():

            if key == file:
                num_objs = 0
                boxes = []
                for i in filenames[key]:
                    num_objs +=1
                    xmin = csv_file.iloc[i][1]
                    ymin = csv_file.iloc[i][2]
                    xmax = csv_file.iloc[i][3]
                    ymax = csv_file.iloc[i][4]
                    boxes.append([xmin, ymin, xmax, ymax])
                    # cv2.rectangle(a, (xmin, ymin), (xmax, ymax), (0,0,255), 3)
        # cv2.imwrite(file, a)
        img,boxes = self.resize(img,boxes)
        # exit(0)
        img = np.asarray(img).astype('float32')
        img = img/255.
        img = img.transpose((2,0,1))
        # Normalizing image
        normalize = tvtsf.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
        img = normalize(torch.from_numpy(img))
        img = img.numpy()
        # img = img.transpose((1,2,0))       
        
        # print(boxes)
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)
        # there is only one class
        labels = torch.ones((num_objs,), dtype=torch.int64)
        # masks = torch.as_tensor(masks, dtype=torch.uint8)

        image_id = torch.tensor([idx])
        boxes = boxes.reshape(-1,4)
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        area = torch.as_tensor(area,dtype=torch.int64)
        boxes = torch.as_tensor(boxes, dtype=torch.int64)

        # suppose all instances are not crowd
        # iscrowd = torch.zeros((num_objs,), dtype=torch.int64)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
#         target["masks"] = masks
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd

        if self.transforms is not None:
            img, target = self.transforms(img, target)
        return img, target

    def __len__(self):
        return len(self.imgs)


# class PennFudanDataset(torch.utils.data.Dataset):
#     def __init__(self, root, transforms=None):
#         self.root = root
#         self.transforms = transforms
#         # load all image files
#         self.imgs = list(sorted(os.listdir(os.path.join(root, 'train'))))

#     def __getitem__(self, idx):
#         img_path = os.path.join(self.root,'train', self.imgs[idx])
#         # image = Image.open(img_path).convert("RGB")
#         image = utils.read_image(img_path)

#         # Rescaling Images
#         C, H, W = image.shape
#         min_size = 600
#         max_size = 1024
#         scale1 = min_size / min(H, W)
#         scale2 = max_size / max(H, W)
#         scale = min(scale1, scale2)
#         image = image / 255.
#         image = sktsf.resize(image, (C, H * scale, W * scale), mode='reflect')

#         # Normalizing image
#         normalize = tvtsf.Normalize(mean=[0.485, 0.456, 0.406],
#                                 std=[0.229, 0.224, 0.225])
#         image = normalize(torch.from_numpy(image))
#         image = image.numpy()

#         # if not os.path.isfile(os.path.join(self.root, 'images')):
#         #     print('[error]',' file not found')
#         #     exit(0)

#         train_labels = pd.read_csv(os.path.join(self.root, 'train.csv'))

#         old_boxes = []
#         num_objs = 0
#         for i in range(train_labels['image_id'].count()):
#             if (self.imgs[idx] == train_labels['image_id'][i]):
#                 xmin = train_labels['xmin'][i]
#                 ymin = train_labels['ymin'][i]
#                 xmax = train_labels['xmax'][i]
#                 ymax = train_labels['ymax'][i]
#                 num_objs += 1
#                 old_boxes.append([xmin, ymin, xmax, ymax])
    
#         # Rescale bounding box
#         _, o_H, o_W = image.shape
#         scale = o_H / H
#         bbox = np.stack(old_boxes).astype(np.float32)
#         resized_boxes = utils.resize_bbox(bbox, (H, W), (o_H, o_W))
        
#         # resized boxes are stacked (R, 4) 
#         # where R is the number of bboxes in the image 
#         # converted it back to simple 2d-array [[xmin1, ymin1, xmax1, ymax1], ...]
#         boxes = []
#         for i in resized_boxes:
#             box = []
#             [box.append(int(b)) for b in i]
#             boxes.append(box)

#         # converting arrays into torch tensors
#         boxes = torch.as_tensor(boxes, dtype=torch.float32)

#         # there is only one class
#         labels = torch.ones((num_objs,), dtype=torch.int64)

#         image_id = torch.tensor([idx])
#         area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
#         # suppose all instances are not crowd
#         iscrowd = torch.zeros((num_objs,), dtype=torch.int64)

#         target = {}
#         target["boxes"] = boxes
#         target["labels"] = labels
#         target["image_id"] = image_id
#         target["area"] = area
#         target["iscrowd"] = iscrowd

#         image = image.transpose((1,2,0))

#         if self.transforms is not None:
#             image, target = self.transforms(image, target)

#         return image, target

#     def __len__(self):
#         return len(self.imgs)
#     