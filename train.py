import numpy as np
import torch
from dataset_custom import PennFudanDataset as data_set
from engine import train_one_epoch, evaluate
from model import *
import utils
import transforms as T


def get_transform(train):
    transforms = []
    # converts the image, a PIL image, into a PyTorch Tensor
    transforms.append(T.ToTensor())
    # if train:
    #     # during training, randomly flip the training images
    #     # and ground-truth for data augmentation
    #     transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)

path = '/home/haroonrashid/Umaid/img'
dataset = data_set(path, get_transform(train = True))
dataset_test = data_set(path, get_transform(train=True))
# test = data_set('/home/ml/UmaidWorkstation/train_rcnn', get_transform(train=False))
# split the dataset in train and test set
torch.manual_seed(1)
indices = torch.randperm(len(dataset)).tolist()
dataset = torch.utils.data.Subset(dataset, indices)
dataset_test = torch.utils.data.Subset(dataset_test, indices[-50:])
# test = torch.utils.data.Subset(dataset_test, indices[-11:-13])


# define training and validation data loaders
data_loader = torch.utils.data.DataLoader(
    dataset, batch_size=1, shuffle=True, num_workers=2,
    collate_fn=utils.collate_fn)

data_loader_test = torch.utils.data.DataLoader(
    dataset_test, batch_size=1, shuffle=False, num_workers=4,
    collate_fn=utils.collate_fn)

# dtest = torch.utils.data.DataLoader(
#     test, batch_size=1, shuffle=False, num_workers=4,
#     collate_fn=utils.collate_fn)

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


# our dataset has two classes only - background and table
num_classes = 2

# get the model using our helper function
model = get_instance_segmentation_model(num_classes)
# move model to the right device
model.to(device)
print('Model loaded')

# construct an optimizer
# for p in model.parameters():
#     print(p.requires_grad)
# assert False
params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(params, lr=0.005,
                            momentum=0.9, weight_decay=0.0005)
print('optimizer ready')
# optimizer =torch.optim.Adam(params, lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)

# and a learning rate scheduler which decreases the learning rate by
# 10x every 3 epochs
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                               step_size=5,
                                               gamma=0.1)

"""And now let's train the model for 10 epochs, evaluating at the end of every epoch."""

# let's train it for 10 epochs
num_epochs = 200
print('Start training')
for epoch in range(num_epochs):
    # train for one epoch, printing every 10 iterations
    train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=5)
    print('Epoch number '+str(epoch+1)+' done')
    # update the learning rate
    lr_scheduler.step()
    # evaluate on the test dataset
    # evaluate(model, dtest, device=device)
    if epoch % 5 == 0:
        torch.save(model.state_dict(), 'work/model_epoch_'+str(epoch)+'.pth')

        # torch.save(model.state_dict(), 'checkpoints/checkpoint_'+str(epoch)+'.pth')





# """Now that training has finished, let's have a look at what it actually predicts in a test image"""

# # pick one image from the test set
# img, _ = dataset_test[0]
# # put the model in evaluation mode
# model.eval()
# with torch.no_grad():
#     prediction = model([img.to(device)])

# """Printing the prediction shows that we have a list of dictionaries. Each element of the list corresponds to a different image. As we have a single image, there is a single dictionary in the list.
# The dictionary contains the predictions for the image we passed. In this case, we can see that it contains `boxes`, `labels`, `masks` and `scores` as fields.
# """

# print(prediction)

"""Let's inspect the image and the predicted segmentation masks.

For that, we need to convert the image, which has been rescaled to 0-1 and had the channels flipped so that we have it in `[C, H, W]` format.
"""

# Image.fromarray(img.mul(255).permute(1, 2, 0).byte().numpy())

# """And let's now visualize the top predicted segmentation mask. The masks are predicted as `[N, 1, H, W]`, where `N` is the number of predictions, and are probability maps between 0-1."""

# Image.fromarray(prediction[0]['masks'][0, 0].mul(255).byte().cpu().numpy())