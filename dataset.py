import os
from PIL import Image
import numpy as np

from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.utils.data.sampler import SubsetRandomSampler

'''
    Class that defines the reading and processing of the images.
    Specialized on BBBC006 dataset.
'''
class BBBCDataset(Dataset):
    def __init__(self, ids, dir_data, dir_gt, extension='.png', isWCE=False, dir_weights = ''):

        self.dir_data = dir_data
        self.dir_gt = dir_gt
        self.extension = extension
        self.isWCE = isWCE
        self.dir_weights = dir_weights

        # Transforms
        self.data_transforms = {
            'imgs': transforms.Compose([
#                 transforms.RandomResizedCrop(256),
#                 transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.0054],[0.0037])
            ]),
            'masks': transforms.Compose([
                transforms.ToTensor()
            ]),
        }

        # Images IDS
        self.ids = ids

        # Calculate len of data
        self.data_len = len(self.ids)

    '''
        Ask for an image.
    '''
    def __getitem__(self, index):
        # Get an ID of a specific image
        id_img = self.dir_data + self.ids[index] + self.extension
        id_gt = self.dir_gt + self.ids[index] + self.extension
        # Open Image and GroundTruth
        img = Image.open(id_img).convert('L')
        gt = Image.open(id_gt)
        # Applies transformations
        img = self.data_transforms['imgs'](img)
        gt = self.data_transforms['masks'](gt)
        if self.isWCE:
            id_weight = self.dir_weights + self.ids[index] + self.extension
            weight = Image.open(id_weight).convert('L')
            weight = self.data_transforms['masks'](weight)
            return (img, gt, weight)

        return (img, gt, gt)

    '''
        Length of the dataset.
        It's needed for the upper class.
    '''
    def __len__(self):
        return self.data_len


'''
    Returns the dataset separated in batches.
    Used inside every epoch for retrieving the images.
'''
def get_dataloaders(dir_img, dir_gt, test_percent=0.2, batch_size=10, isWCE = False, dir_weights=''):
    # Validate a correct percentage
    test_percent = test_percent/100 if test_percent > 1 else test_percent
    # Read the names of the images
    ids = [f[:-4] for f in os.listdir(dir_img)]
    # Creates the dataset
    if not isWCE:
        dset = BBBCDataset(ids, dir_img, dir_gt)
    else:
        dset = BBBCDataset(ids, dir_img, dir_gt, isWCE = isWCE, dir_weights = dir_weights)
    
    # Calculate separation index for training and validation
    num_train = len(dset)
    indices = list(range(num_train))
    split = int(np.floor(test_percent * num_train))
    np.random.shuffle(indices)
    train_idx, valid_idx = indices[split:], indices[:split]

    # Create the dataloaders
    dataloaders = {}
    dataloaders['train'] = DataLoader(dset, batch_size=batch_size,
                                               sampler=SubsetRandomSampler(train_idx))
    dataloaders['val'] = DataLoader(dset, batch_size=batch_size,
                                                   sampler=SubsetRandomSampler(valid_idx))
   
    return dataloaders['train'], dataloaders['val']


'''
    Returns few images for showing the results.
'''
def get_dataloader_show(dir_img, dir_gt):
    # Read the names of the images
    ids = [f[:-4] for f in os.listdir(dir_img)]
    # Creates the dataset
    dset = BBBCDataset(ids, dir_img, dir_gt)

    # Create the dataloader
    dataloader = DataLoader(dset, batch_size=len(ids))
   
    return dataloader

'''
    Class that defines the reading and processing of the images.
    Specialized on BBBC006 dataset.
'''
class BBBCDataset_Transform(Dataset):
    def __init__(self, ids, dir_data, extension='.png'):

        self.dir_data = dir_data
        self.extension = extension

        # Transforms
        self.data_transforms = {
            'imgs': transforms.Compose([
#                 transforms.RandomResizedCrop(256),
#                 transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.0054],[0.0037])
            ]),
            'masks': transforms.Compose([
                transforms.ToTensor()
            ]),
        }

        # Images IDS
        self.ids = ids

        # Calculate len of data
        self.data_len = len(self.ids)

    '''
        Ask for an image.
    '''
    def __getitem__(self, index):
        # Get an ID of a specific image
        id_img = self.dir_data + self.ids[index] + self.extension
        # Open Image and GroundTruth
        img = Image.open(id_img).convert('L')
        # Applies transformations
        img = self.data_transforms['imgs'](img)
        return (img, self.ids[index]+self.extension)

    '''
        Length of the dataset.
        It's needed for the upper class.
    '''
    def __len__(self):
        return self.data_len

'''
    Returns whole dataset to transform.
'''
def get_dataloader_transform(dir_img, batch_size = 1):
    # Read the names of the images
    ids = [f[:-4] for f in os.listdir(dir_img)]
    # Creates the dataset
    dset = BBBCDataset_Transform(ids, dir_img)

    # Create the dataloader
    dataloader = DataLoader(dset, batch_size=batch_size)

    return dataloader
