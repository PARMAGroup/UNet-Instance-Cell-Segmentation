
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
from optparse import OptionParser
import torch
from torchvision.utils import save_image
import torch.nn.functional as F

from inspect import getsourcefile
import os.path
import sys

current_path = os.path.abspath(getsourcefile(lambda:0))
current_dir = os.path.dirname(current_path)
parent_dir = current_dir[:current_dir.rfind(os.path.sep)]

sys.path.insert(0, parent_dir)

from model import UNet
from dataset import get_dataloader_transform


'''
    Pass an image through the net.
'''
def transform(n_channels, n_classes, load_weights, dir_img, savedir):
    # Use GPU or not
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    # Create the model
    net = UNet(n_channels, n_classes).to(device)
    net = torch.nn.DataParallel(net, device_ids=list(
        range(torch.cuda.device_count()))).to(device)

    # Load old weights
    checkpoint = torch.load(load_weights, map_location='cpu')
    net.load_state_dict(checkpoint['state_dict'])

    # Load the dataset
    loader = get_dataloader_transform(dir_img)

    # If savedir does not exists make folder
    if not os.path.exists(savedir):
        os.makedirs(savedir)

    net.eval()
    with torch.no_grad():
        for (data, name) in loader:
            # Use GPU or not
            data = data.to(device)

            # Forward
            predictions = net(data)
            predictions = F.softmax(predictions)
            predictions = torch.round(F.softmax(predictions))
            save_image(predictions, savedir+name[0])


'''
    Definition of the optional and needed parameters.
'''
def get_args():
    parser = OptionParser()
    parser.add_option('-f', '--folder', dest='folder', default="./", 
                    help='Folder containing the csv files.') 
    parser.add_option('-i', '--n-channels', dest='n_channels', default=1, type='int', 
                      help='Number of channels of the inputs.')
    parser.add_option('-c', '--n-classes', dest='n_classes', default=1, type='int', 
                      help='Number of classes of the output.')
    parser.add_option('-l', '--load', dest='load',
                      default=None, help='Path and name of the load file.')
    parser.add_option('-d', '--dataset', dest='dataset', 
                      help='Path of the Dataset for the inputs.')

    (options, args) = parser.parse_args()
    return options


'''
    Runs the application.
'''
if __name__ == '__main__':
    args = get_args()
    transform(n_channels = args.n_channels, 
        n_classes = args.n_classes, 
        load_weights = args.load, 
        dir_img = args.dataset, 
        savedir = args.folder)
