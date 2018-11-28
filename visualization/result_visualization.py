
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
from optparse import OptionParser
import torch
from torchvision.utils import save_image

from inspect import getsourcefile
import os.path
import sys

current_path = os.path.abspath(getsourcefile(lambda:0))
current_dir = os.path.dirname(current_path)
parent_dir = current_dir[:current_dir.rfind(os.path.sep)]

sys.path.insert(0, parent_dir)

from model import UNet
from dataset import get_dataloader_show


'''
    Creates 4 plots with the information on the csv.
'''
def plot_one(dataframe, title, savedir):
    # Initiation
    epoch = dataframe.iloc[:, 0]
    prefix = ['Training', 'Testing']
    suffix = [['Loss', 'Loss'], ['Accuracy','Dice Index']]
    cont = 1
    fig = plt.figure(figsize=(10,8))
    fig.subplots_adjust(hspace=0.4, wspace=0.4)
    # Goes around the data in the csv
    for i in range(1,len(dataframe.columns)):
        index = 0 if cont%2 != 0 else 1
        pre = prefix[index]
        suf = suffix[index]

        ax = fig.add_subplot(2, 2, cont)
        ax.set_title(pre + " " + suf[0])
        ax.plot(epoch, dataframe.iloc[:, i])
        ax.set_ylabel(suf[1])

        cont += 2
        cont = 2 if cont > len(suffix)*2 else cont
    
    plt.suptitle(title)
    plt.savefig(savedir+title+'.png', dpi=199)


'''
    Reads all the runs and create a mean dataframe.
'''
def mean_results(folder):
    dataframes = [pd.read_csv(folder+names) for names in os.listdir(folder) if names.endswith(".csv")]
    values = dataframes[0]
    for df in dataframes[1:]:
        values = pd.concat((values, df))

    by_row_index = values.groupby(values.index)
    df_means = by_row_index.mean()

    return df_means


'''
    Pass an image through the net.
'''
def see_results(n_channels, n_classes, load_weights, dir_img, dir_cmp, savedir, title):
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
    loader = get_dataloader_show(dir_img, dir_cmp)

    # If savedir does not exists make folder
    if not os.path.exists(savedir):
        os.makedirs(savedir)

    net.eval()
    with torch.no_grad():
        for (data, gt) in loader:
            # Use GPU or not
            data, gt = data.to(device), gt.to(device)

            # Forward
            predictions = net(data)

            save_image(predictions, savedir+title+"_pred.png")
            save_image(gt, savedir+title+"_gt.png")


'''
    Definition of the optional and needed parameters.
'''
def get_args():
    parser = OptionParser()
    parser.add_option('-f', '--folder', dest='folder', default="./", 
                    help='Folder containing the csv files.') 
    parser.add_option('-t', '--title', dest='title', default="Results", 
                    help='Title of the plot')
    parser.add_option('-i', '--n-channels', dest='n_channels', default=1, type='int', 
                      help='Number of channels of the inputs.')
    parser.add_option('-c', '--n-classes', dest='n_classes', default=1, type='int', 
                      help='Number of classes of the output.')
    parser.add_option('-l', '--load', dest='load',
                      default=None, help='Path and name of the load file.')
    parser.add_option('-d', '--dataset', dest='dataset', default="original", 
                      choices=["original", "dist_trans", "one_class", "pre_0_3", "three_classes", "two_classes"], 
                      help='Dataset for the inputs.')
    parser.add_option('-g', '--gt', dest='gt', default="one_class", 
                      choices=["original", "dist_trans", "one_class", "pre_0_3", "three_classes", "two_classes"], 
                      help='Gt to compare.')

    (options, args) = parser.parse_args()
    return options


'''
    Runs the application.
'''
if __name__ == '__main__':
    args = get_args()
    df = mean_results(args.folder)
    plot_one(df, args.title, args.folder)
    see_results(n_channels = args.n_channels, 
        n_classes = args.n_classes, 
        load_weights = args.load, 
        dir_img = "./img_test/"+args.dataset+"/", 
        dir_cmp = "./img_test/"+args.gt+"/", 
        savedir = args.folder, 
        title = args.title)
