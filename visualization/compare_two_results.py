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
    Creates 4 plots with the information on two csv's.
'''
def plot_two(dataframe1, dataframe2, name1, name2, title, savedir):
    # Initiation
    epoch = dataframe1.iloc[:, 0]
    prefix = ['Training', 'Testing']
    suffix = [['Loss', 'Loss'], ['Accuracy','Dice Index']]
    cont = 1
    fig = plt.figure(figsize=(10,8))
    fig.subplots_adjust(hspace=0.4, wspace=0.4)
    # Goes around the data in the csv
    for i in range(1,len(dataframe1.columns)):
        index = 0 if cont%2 != 0 else 1
        pre = prefix[index]
        suf = suffix[index]

        ax = fig.add_subplot(2, 2, cont)
        ax.set_title(pre + " " + suf[0])
        ax.plot(epoch, dataframe1.iloc[:, i],"b",epoch, dataframe2.iloc[:, i],"r")
        ax.set_ylabel(suf[1])
        ax.legend([name1, name2])

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
    Definition of the optional and needed parameters.
'''
def get_args():
    parser = OptionParser()
    parser.add_option('-f', '--folder', dest='folder', default="./", 
                    help='Folder containing the first to compare csv files.') 
    parser.add_option('-c', '--folder_compare', dest='f_compare', default="./", 
                    help='Folder containing the second to compare csv files.') 
    parser.add_option('-t', '--title', dest='title', default="Results", 
                    help='Title of the plot')
    parser.add_option('-s', '--savedir', dest='savedir',
                      default='./', help='Which folder should use for saving the results.') 


    (options, args) = parser.parse_args()
    return options


'''
    Runs the application.
'''
if __name__ == '__main__':
    args = get_args()
    df1 = mean_results(args.folder)
    name1 = args.folder.split("\\")[-2]
    df2 = mean_results(args.f_compare)
    name2 = args.f_compare.split("\\")[-2]
    plot_two(df1, df2, name1, name2, args.title, args.savedir)
