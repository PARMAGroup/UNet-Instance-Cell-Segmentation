import os
import csv
import torch

""" 
    Export data to csv format. Creates new file if doesn't exist,
    otherwise update it.
    Args:
        header (list): headers of the column
        value (list): values of correspoding column
        folder: folder path
        file_name: file name with path
"""
def export_history(header, value, folder, file_name):
    # If folder does not exists make folder
    if not os.path.exists(folder):
        os.makedirs(folder)

    file_path = folder + file_name
    file_existence = os.path.isfile(file_path)

    # If there is no file make file
    if file_existence == False:
        file = open(file_path, 'w', newline='')
        writer = csv.writer(file)
        writer.writerow(header)
        writer.writerow(value)
    # If there is file overwrite
    else:
        file = open(file_path, 'a', newline='')
        writer = csv.writer(file)
        writer.writerow(value)
    # Close file when it is done with writing
    file.close()


""" 
    Save the state of a net.
"""
def save_checkpoint(state, path='checkpoint/', filename='weights.pth'):
    # If folder does not exists make folder
    if not os.path.exists(path):
        os.makedirs(path)

    filepath = os.path.join(path, filename)
    torch.save(state, filepath)


""" 
    Computes and stores the average and current value
    Imported from https://github.com/pytorch/examples/blob/master/imagenet/main.py#L247-L262
"""  
class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
