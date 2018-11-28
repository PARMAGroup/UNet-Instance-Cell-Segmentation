import time

import torch.nn as nn
import torch

from misc import AverageMeter


""" 
    Class that defines the Dice Loss function.
"""
class DiceLoss(nn.Module):
  
    def __init__(self, smooth = 1):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def dice_coef(self, y_pred, y_true):
        pred_probs = torch.sigmoid(y_pred)
        y_true_f = y_true.view(-1)
        y_pred_f = pred_probs.view(-1)
        intersection = torch.sum(y_true_f * y_pred_f)
        return (2. * intersection + self.smooth) / (torch.sum(y_true_f) + torch.sum(y_pred_f) + self.smooth)
  
    def forward(self, y_pred, y_true):
        return -self.dice_coef(y_pred, y_true)


""" 
    Functions that trains a net.
"""
def train_net(net, device, loader, optimizer, criterion, batch_size):
    net.train()
    train_loss = AverageMeter()
    train_acc = AverageMeter()
    time_start = time.time()
    for batch_idx, (data, gt) in enumerate(loader):

        # Use GPU or not
        data, gt = data.to(device), gt.to(device)

        # Forward
        predictions = net(data)

        # Loss Calculation
        loss = criterion(predictions, gt)
        
        # Updates the record
        train_loss.update(loss.item(), predictions.size(0))
        train_acc.update(-loss.item(), predictions.size(0))
        
        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print('[{}/{} ({:.0f}%)]\t\tLoss: {:.6f}'.format(
            batch_idx * len(data), len(loader)*batch_size,
            100. * batch_idx / len(loader), loss.item()))

    time_dif = time.time() - time_start
    print('\nAverage Training Loss: ' + str(train_loss.avg))
    print('\nAverage Training Accuracy: ' + str(train_acc.avg))
    print('Train Time: It tooks %.4fs to finish the epoch.' % (time_dif))
            
    return train_loss.avg, train_acc.avg


""" 
    Function that validates the net.
"""
def val_net(net, device, loader, criterion, batch_size):
    net.eval()
    val_loss = AverageMeter()
    val_acc = AverageMeter()
    time_start = time.time()
    with torch.no_grad():
        for batch_idx, (data, gt) in enumerate(loader):

            # Use GPU or not
            data, gt = data.to(device), gt.to(device)

            # Forward
            predictions = net(data)
            
            # Loss Calculation
            loss = criterion(predictions, gt)

            # Updates the record
            val_loss.update(loss.item(), predictions.size(0))
            val_acc.update(-loss.item(), predictions.size(0))
            
            print('[{}/{} ({:.0f}%)]\t\tLoss: {:.6f}'.format(
                batch_idx * len(data), len(loader)*batch_size,
                100. * batch_idx / len(loader), loss.item()))
    
    time_dif = time.time() - time_start
    print('\nValidation set: Average loss: '+ str(val_loss.avg))
    print('\nAverage Validation Accuracy: ' + str(val_acc.avg))
    print('Validation time: It tooks %.4fs to finish the Validation.' % (time_dif))
    
    return val_loss.avg, val_acc.avg