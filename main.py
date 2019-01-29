import torch
import torch.nn as nn
import time
from optparse import OptionParser

from model import UNet
from dataset import get_dataloaders
from train_val import DiceLoss, RMSELoss, CELoss, WCELoss, train_net #, val_net
from misc import export_history, save_checkpoint

'''
    Configure every aspect of the run.
    Runs the training and validation.
'''
def setup_and_run_train(n_channels, n_classes, dir_img, dir_gt, dir_results, load, 
                val_perc, batch_size, epochs, lr, run, optimizer, loss, evaluation, dir_weights):
    
    # Use GPU or not
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    # Create the model
    net = UNet(n_channels, n_classes).to(device)
    net = torch.nn.DataParallel(net, device_ids=list(
        range(torch.cuda.device_count()))).to(device)

    # Load old weights
    if load:
        net.load_state_dict(torch.load(load))
        print('Model loaded from {}'.format(load))

    # Load the dataset
    if loss != "WCE":
        train_loader, val_loader = get_dataloaders(dir_img, dir_gt, val_perc, batch_size)
    else:
        train_loader, val_loader = get_dataloaders(dir_img, dir_gt, val_perc, batch_size, isWCE = True, dir_weights = dir_weights)

    # Pretty print of the run
    print('''\n
    Starting training:
        Dataset: {}
        Num Channels: {}
        Groundtruth: {}
        Num Classes: {}
        Folder to save: {}
        Load previous: {}
        Training size: {}
        Validation size: {}
        Validation Percentage: {}
        Batch size: {}
        Epochs: {}
        Learning rate: {}
        Optimizer: {}
        Loss Function: {}
        Evaluation Function: {}
        CUDA: {}
    '''.format(dir_img, n_channels, dir_gt, n_classes, dir_results, load, 
            len(train_loader)*batch_size, len(val_loader)*batch_size, 
            val_perc, batch_size, epochs, lr, optimizer, loss, evaluation, use_cuda))

    # Definition of the optimizer ADD MORE IF YOU WANT
    if optimizer == "Adam":
        optimizer = torch.optim.Adam(net.parameters(),
                             lr=lr)
    elif optimizer == "SGD":
        optimizer = torch.optim.SGD(net.parameters(),
                        lr=lr,
                        momentum=0.9,
                        weight_decay=0.0005)

    # Definition of the loss function ADD MORE IF YOU WANT
    if loss == "Dice":
        criterion = DiceLoss()
    elif loss == "RMSE":
        criterion = RMSELoss()
    elif loss == "MSE":
        criterion = nn.MSELoss()
    elif loss == "MAE":
        criterion = nn.L1Loss()
    elif loss == "CE":
        criterion = CELoss()
    elif loss == "WCE":
        criterion = WCELoss()

    # Saving History to csv
    header = ['epoch', 'train loss']

    best_loss = 10000
    time_start = time.time()
    # Run the training and validation
    for epoch in range(epochs):
        print('\nStarting epoch {}/{}.'.format(epoch + 1, epochs))

        train_loss = train_net(net, device, train_loader, optimizer, criterion, batch_size, isWCE = (loss == "WCE"))
        #val_loss = val_net(net, device, val_loader, criterion_val, batch_size)
        
        values = [epoch+1, train_loss]
        export_history(header, values, dir_results, "result"+run+".csv")
        
        # save model
        if train_loss < best_loss:
            best_loss = train_loss
            save_checkpoint({
                    'epoch': epoch + 1,
                    'state_dict': net.state_dict(),
                    'loss': train_loss,
                    'optimizer' : optimizer.state_dict(),
                }, path=dir_results, filename="weights"+run+".pth")

    time_dif = time.time() - time_start
    print("It tooks %.4f seconds to finish the run." % (time_dif))


'''
    Definition of the optional and needed parameters.
'''
def get_args():
    parser = OptionParser()
    parser.add_option('-e', '--epochs', dest='epochs', default=30, type='int',
                      help='number of epochs')
    parser.add_option('-b', '--batch-size', dest='batchsize', default=25,
                      type='int', help='batch size')
    parser.add_option('-l', '--learning-rate', dest='lr', default=0.0001,
                      type='float', help='learning rate')
    parser.add_option('-a', '--load', dest='load',
                      default=False, help='load file model')
    parser.add_option('-r', '--runs', dest='runs', type='int',
                      default=1, help='How many runs') 
    parser.add_option('-d', '--dataset', dest='dataset',
                      default='Data', help='Which dataset should use.') 
    parser.add_option('-g', '--groundtruth', dest='gt',
                      default='GT_One_Class', help='Which gt should use.') 
    parser.add_option('-s', '--savedir', dest='savedir',
                      default='checkpoints/', help='Which folder should use for checkpoints.') 
    parser.add_option('-t', '--val-percentage', dest='val_perc', default=0.3,type='float', 
                      help='Validation Percentage')
    parser.add_option('-i', '--n-channels', dest='n_channels', default=1, type='int', 
                      help='Number of channels of the inputs.')
    parser.add_option('-c', '--n-classes', dest='n_classes', default=1, type='int', 
                      help='Number of classes of the output.')                  
    parser.add_option('-o', '--optimizer', dest='optimizer', default="Adam", choices=["Adam", "SGD"], 
                      help='Optimizer to use.')
    parser.add_option('-f', '--loss', dest='loss', default="Dice", choices=["Dice", "RMSE", "MSE", "MAE", "CE", "WCE"], 
                      help='Loss functios to use.')
    parser.add_option('-v', '--evaluation', dest='evaluation', default="Dice", choices=["Dice", "RMSE", "MSE", "MAE", "CE", "WCE"], 
                      help='Evaluation function to use.')
    parser.add_option('-w', '--weights', dest='weights',
                      default='', help='Which weights should use.')

    (options, args) = parser.parse_args()
    return options


'''
    Runs the application.
'''
if __name__ == "__main__":
    args = get_args()
    for r in range(args.runs):
        setup_and_run_train(
                n_channels = args.n_channels, 
                n_classes = args.n_classes,
                dir_img = args.dataset,
                dir_gt = args.gt,
                dir_results = args.savedir,
                load = args.load,
                val_perc = args.val_perc,
                batch_size = args.batchsize,
                epochs = args.epochs,
                lr = args.lr,
                run=str(r),
                optimizer = args.optimizer,
                loss = args.loss,
                evaluation = args.evaluation,
                dir_weights = args.weights)
                        


