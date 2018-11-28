# UNet for Instance Cell Segmentation on Pytorch

This project aims to perform well at instance segmentation on the BBBC006 cells dataset. We tested UNet over several configurations including the loss function, evaluation function and the datasets.

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.

### Prerequisites

- [Python 3.7](https://www.python.org/downloads/)
- [Pytorch](https://pytorch.org/)
- [BBBC006 Dataset with z=16](https://data.broadinstitute.org/bbbc/BBBC006/)

## Running the Training and Validation

For running the training you will use the [main.py](main.py) file. In case you're using a sort of cluster, you should see [pbs_unet.pbs](pbs_unet.pbs). In either case, the options for configuration are the same.

### Options for configuration

```
Options:
  -h, --help            Show this help message and exit
  -e EPOCHS, --epochs=EPOCHS
                        Number of epochs
  -b BATCHSIZE, --batch-size=BATCHSIZE
                        Batch size
  -l LR, --learning-rate=LR
                        Learning rate
  -a LOAD, --load=LOAD  Load model file
  -r RUNS, --runs=RUNS  How many runs
  -d DATASET, --dataset=DATASET
                        Which dataset should use.
  -g GT, --groundtruth=GT
                        Which gt should use.
  -s SAVEDIR, --savedir=SAVEDIR
                        Which folder should use for checkpoints.
  -t VAL_PERC, --val-percentage=VAL_PERC
                        Validation Percentage
  -i N_CHANNELS, --n-channels=N_CHANNELS
                        Number of channels of the inputs.
  -c N_CLASSES, --n-classes=N_CLASSES
                        Number of classes of the output.
  -o OPTIMIZER, --optimizer=OPTIMIZER
                        Optimizer to use.
  -f LOSS, --loss=LOSS  Loss functios to use.
  -v EVALUATION, --evaluation=EVALUATION
                        Evaluation function to use.
```

If you want to use another optimizer or loss/evaluation function you must code it on [main.py](main.py)

## Running the Visualization of Results

For running the visualizations you will use the **visualization** folder.

### Show Visualization

If you want to see how a trained model performs out, you should use the [result_visualization.py](visualization/result_visualization.py) file.
These report will create an image of the performance (Loss, Accuracy), an image of the outputs of the model and an image of the gt to compare.

#### Options for configuration

```
Options:
  -h, --help            Show this help message and exit
  -f FOLDER, --folder=FOLDER
                        Folder containing the csv files.
  -t TITLE, --title=TITLE
                        Title of the plot
  -i N_CHANNELS, --n-channels=N_CHANNELS
                        Number of channels of the inputs.
  -c N_CLASSES, --n-classes=N_CLASSES
                        Number of classes of the output.
  -l LOAD, --load=LOAD  Path and name of the load file.
  -d DATASET, --dataset=DATASET
                        Dataset for the inputs.
  -g GT, --gt=GT        Gt to compare.
```

### Compare two Trained Models

If you want to show how the model performs out compared with another one, you should use the [compare_two_results.py](visualization/compare_two_results.py) file.
These report will create an image of the performance (Loss, Accuracy) of both models.

#### Options for configuration

```
Options:
  -h, --help            Show this help message and exit
  -f FOLDER, --folder=FOLDER
                        Folder containing the first to compare csv files.
  -c F_COMPARE, --folder_compare=F_COMPARE
                        Folder containing the second to compare csv files.
  -t TITLE, --title=TITLE
                        Title of the plot
  -s SAVEDIR, --savedir=SAVEDIR
                        Which folder should use for saving the results.
```

## Authors of the Code

* **Mauro Méndez** - [mamemo](https://github.com/mamemo)

See also the list of [contributors](https://github.com/your/project/contributors) who participated in this project.

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details

## Acknowledgments

* All of the UNet implementations that cross into my way.
* Jose Carranza - [maeotaku](https://github.com/maeotaku) - for helping in the misc files.
