import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from sklearn.metrics import confusion_matrix
import sys
import pickle
import torchmetrics
from cnn_classifier import SimpleCNN
from classifier_module import Classifier
sys.path.insert(1, '../')
from utils.dataset import CLASSIFIER_PKL_DATASET
import pickle
import numpy as np

BATCH_SIZE = 128
DATASET_PATH = "../processing/dataset_path" # Add path to dataset
LOG_PATH = "./logs/"
NUM_WORKERS = 20
GENRES = ['Classic', 'Jazz']
EPOCHS = 20

def read_dataset(path, genres):
    dset_file = open(path+'dataset.pkl', 'rb')
    dataset = pickle.load(dset_file)
    dset_file.close()

    datasetA = dataset[genres[0]]
    datasetB = dataset[genres[1]]

    least_specs = min(len(datasetA), len(datasetB))

    dataset = []

    for i in range(least_specs):
        dataset.append((datasetA[i], 0))
        dataset.append((datasetB[i], 1))

    print(len(dataset))
    return dataset

def main():
    # Create the pytorch dataset and dataloader. Input transform 'torch.from_numpy' is required for data
    # preprocessed by 'processing.preprocess.Preprocess'.

    print('Saved in:', 'trained_classifiers/classifier_spec.pt')

    dataset = read_dataset(DATASET_PATH, GENRES)

    training_dataset = CLASSIFIER_PKL_DATASET(dataset, idx=(2250,12250))
    test_dataset = CLASSIFIER_PKL_DATASET(dataset, idx=(0, 2250))

    training_dataloader = DataLoader(training_dataset,
                                     batch_size=BATCH_SIZE,
                                     shuffle=True,
                                     num_workers=NUM_WORKERS)

    test_dataloader = DataLoader(test_dataset,
                                 batch_size=100000,
                                 shuffle=False,
                                 num_workers=NUM_WORKERS)

    # Get data shape
    shape = next(iter(training_dataloader))[0][0].shape
    print('\nShape of the data is: {} channels, ({}, {}) width and height.'.format(shape[0], shape[1],
                                                                                   shape[2]))

    # Create the classifier module of wanted type
    classifier = Classifier(SimpleCNN, num_channels = shape[0], input_shape = shape[1:], dropout_reg = 0.5)
    print('\nCreated classifier with {} convolutional layers.'.format(classifier.classifier.convs))

    # Create trainer then train the model
    print('\nLogs are stored in "{}".'.format(LOG_PATH))
    trainer = pl.Trainer(
            max_epochs=EPOCHS,
            logger=pl.loggers.TensorBoardLogger(save_dir=LOG_PATH),
            log_every_n_steps=1,
            gpus=-1,
            strategy='fsdp'
        )

    print('\nTraining...')
    trainer.fit(
        classifier,
        train_dataloaders=training_dataloader,
    )

    classifier.eval()
    test_data, labels = next(iter(test_dataloader))
    preds = torch.squeeze(torch.round(classifier(test_data)).detach(), dim=1)
    acc = torchmetrics.Accuracy()
    accuracy = acc(preds, labels)
    print('Test set accuracy:', accuracy)
    print(confusion_matrix(labels, preds.numpy()))

    torch.save(classifier.state_dict(), 'trained_classifiers/classifier.pt')

if __name__ == '__main__':
    main()