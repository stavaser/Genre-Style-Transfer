import torch
from torch import nn
from torch.nn import functional as F
from collections import OrderedDict
import torchmetrics
import pytorch_lightning as pl

import sys
sys.path.insert(1, '../')

class Classifier(pl.LightningModule):
    """Universal classifier module

    Input any of the implemented classifiers with their required parameters.

    Attributes
    ----------
    classifier : object
        Any classifier class defined in './classifiers/'.
    acc : torchmetrics.Accuracy()
        Accuracy metric.
    """
    def __init__(self,
                classifier,
                lr: float = 0.000333,
                **kwargs
    ):
        super().__init__()
        self.save_hyperparameters()

        self.classifier = classifier(**kwargs)
        self.acc = torchmetrics.Accuracy()

    def forward(self, img):
        return self.classifier(img)

    def BCELoss(self, y_hat, y):
        return F.binary_cross_entropy(y_hat, y)

    def training_step(self, batch):
        segment, target = batch

        classified_segments = self(segment)
        loss = self.BCELoss(classified_segments, target.unsqueeze(1).float())
        accuracy = self.acc(torch.round(classified_segments), target.unsqueeze(1))

        tqdm_dict = {"loss": loss.detach()}
        output = OrderedDict({"loss": loss, "progress_bar": tqdm_dict, "log": tqdm_dict})

        self.log("loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("training_accuracy", accuracy, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        return output

    def validation_step(self, batch, batch_idx):
        segment, target = batch

        classified_segments = self(segment)
        loss = self.BCELoss(classified_segments, target.unsqueeze(1).float())
        accuracy = self.acc(torch.round(classified_segments), target.unsqueeze(1))

        tqdm_dict = {"loss": loss.detach()}
        output = OrderedDict({"loss": loss, "progress_bar": tqdm_dict, "log": tqdm_dict})

        self.log("loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("validation_accuracy", accuracy, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        return output

    def configure_optimizers(self):
        lr = self.hparams.lr
        opt = torch.optim.Adam(self.classifier.parameters(), lr=lr, weight_decay=0.001)
        return [opt], []
