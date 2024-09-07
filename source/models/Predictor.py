import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as L
import torchmetrics

from models.PredictionHead import PredictionHead
from models.SharedHeadBackbone import SharedHeadBackbone


class FTCC(L.LightningModule):
    """
    FTCC: Fine-Tuned Classifier and Clusterizer
    - Fine-Tuned: it is trained on a specific dataset and is meant to do predictions on it
    - Classifier: predict the class/category of a point cloud
    - Clusterizer: predict the number of parts an object has to guide unsupervised segmentation algorithms
    """

    def __init__(self, input_dim, output_dim_class, output_dim_k, hidden_dim=512, lr=5e-3, lambda_=0.5, optimizer='adam', scheduler='constant'):
        super().__init__()

        self.optimizer = optimizer
        self.scheduler = scheduler

        self.head_class = nn.Sequential(
            nn.ReLU(),
            nn.Linear(input_dim, output_dim_class)
        )
        self.head_k = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, output_dim_k),
        )

        # Training
        self.loss = nn.CrossEntropyLoss()
        self.lr = lr
        self.lambda_ = lambda_

        # Evaluation
        self.accuracy_class = torchmetrics.Accuracy(task='multiclass', num_classes=output_dim_class)
        self.accuracy_k = torchmetrics.Accuracy(task='multiclass', num_classes=output_dim_k)
        self.f1_class = torchmetrics.F1Score(task='multiclass', num_classes=output_dim_class)
        self.f1_k = torchmetrics.F1Score(task='multiclass', num_classes=output_dim_k)

        # Output dimensionality
        self.output_dim_class = output_dim_class
        self.output_dim_k = output_dim_k

    def forward(self, x):
        # x = self.backbone(x)
        y_hat_class = self.head_class(x)
        y_hat_k = self.head_k(x)

        return y_hat_class, y_hat_k

    def training_step(self, batch):
        """
        Training step for one batch of data

        :param batch: tuple of tensors representing the [CLS] token (`x`), the class (category) of each object in the
            batch, and the number of clusters/parts `k` of each object in the batch
        :return: the loss computed on the batch of data
        """

        # Get the data from the batch
        # x: [CLS] token from a backbone model, like DINOv2
        # y: class/category for the classification task
        # k: number of clusters
        x, y, k = batch

        # Predict both the label for the class and the number of clusters
        y_hat = self(x)

        # Compute the loss for both the classification and k estimation tasks
        loss_class = self.loss(y_hat[0], torch.argmax(y, dim=-1))
        # loss_k = self.loss(y_hat[1], torch.argmax(k, dim=-1))
        # loss_class = self.loss(F.softmax(y_hat[0], dim=-1), y.float())
        loss_k = self.loss(F.softmax(y_hat[1], dim=-1), k.float())

        loss = self.lambda_ * loss_class + (1. - self.lambda_) * loss_k

        # Measure accuracy on both tasks
        acc_class = self.accuracy_class(y_hat[0], torch.argmax(y, dim=-1))
        acc_k = self.accuracy_k(y_hat[1], torch.argmax(k, dim=-1))

        # Measure f1 score on both tasks
        f1_class = self.f1_class(y_hat[0], torch.argmax(y, dim=-1))
        f1_k = self.f1_k(y_hat[1], torch.argmax(k, dim=-1))

        # Log information
        self.log('train_loss',      loss,       on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('train_acc_class', acc_class,  on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('train_acc_k',     acc_k,      on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('train_f1_class',  f1_class,   on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('train_f1_k',      f1_k,       on_step=True, on_epoch=True, prog_bar=True, logger=True)

        return loss

    def validation_step(self, batch):
        """
        Validation (test) step for one batch of data

        :param batch: tuple of tensors representing the [CLS] token (`x`), the class (category) of each object in the
            batch, and the number of clusters/parts `k` of each object in the batch
        :return: the loss computed on the batch of data
        """

        # Get the data from the batch
        # x: [CLS] token from a backbone model, like DINOv2
        # y: class/category for the classification task
        # k: number of clusters
        x, y, k = batch

        # Predict both the label for the class and the number of clusters
        y_hat = self(x)
        loss_class = self.loss(y_hat[0], torch.argmax(y, dim=-1))
        # loss_k = self.loss(y_hat[1], torch.argmax(k, dim=-1))
        # loss_class = self.loss(F.softmax(y_hat[0], dim=-1), y.float())
        loss_k = self.loss(F.softmax(y_hat[1], dim=-1), k.float())
        loss = self.lambda_ * loss_class + (1. - self.lambda_) * loss_k

        # Measure accuracy on both tasks
        acc_class = self.accuracy_class(y_hat[0], torch.argmax(y, dim=-1))
        acc_k = self.accuracy_k(y_hat[1], torch.argmax(k, dim=-1))

        # Measure f1 score on both tasks
        f1_class = self.f1_class(y_hat[0], torch.argmax(y, dim=-1))
        f1_k = self.f1_k(y_hat[1], torch.argmax(k, dim=-1))

        # Log information
        self.log('val_loss',        loss,       on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('val_acc_class',   acc_class,  on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('val_acc_k',       acc_k,      on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('val_f1_class',    f1_class,   on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('val_f1_k',        f1_k,       on_step=True, on_epoch=True, prog_bar=True, logger=True)

        return loss

    def configure_callbacks(self):
        early_stopping = L.pytorch.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=50,
            mode='min'
        )
        model_checkpoint = L.pytorch.callbacks.ModelCheckpoint(
            monitor='val_loss',
            filename='{epoch}-{val_loss:.2f}',
            mode='min',
            save_top_k=1,
            dirpath='./checkpoints/'
        )

        return [early_stopping, model_checkpoint]

    def configure_optimizers(self):
        if self.optimizer == 'adam':
            optimizer = torch.optim.Adam([
                {'params': self.head_class.parameters(), 'lr': self.lr['class']},
                {'params': self.head_k.parameters()},
            ], lr=self.lr['default'])
        else:
            optimizer = torch.optim.AdamW([
                {'params': self.head_class.parameters(), 'lr': self.lr['class']},
                {'params': self.head_k.parameters()},
            ], lr=self.lr['default'])

        if self.scheduler == 'constant':
            scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer)
        else:
            scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                optimizer,
                T_0=10, T_mult=2, eta_min=1e-6,
                last_epoch=-1
            )

        return [optimizer], [scheduler]
