import torch
import torch.nn as nn
import timm

from sklearn.metrics import f1_score
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR


class Trainer:

    def __init__(
        self,
        num_classes,
        class_weights,
        device
    ):

        self.device = device

        ###########################################################
        # Load pretrained ConvNeXt
        ###########################################################

        self.model = timm.create_model(
            "convnext_tiny",
            pretrained=True,
            num_classes=num_classes
        )

        ###########################################################
        # Freeze entire backbone
        ###########################################################

        for param in self.model.parameters():

            param.requires_grad = False

        ###########################################################
        # Unfreeze last ConvNeXt stage
        ###########################################################

        if hasattr(self.model, "stages"):

            for param in self.model.stages[-1].parameters():

                param.requires_grad = True

        ###########################################################
        # Unfreeze classifier
        ###########################################################

        if hasattr(self.model, "head"):

            for param in self.model.head.parameters():

                param.requires_grad = True

        ###########################################################

        self.model.to(device)

        ###########################################################
        # Loss
        ###########################################################

        self.criterion = nn.CrossEntropyLoss(

            weight=class_weights.to(device),

            label_smoothing=0.1

        )

        ###########################################################
        # Optimizer
        ###########################################################

        trainable_parameters = filter(

            lambda p: p.requires_grad,

            self.model.parameters()

        )

        self.optimizer = AdamW(

            trainable_parameters,

            lr=1e-4,

            weight_decay=1e-4

        )

        ###########################################################
        # Scheduler
        ###########################################################

        self.scheduler = CosineAnnealingLR(

            self.optimizer,

            T_max=5

        )

    ##############################################################

    def train_one_epoch(
        self,
        loader
    ):

        self.model.train()

        running_loss = 0.0

        correct = 0

        total = 0

        for batch_index, (images, labels) in enumerate(loader):

            if batch_index % 10 == 0:

                print(
                    f"Batch {batch_index}/{len(loader)}"
                )

            images = images.to(self.device)

            labels = labels.to(self.device)

            self.optimizer.zero_grad()

            outputs = self.model(images)

            loss = self.criterion(
                outputs,
                labels
            )

            loss.backward()

            self.optimizer.step()

            running_loss += loss.item()

            _, predicted = outputs.max(1)

            total += labels.size(0)

            correct += (

                predicted == labels

            ).sum().item()

        accuracy = (

            correct / total

        ) * 100

        average_loss = (

            running_loss / len(loader)

        )

        return average_loss, accuracy

    ##############################################################

    def validate(
        self,
        loader
    ):

        self.model.eval()

        running_loss = 0.0

        correct = 0

        total = 0

        all_predictions = []

        all_labels = []

        with torch.no_grad():

            for images, labels in loader:

                images = images.to(self.device)

                labels = labels.to(self.device)

                outputs = self.model(images)

                loss = self.criterion(
                    outputs,
                    labels
                )

                running_loss += loss.item()

                _, predicted = outputs.max(1)

                all_predictions.extend(

                    predicted.cpu().numpy()

                )

                all_labels.extend(

                    labels.cpu().numpy()

                )

                total += labels.size(0)

                correct += (

                    predicted == labels

                ).sum().item()

        accuracy = (

            correct / total

        ) * 100

        average_loss = (

            running_loss / len(loader)

        )

        f1 = f1_score(

            all_labels,

            all_predictions,

            average="weighted",

            zero_division=0

        )

        return average_loss, accuracy, f1

    ##############################################################

    def save_model(
        self,
        filename
    ):

        torch.save(

            self.model.state_dict(),

            filename

        )