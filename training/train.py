import torch
import numpy as np

from torch.utils.data import DataLoader

from sklearn.utils.class_weight import compute_class_weight

from training.dataset import load_dataset
from training.trainer import Trainer
from training.utils import get_device


############################################################
# Configuration
############################################################

DATASET = "reference"

BATCH_SIZE = 8

EPOCHS = 5

BEST_MODEL = "database/best_convnext.pth"

PATIENCE = 3

############################################################

device = get_device()

print()

print("Device :", device)

############################################################
# Dataset
############################################################

train_dataset, val_dataset, classes = load_dataset(
    DATASET
)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)

############################################################
# Class Weights
############################################################

labels = []

for _, label in train_dataset:

    labels.append(label)

weights = compute_class_weight(

    class_weight="balanced",

    classes=np.unique(labels),

    y=labels

)

weights = torch.tensor(

    weights,

    dtype=torch.float32

)

############################################################
# Trainer
############################################################

trainer = Trainer(

    num_classes=len(classes),

    class_weights=weights,

    device=device

)

############################################################
# Training Variables
############################################################

best_f1 = 0.0

best_accuracy = 0.0

early_stop_counter = 0

############################################################
# Training Loop
############################################################

for epoch in range(EPOCHS):

    print()

    print("=" * 60)

    print(f"Epoch {epoch + 1}/{EPOCHS}")

    print("=" * 60)

    ########################################################

    train_loss, train_acc = trainer.train_one_epoch(
        train_loader
    )

    ########################################################

    val_loss, val_acc, val_f1 = trainer.validate(
        val_loader
    )

    ########################################################

    trainer.scheduler.step()

    ########################################################

    print()

    print(f"Train Loss : {train_loss:.4f}")

    print(f"Train Acc  : {train_acc:.2f}%")

    print()

    print(f"Val Loss   : {val_loss:.4f}")

    print(f"Val Acc    : {val_acc:.2f}%")

    print(f"Val F1     : {val_f1:.4f}")

    ########################################################
    # Save Best Model
    ########################################################

    if val_f1 > best_f1:

        best_f1 = val_f1

        best_accuracy = val_acc

        early_stop_counter = 0

        trainer.save_model(
            BEST_MODEL
        )

        print()

        print("Best Model Saved")

    else:

        early_stop_counter += 1

        print()

        print(
            f"No Improvement ({early_stop_counter}/{PATIENCE})"
        )

    ########################################################
    # Early Stopping
    ########################################################

    if early_stop_counter >= PATIENCE:

        print()

        print("Early Stopping Triggered")

        break

############################################################
# Finished
############################################################

print()

print("=" * 60)

print("Training Finished")

print("=" * 60)

print()

print(
    f"Best Validation Accuracy : {best_accuracy:.2f}%"
)

print(
    f"Best Validation F1       : {best_f1:.4f}"
)

print()

print(
    "Saved Model :",
    BEST_MODEL
)