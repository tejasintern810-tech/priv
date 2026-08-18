import os

from PIL import Image

from sklearn.model_selection import train_test_split

from torch.utils.data import Dataset

from training.augmentations import (
    train_transform,
    val_transform
)

VALID_EXTENSIONS = (
    ".png",
    ".jpg",
    ".jpeg",
    ".bmp",
    ".tif",
    ".tiff",
)


############################################################


class ELDataset(Dataset):

    def __init__(
        self,
        samples,
        transform=None
    ):

        self.samples = samples

        self.transform = transform

    ##########################################################

    def __len__(self):

        return len(self.samples)

    ##########################################################

    def __getitem__(self, index):

        image_path, label = self.samples[index]

        image = Image.open(
            image_path
        ).convert("RGB")

        if self.transform is not None:

            image = self.transform(image)

        return image, label


############################################################


def load_dataset(root_folder):

    classes = sorted(

        [

            d

            for d in os.listdir(root_folder)

            if os.path.isdir(
                os.path.join(root_folder, d)
            )

        ]

    )

    class_to_idx = {

        cls: idx

        for idx, cls in enumerate(classes)

    }

    samples = []

    labels = []

    ##########################################################

    for cls in classes:

        folder = os.path.join(
            root_folder,
            cls
        )

        for file in os.listdir(folder):

            if not file.lower().endswith(
                VALID_EXTENSIONS
            ):
                continue

            path = os.path.join(
                folder,
                file
            )

            label = class_to_idx[cls]

            samples.append(

                (

                    path,

                    label

                )

            )

            labels.append(label)

    ##########################################################
    # Stratified Train / Validation Split
    ##########################################################

    train_samples, val_samples = train_test_split(

        samples,

        test_size=0.20,

        random_state=42,

        stratify=labels

    )

    train_dataset = ELDataset(

        train_samples,

        transform=train_transform

    )

    val_dataset = ELDataset(

        val_samples,

        transform=val_transform

    )

    return (

        train_dataset,

        val_dataset,

        classes

    )