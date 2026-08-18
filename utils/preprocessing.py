import cv2
import torch
import numpy as np
from torchvision.transforms import Normalize


normalize = Normalize(
    mean=(0.485, 0.456, 0.406),
    std=(0.229, 0.224, 0.225)
)


def preprocess_image(image_path):

    # Read image with OpenCV (faster than PIL)
    image = cv2.imread(image_path)

    if image is None:
        raise FileNotFoundError(image_path)

    # BGR -> RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Resize
    image = cv2.resize(
        image,
        (224, 224),
        interpolation=cv2.INTER_LINEAR
    )

    # Convert to float32
    image = image.astype(np.float32) / 255.0

    # HWC -> CHW
    image = np.transpose(image, (2, 0, 1))

    image = torch.from_numpy(image)

    image = normalize(image)

    image = image.unsqueeze(0)

    image = image.contiguous(
        memory_format=torch.channels_last
    )

    return image