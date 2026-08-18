import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights

from utils.preprocessing import preprocess_image


class FeatureExtractor:

    def __init__(self):

        weights = EfficientNet_B0_Weights.IMAGENET1K_V1

        self.model = efficientnet_b0(
            weights=weights
        )

        self.model.classifier = nn.Identity()
        self.model.eval()
        self.model.to("cpu")

    def extract_features(self, image_path):

        image = preprocess_image(image_path)

        if not isinstance(image, torch.Tensor):
            raise RuntimeError("Expected a torch.Tensor from preprocessing")

        with torch.no_grad():
            image = image.to("cpu")
            embedding = self.model(image)
            embedding = F.normalize(
                embedding,
                p=2,
                dim=1
            )

        return embedding.squeeze(0)
    