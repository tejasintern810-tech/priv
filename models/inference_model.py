import torch
import torch.nn as nn
import timm


class InferenceModel:

    def __init__(self, model_path, num_classes):

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        self.model = timm.create_model(
            "convnext_tiny",
            pretrained=False,
            num_classes=num_classes
        )

        state = torch.load(
            model_path,
            map_location=self.device
        )

        self.model.load_state_dict(state)

        self.model.eval()

        self.model.to(self.device)

    #######################################################

    def predict(self, image):

        with torch.no_grad():

            image = image.to(self.device)

            logits = self.model(image)

            probabilities = torch.softmax(
                logits,
                dim=1
            )

            confidence, prediction = torch.max(
                probabilities,
                dim=1
            )

        return (

            prediction.item(),

            confidence.item(),

            probabilities.squeeze(0).cpu().numpy()
        )

    #######################################################

    def extract_embedding(self, image):

        with torch.no_grad():

            image = image.to(self.device)

            features = self.model.forward_features(image)

            features = features.mean(dim=(-2, -1))

            features = nn.functional.normalize(
                features,
                dim=1
            )

        return features.cpu().numpy().astype("float32")