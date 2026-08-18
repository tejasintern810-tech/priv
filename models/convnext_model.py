import os
import torch
import timm
import torch.nn as nn

from config import MODEL_NAME

# -------------------------------------------------------------

BEST_MODEL = "database/best_convnext.pth"

# -------------------------------------------------------------

if "OMP_NUM_THREADS" not in os.environ:
    torch.set_num_threads(4)

# -------------------------------------------------------------

def load_model():

    model = timm.create_model(
        MODEL_NAME,
        pretrained=False,
        num_classes=9
    )

    if os.path.exists(BEST_MODEL):

        print("Loading Fine-Tuned ConvNeXt...")

        state_dict = torch.load(
            BEST_MODEL,
            map_location="cpu"
        )

        model.load_state_dict(state_dict)

    else:

        print("Fine-tuned model not found.")
        print("Using ImageNet weights.")

        model = timm.create_model(
            MODEL_NAME,
            pretrained=True,
            num_classes=9
        )

    # ---------------------------------------------------------
    # Remove classifier
    # ---------------------------------------------------------

    model.head = nn.Identity()

    model.eval()

    model = model.to(memory_format=torch.channels_last)

    torch.set_grad_enabled(False)

    for p in model.parameters():
        p.requires_grad_(False)

    return model