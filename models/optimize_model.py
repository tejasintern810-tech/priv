import torch
import timm
import torch.nn as nn

from config import MODEL_NAME
from config import PRETRAINED


print("Loading ConvNeXt...")

model = timm.create_model(
    MODEL_NAME,
    pretrained=PRETRAINED,
    num_classes=0,
    global_pool="avg"
)

model.eval()

dummy = torch.randn(
    1,
    3,
    224,
    224
)

print("Exporting TorchScript...")

scripted = torch.jit.trace(
    model,
    dummy
)

scripted.save(
    "database/convnext_scripted.pt"
)

print("Exporting ONNX...")

torch.onnx.export(
    model,
    dummy,
    "database/convnext.onnx",
    export_params=True,
    opset_version=17,
    do_constant_folding=True,
    input_names=["input"],
    output_names=["embedding"],
    dynamic_axes={
        "input": {
            0: "batch"
        },
        "embedding": {
            0: "batch"
        }
    }
)

print()

print("Finished")