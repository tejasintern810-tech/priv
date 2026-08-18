import torch
import timm
import torch.nn as nn

from config import MODEL_NAME


print("Loading Fine-Tuned ConvNeXt...")

model = timm.create_model(
    MODEL_NAME,
    pretrained=False,
    num_classes=9
)

model.load_state_dict(
    torch.load(
        "database/best_convnext.pth",
        map_location="cpu"
    )
)

model.head.fc = nn.Identity()

model.eval()

dummy = torch.randn(
    1,
    3,
    224,
    224
)

print("Exporting TorchScript...")

torch.jit.trace(
    model,
    dummy
).save(
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
    output_names=["output"],
    dynamic_axes={
        "input": {
            0: "batch"
        },
        "output": {
            0: "batch"
        }
    }
)

print()
print("Fine-Tuned Model Exported Successfully")
print("TorchScript : database/convnext_scripted.pt")
print("ONNX        : database/convnext.onnx")