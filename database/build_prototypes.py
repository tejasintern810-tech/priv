import os
import pickle

import faiss
import numpy as np
import onnxruntime as ort

from PIL import Image, ImageEnhance

import config


# ============================================================
# CPU SETTINGS
# ============================================================

BATCH_SIZE = 8

IMAGE_SIZE = 224

# Use a few CPU threads on the old Intel i5.
# This avoids creating excessive CPU overhead.
ORT_THREADS = 4


# ============================================================
# LOAD ONNX MODEL
# ============================================================

print("Loading ConvNeXt ONNX model...")

session_options = ort.SessionOptions()

session_options.intra_op_num_threads = ORT_THREADS
session_options.inter_op_num_threads = 1

session_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL

session = ort.InferenceSession(
    "database/convnext.onnx",
    sess_options=session_options,
    providers=["CPUExecutionProvider"]
)

input_name = session.get_inputs()[0].name

print("Model loaded.")
print("Input:", input_name)


# ============================================================
# IMAGE PREPROCESSING
# ============================================================

MEAN = np.array(
    [0.485, 0.456, 0.406],
    dtype=np.float32
).reshape(1, 1, 3)

STD = np.array(
    [0.229, 0.224, 0.225],
    dtype=np.float32
).reshape(1, 1, 3)


def preprocess(img):
    """
    Convert PIL image into ConvNeXt input.

    Output:
        float32 array with shape (3, 224, 224)
    """

    img = img.resize(
        (IMAGE_SIZE, IMAGE_SIZE),
        Image.Resampling.BILINEAR
    )

    arr = np.asarray(
        img,
        dtype=np.float32
    )

    # Convert [0,255] -> [0,1]
    arr /= 255.0

    # ImageNet normalization
    arr = (arr - MEAN) / STD

    # HWC -> CHW
    arr = np.transpose(
        arr,
        (2, 0, 1)
    )

    return arr.astype(np.float32)


# ============================================================
# EXTRACT FEATURES IN BATCH
# ============================================================

def extract_batch(images):

    batch = np.asarray(
        [
            preprocess(img)
            for img in images
        ],
        dtype=np.float32
    )

    outputs = session.run(
        None,
        {
            input_name: batch
        }
    )[0]

    # Some ONNX exports return:
    #
    # (B, C, H, W)
    #
    # Others return:
    #
    # (B, C)
    #
    if outputs.ndim == 4:

        outputs = outputs.mean(
            axis=(2, 3)
        )

    outputs = outputs.astype(
        np.float32
    )

    # L2 normalize every embedding
    norms = np.linalg.norm(
        outputs,
        axis=1,
        keepdims=True
    )

    outputs = outputs / (
        norms + 1e-12
    )

    return outputs


# ============================================================
# CREATE REFERENCE IMAGE LIST
# ============================================================

print()
print("Scanning reference database...")

image_records = []

class_names = []

for cls in sorted(
    os.listdir(
        config.REFERENCE_FOLDER
    )
):

    folder = os.path.join(
        config.REFERENCE_FOLDER,
        cls
    )

    if not os.path.isdir(folder):
        continue

    class_names.append(cls)

    for file in sorted(
        os.listdir(folder)
    ):

        if not file.lower().endswith(
            (
                ".png",
                ".jpg",
                ".jpeg",
                ".bmp",
                ".tif",
                ".tiff"
            )
        ):
            continue

        path = os.path.join(
            folder,
            file
        )

        image_records.append(
            {
                "class": cls,
                "path": path
            }
        )


print(
    "Reference images:",
    len(image_records)
)

print(
    "Classes:",
    len(class_names)
)

print()


# ============================================================
# BUILD EMBEDDINGS
# ============================================================

embeddings = []

metadata = []

total = len(image_records)

processed = 0

print("Building Database...")
print(
    "Batch size:",
    BATCH_SIZE
)

print()


for start in range(
    0,
    total,
    BATCH_SIZE
):

    batch_records = image_records[
        start:start + BATCH_SIZE
    ]

    images = []

    valid_records = []

    # --------------------------------------------------------
    # LOAD IMAGES
    # --------------------------------------------------------

    for record in batch_records:

        try:

            img = Image.open(
                record["path"]
            ).convert("RGB")

            images.append(img)

            valid_records.append(
                record
            )

        except Exception as e:

            print(
                "Skipping:",
                record["path"],
                "|",
                e
            )


    if not images:
        continue


    # --------------------------------------------------------
    # ORIGINAL IMAGES
    # --------------------------------------------------------

    output = extract_batch(
        images
    )

    for i, record in enumerate(
        valid_records
    ):

        embeddings.append(
            output[i]
        )

        metadata.append(
            {
                "class": record["class"],
                "path": record["path"]
            }
        )


    # --------------------------------------------------------
    # HORIZONTAL FLIP
    #
    # Only one augmentation is used.
    #
    # This gives some robustness without creating
    # 4x the inference workload.
    # --------------------------------------------------------

    flipped_images = [
        img.transpose(
            Image.Transpose.FLIP_LEFT_RIGHT
        )
        for img in images
    ]

    flipped_output = extract_batch(
        flipped_images
    )

    for i, record in enumerate(
        valid_records
    ):

        embeddings.append(
            flipped_output[i]
        )

        metadata.append(
            {
                "class": record["class"],
                "path": record["path"]
            }
        )


    # --------------------------------------------------------
    # PROGRESS
    # --------------------------------------------------------

    processed += len(
        valid_records
    )

    print(
        f"Processed {processed}/{total} "
        f"({processed / total * 100:.1f}%)"
    )


# ============================================================
# CONVERT TO NUMPY
# ============================================================

if not embeddings:

    raise RuntimeError(
        "No embeddings were generated."
    )


embeddings = np.asarray(
    embeddings,
    dtype=np.float32
)


# ============================================================
# BUILD FAISS INDEX
# ============================================================

print()
print("Building FAISS index...")

dimension = embeddings.shape[1]

index = faiss.IndexFlatIP(
    dimension
)

index.add(
    embeddings
)


# ============================================================
# SAVE FAISS INDEX
# ============================================================

faiss.write_index(
    index,
    config.FAISS_INDEX
)


# ============================================================
# SAVE METADATA
# ============================================================

with open(
    config.METADATA,
    "wb"
) as f:

    pickle.dump(
        metadata,
        f
    )


# ============================================================
# FINAL REPORT
# ============================================================

print()
print("=" * 60)
print("DATABASE BUILD COMPLETE")
print("=" * 60)

print(
    "Reference images :",
    len(image_records)
)

print(
    "Embeddings        :",
    len(metadata)
)

print(
    "Embedding size    :",
    dimension
)

print(
    "Classes            :",
    len(
        set(
            item["class"]
            for item in metadata
        )
    )
)

print(
    "FAISS index       :",
    config.FAISS_INDEX
)

print(
    "Metadata          :",
    config.METADATA
)

print("=" * 60)

