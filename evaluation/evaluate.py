import os
import json
import csv

from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)

from classifier.predictor import Predictor


############################################################
# TEST DATASET LOCATION
############################################################

TEST_FOLDER = "test_images"

VALID_EXTENSIONS = (
    ".png",
    ".jpg",
    ".jpeg",
    ".bmp",
    ".tif",
    ".tiff",
)

############################################################

predictor = Predictor()

############################################################


def get_dataset():

    dataset = []

    if not os.path.exists(TEST_FOLDER):

        raise FileNotFoundError(
            f"{TEST_FOLDER} not found."
        )

    for cls in sorted(os.listdir(TEST_FOLDER)):

        class_folder = os.path.join(
            TEST_FOLDER,
            cls
        )

        if not os.path.isdir(class_folder):
            continue

        for root, _, files in os.walk(class_folder):

            for file in files:

                if not file.lower().endswith(
                    VALID_EXTENSIONS
                ):
                    continue

                dataset.append(
                    (
                        os.path.join(root, file),
                        cls
                    )
                )

    return dataset


############################################################

dataset = get_dataset()

print()

print("=" * 70)
print("EL Defect Evaluation")
print("=" * 70)

print()

print("Test Images :", len(dataset))

print()

############################################################
# Storage
############################################################

ground_truth = []

predictions = []

prediction_rows = []

feature_times = []

search_times = []

vote_times = []

total_times = []

top3_correct = 0

top5_correct = 0

per_class_total = defaultdict(int)

per_class_correct = defaultdict(int)
############################################################
# Prediction Loop
############################################################

for index, (image_path, true_class) in enumerate(dataset):

    print(
        f"[{index + 1}/{len(dataset)}] "
        f"{os.path.basename(image_path)}"
    )

    result = predictor.predict(image_path)

    predicted_class = result["prediction"]

    confidence = result["confidence"]

    matches = result["matches"]

    ########################################################

    ground_truth.append(true_class)

    predictions.append(predicted_class)

    ########################################################
    # Per-class statistics
    ########################################################

    per_class_total[true_class] += 1

    if predicted_class == true_class:

        per_class_correct[true_class] += 1

    ########################################################
    # Top-3 Accuracy
    ########################################################

    top3_classes = []

    for item in matches[:3]:

        if item["class"] not in top3_classes:

            top3_classes.append(item["class"])

    if true_class in top3_classes:

        top3_correct += 1

    ########################################################
    # Top-5 Accuracy
    ########################################################

    top5_classes = []

    for item in matches[:5]:

        if item["class"] not in top5_classes:

            top5_classes.append(item["class"])

    if true_class in top5_classes:

        top5_correct += 1

    ########################################################
    # Timings
    ########################################################

    feature_times.append(
        result["timings"]["feature_extraction"]
    )

    search_times.append(
        result["timings"]["faiss"]
    )

    vote_times.append(
        result["timings"]["vote"]
    )

    total_times.append(
        result["timings"]["total"]
    )

    ########################################################
    # Store prediction row
    ########################################################

    prediction_rows.append(

        {

            "image": os.path.basename(image_path),

            "ground_truth": true_class,

            "prediction": predicted_class,

            "confidence": round(confidence, 2),

            "correct": predicted_class == true_class

        }

    )
############################################################
# Overall Metrics
############################################################

accuracy = accuracy_score(
    ground_truth,
    predictions
)

precision = precision_score(
    ground_truth,
    predictions,
    average="weighted",
    zero_division=0
)

recall = recall_score(
    ground_truth,
    predictions,
    average="weighted",
    zero_division=0
)

f1 = f1_score(
    ground_truth,
    predictions,
    average="weighted",
    zero_division=0
)

top3_accuracy = top3_correct / len(dataset)

top5_accuracy = top5_correct / len(dataset)

############################################################

labels = sorted(list(set(ground_truth)))

report = classification_report(
    ground_truth,
    predictions,
    labels=labels,
    digits=4,
    zero_division=0
)

cm = confusion_matrix(
    ground_truth,
    predictions,
    labels=labels
)

############################################################
# Create evaluation folder
############################################################

os.makedirs(
    "evaluation",
    exist_ok=True
)

############################################################
# Save Classification Report
############################################################

with open(
    "evaluation/classification_report.txt",
    "w",
    encoding="utf-8"
) as f:

    f.write(report)

############################################################
# Save CSV
############################################################

with open(
    "evaluation/predictions.csv",
    "w",
    newline="",
    encoding="utf-8"
) as csvfile:

    writer = csv.DictWriter(

        csvfile,

        fieldnames=[

            "image",

            "ground_truth",

            "prediction",

            "confidence",

            "correct"

        ]

    )

    writer.writeheader()

    writer.writerows(prediction_rows)

############################################################
# Per-class Accuracy
############################################################

per_class_accuracy = {}

for cls in labels:

    total = per_class_total[cls]

    correct = per_class_correct[cls]

    if total == 0:

        per_class_accuracy[cls] = 0

    else:

        per_class_accuracy[cls] = round(
            (correct / total) * 100,
            2
        )

############################################################
# Metrics JSON
############################################################

metrics = {

    "accuracy":
        round(float(accuracy), 4),

    "precision":
        round(float(precision), 4),

    "recall":
        round(float(recall), 4),

    "f1_score":
        round(float(f1), 4),

    "top3_accuracy":
        round(float(top3_accuracy), 4),

    "top5_accuracy":
        round(float(top5_accuracy), 4),

    "average_feature_time_ms":
        round(float(np.mean(feature_times)), 2),

    "average_search_time_ms":
        round(float(np.mean(search_times)), 2),

    "average_vote_time_ms":
        round(float(np.mean(vote_times)), 2),

    "average_total_time_ms":
        round(float(np.mean(total_times)), 2),

    "per_class_accuracy":
        per_class_accuracy

}

with open(
    "evaluation/metrics.json",
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        metrics,
        f,
        indent=4
    )

############################################################
# Confusion Matrix
############################################################

plt.figure(figsize=(10,8))

plt.imshow(
    cm,
    interpolation="nearest"
)

plt.title("Confusion Matrix")

plt.colorbar()

ticks = np.arange(len(labels))

plt.xticks(
    ticks,
    labels,
    rotation=45,
    ha="right"
)

plt.yticks(
    ticks,
    labels
)

plt.xlabel("Predicted")

plt.ylabel("True")

for i in range(cm.shape[0]):

    for j in range(cm.shape[1]):

        plt.text(
            j,
            i,
            str(cm[i, j]),
            ha="center",
            va="center",
            fontsize=8
        )

plt.tight_layout()

plt.savefig(
    "evaluation/confusion_matrix.png",
    dpi=300
)

plt.close()    
############################################################
# Human Readable Evaluation Report
############################################################

with open(
    "evaluation/evaluation_report.txt",
    "w",
    encoding="utf-8"
) as f:

    f.write("=" * 70 + "\n")
    f.write("EL Defect Classification Evaluation\n")
    f.write("=" * 70 + "\n\n")

    f.write(f"Total Test Images : {len(dataset)}\n\n")

    f.write("Overall Metrics\n")
    f.write("------------------------------\n")

    f.write(f"Accuracy       : {accuracy * 100:.2f}%\n")
    f.write(f"Precision      : {precision * 100:.2f}%\n")
    f.write(f"Recall         : {recall * 100:.2f}%\n")
    f.write(f"F1 Score       : {f1 * 100:.2f}%\n")
    f.write(f"Top-3 Accuracy : {top3_accuracy * 100:.2f}%\n")
    f.write(f"Top-5 Accuracy : {top5_accuracy * 100:.2f}%\n\n")

    f.write("Average Timings\n")
    f.write("------------------------------\n")

    f.write(
        f"Feature Extraction : {np.mean(feature_times):.2f} ms\n"
    )

    f.write(
        f"FAISS Search       : {np.mean(search_times):.2f} ms\n"
    )

    f.write(
        f"Voting             : {np.mean(vote_times):.2f} ms\n"
    )

    f.write(
        f"Total              : {np.mean(total_times):.2f} ms\n\n"
    )

    f.write("Per Class Accuracy\n")
    f.write("------------------------------\n")

    for cls in labels:

        f.write(
            f"{cls:25s} : "
            f"{per_class_accuracy[cls]:6.2f}%\n"
        )

############################################################
# Console Output
############################################################

print()

print("=" * 70)
print("Evaluation Completed")
print("=" * 70)

print()

print(f"Total Images     : {len(dataset)}")

print(f"Accuracy         : {accuracy * 100:.2f}%")

print(f"Precision        : {precision * 100:.2f}%")

print(f"Recall           : {recall * 100:.2f}%")

print(f"F1 Score         : {f1 * 100:.2f}%")

print(f"Top-3 Accuracy   : {top3_accuracy * 100:.2f}%")

print(f"Top-5 Accuracy   : {top5_accuracy * 100:.2f}%")

print()

print("Average Timings")

print("------------------------------")

print(
    f"Feature Extraction : {np.mean(feature_times):.2f} ms"
)

print(
    f"FAISS Search       : {np.mean(search_times):.2f} ms"
)

print(
    f"Voting             : {np.mean(vote_times):.2f} ms"
)

print(
    f"Total              : {np.mean(total_times):.2f} ms"
)

print()

print("Per Class Accuracy")

print("------------------------------")

for cls in labels:

    print(
        f"{cls:25s} "
        f"{per_class_accuracy[cls]:6.2f}%"
    )

print()

print("Reports Saved")

print("------------------------------")

print("evaluation/classification_report.txt")
print("evaluation/evaluation_report.txt")
print("evaluation/confusion_matrix.png")
print("evaluation/predictions.csv")
print("evaluation/metrics.json")
