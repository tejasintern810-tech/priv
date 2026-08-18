"""
Project Configuration

All important settings should be changed ONLY here.
"""

# ---------------------------------------------------
# Folder Locations
# ---------------------------------------------------

REFERENCE_FOLDER = "reference"

INPUT_FOLDER = "input"

OUTPUT_FOLDER = "output"

DATABASE_FOLDER = "database"


# ---------------------------------------------------
# Database
# ---------------------------------------------------

FAISS_INDEX = "database/reference.index"

METADATA = "database/reference_metadata.pkl"


# ---------------------------------------------------
# Prediction
# ---------------------------------------------------

TOP_K = 25


# ---------------------------------------------------
# Threading
# ---------------------------------------------------

MAX_WORKERS = 1


# ---------------------------------------------------
# Watcher
# ---------------------------------------------------

WATCH_DELAY = 0.20


# ---------------------------------------------------
# Voting Weights
# ---------------------------------------------------

WEIGHT_TOTAL = 0.40

WEIGHT_AVERAGE = 0.25

WEIGHT_BEST = 0.35


# ---------------------------------------------------
# Similarity
# ---------------------------------------------------

SIMILARITY_POWER = 8


# ---------------------------------------------------
# Confidence
# ---------------------------------------------------

MIN_CONFIDENCE = 35.0


# ---------------------------------------------------
# Logging
# ---------------------------------------------------

PRINT_TIMINGS = True

PRINT_TOP_MATCHES = False
# ---------------------------------------------------
# Model
# ---------------------------------------------------

MODEL_NAME = "efficientnet_b0"

PRETRAINED = True

INPUT_SIZE = 224

CONFIDENCE_THRESHOLD = 75.0

REFINE_TOP_K = 40