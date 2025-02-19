import os
from datetime import datetime

def get_current_time_stamp():
    return datetime.now().strftime("%Y-%m-%d-%H-%M-%S")

CURRENT_TIME_STAMP = get_current_time_stamp()
ROOT_DIR_KEY = os.getcwd()

# Directory for raw data
DATA_DIR = os.path.join("data", "raw")

# Paths to CSV files
CONTENT_DATA_FILE = os.path.join(DATA_DIR, "content_data_MASTER.csv")
LABELS_DATA_FILE = os.path.join(DATA_DIR, "labels_MASTER.csv")
TEST_DATA_FILE = os.path.join(DATA_DIR, "test_MASTER.csv")

# Example artifact directory (if you need to store processed data, models, etc.)
ARTIFACT_DIR_KEY = "Artifact"

# (Optional) placeholders if you have a data ingestion pipeline:
DATA_INGESTION_KEY = "data_ingestion"
DATA_INGESTION_RAW_DATA_DIR = "raw_data"
DATA_INGESTION_INGESTED_DATA_DIR = "ingested_data"
TRAIN_DATA_DIR_KEY = "train"
TEST_DATA_DIR_KEY = "test"

# (Optional) placeholders for data transformation steps:
DATA_TRANSFORMATION_ARTIFACT = "data_transformation"
DATA_PREPROCCED_DIR = "preprocessed"
DATA_TRANSFORMTION_PROCESSING_OBJ = "transformer.pkl"
DATA_TRANSFORM_DIR = "transform"
TRANSFORM_TRAIN_DIR_KEY = "transform_train"
TRANSFORM_TEST_DIR_KEY = "transform_test"
