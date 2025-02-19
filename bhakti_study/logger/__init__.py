import logging
import os
from datetime import datetime

# Create logs directory if it doesn't exist
LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)

# Generate a timestamped filename for the log
CURRENT_TIME_STAMP = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
file_name = f"log_{CURRENT_TIME_STAMP}.log"

# Full path to the log file
log_file_path = os.path.join(LOG_DIR, file_name)

# Configure logging
logging.basicConfig(
    filename=log_file_path,
    filemode="w",  # overwrite each run, or use "a" for append
    format="[%(asctime)s] %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO
)

# Expose a logger instance (optional convenience)
logger = logging.getLogger(__name__)
