"This module defines paths used in the project."
from pathlib import Path

main_dir = Path(__file__).resolve().parent

RAW_DATA_DIR = Path().resolve().parent / "data" / "raw_data"
PROCESSED_DATA_DIR = Path().resolve().parent / "data" / "processed_data"
SRC_DIR = Path().resolve().parent / "src"
