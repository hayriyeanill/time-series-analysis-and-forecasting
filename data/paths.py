"This module defines paths used in the project."
from pathlib import Path

main_dir = Path(__file__).resolve().parent

RAW_DATA_DIR = Path().resolve().parent / "data" / "raw_data"
PROCESSED_DATA_DIR = Path().resolve().parent / "data" / "processed_data"
SRC_DIR = Path().resolve().parent / "src"
BASELINE_MODEL_DIR = Path().resolve().parent / "data" / "baseline_model_results"
STATISTICAL_MODEL_DIR = Path().resolve().parent / "data" / "statistical_model_results"
PERFORMANCE_DIR = Path().resolve().parent / "data" / "performance_results"
