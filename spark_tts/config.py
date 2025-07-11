"""Provides configuration options for inference jobs with the spark-tts package."""

import logging

SPARK_TTS_MODEL_DIR = "pretrained_models/Spark-TTS-0.5B"
PROMPT_SEGMENT_SIZE = 150
DEFAULT_TEMPERATURE = 0.8
DEFAULT_TOP_K = 50
DEFAULT_TOP_P = 0.95

LOGGING_CONFIG = logging.basicConfig(
    level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s"
)
