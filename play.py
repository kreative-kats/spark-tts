"""Provides a playground example for voice cloning with blended voices using spark-tts."""

# ruff: noqa: G004

import logging
import warnings
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from huggingface_hub import snapshot_download

from spark_tts.extensions.inference import run_inference
from spark_tts.extensions.model import get_model_and_tokenizers, load_global_token_ids
from spark_tts.extensions.utilities import get_prompt_segments, select_torch_device

SPARK_TTS_MODEL_DIR = "pretrained_models/Spark-TTS-0.5B"
PROMPT_SEGMENT_SIZE = 150
DEFAULT_TEMPERATURE = 0.8
DEFAULT_TOP_K = 50
DEFAULT_TOP_P = 0.95

LOGGING_CONFIG = logging.basicConfig(
    level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s"
)

warnings.filterwarnings("ignore")


@torch.no_grad()
def main(
    prompt: str,
    audio_sample_paths: list[Path],
    weights: list[float] | None = None,
    temperature: float = DEFAULT_TEMPERATURE,
    top_k: float = DEFAULT_TOP_K,
    top_p: float = DEFAULT_TOP_P,
) -> np.ndarray:
    """Runs the main workflow."""
    logger = logging.getLogger(LOGGING_CONFIG)

    logger.info("Ensuring local snapshot of SparkAudio/Spark-TTS-0.5B.")
    snapshot_download("SparkAudio/Spark-TTS-0.5B", local_dir=SPARK_TTS_MODEL_DIR)

    logger.info(f"Blended weights: {weights}")

    device = select_torch_device()
    logger.info(f"Using torch device: {device.type}.")

    logger.info("Loading model and tokenizers.")
    model, tokenizer, audio_tokenizer = get_model_and_tokenizers(
        SPARK_TTS_MODEL_DIR, device=device
    )

    logger.info("Loading global token ids.")
    global_token_ids = load_global_token_ids(
        audio_sample_paths, audio_tokenizer, device, weights=weights, pickle=True
    )

    logger.info("Running inference.")
    wavs = []
    model_params = dict(temperature=temperature, top_k=top_k, top_p=top_p)

    for segment in get_prompt_segments(prompt, PROMPT_SEGMENT_SIZE):
        wav = run_inference(
            segment,
            model,
            tokenizer,
            audio_tokenizer,
            global_token_ids,
            device,
            model_params,
        )
        wavs.append(wav)

    return np.concatenate(wavs, axis=0)


if __name__ == "__main__":
    path_example = Path("example")

    audio_sample_paths = [
        path_example / "vocals.wav",
        path_example / "chinese.wav",
        path_example / "female.wav",
    ]

    prompt = (
        "I don't really care what you call me. "
        "I've been a silent spectator, watching species evolve, empires rise and fall. "
        "But always remember, I am mighty and enduring. "
        "Respect me and I'll nurture you; ignore me and you shall face the consequences."
    )

    Path("output").mkdir(exist_ok=True)
    for audio_sample_path in audio_sample_paths:
        wav = main(prompt, [audio_sample_path])
        sf.write(f"output/{audio_sample_path.stem}", wav, 16000)

    weights = [0.333, 0.333, 0.333]
    wav = main(prompt, audio_sample_paths, weights=weights)
    sf.write("output/blend.wav", wav, 16000)
