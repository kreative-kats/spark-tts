"""Includes Spark TTS model inference functions."""

from pathlib import Path
from typing import Self

import tokenizers
import torch
import transformers
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizerFast,
)

from spark_tts.extensions.tensors import combine_tensors, sanitize_weights
from spark_tts.extensions.utilities import (
    get_id_from_path,
    get_id_from_paths_and_weights,
)
from spark_tts.models.audio_tokenizer import BiCodecTokenizer

MODEL_TOKENIZER_CLASSES = [
    tokenizers.AddedToken,
    tokenizers.Tokenizer,
    tokenizers.models.Model,
    torch.nn.modules.activation.SiLU,
    torch.nn.modules.container.ModuleList,
    torch.nn.modules.linear.Linear,
    torch.nn.modules.sparse.Embedding,
    transformers.generation.configuration_utils.GenerationConfig,
    transformers.models.qwen2.configuration_qwen2.Qwen2Config,
    transformers.models.qwen2.modeling_qwen2.Qwen2DecoderLayer,
    transformers.models.qwen2.modeling_qwen2.Qwen2ForCausalLM,
    transformers.models.qwen2.modeling_qwen2.Qwen2MLP,
    transformers.models.qwen2.modeling_qwen2.Qwen2Model,
    transformers.models.qwen2.modeling_qwen2.Qwen2RMSNorm,
    transformers.models.qwen2.modeling_qwen2.Qwen2RotaryEmbedding,
    transformers.models.qwen2.modeling_qwen2.Qwen2SdpaAttention,
    transformers.models.qwen2.tokenization_qwen2_fast.Qwen2TokenizerFast,
    transformers.modeling_rope_utils._compute_default_rope_parameters,
]


class BiCodecTokenizerBlend(BiCodecTokenizer):
    """Extends the Spark-TTS BiCodecTokenizer with support for voice blending."""

    def tokenize_blend(
        self: Self,
        audio_sample_paths: list[Path],
        device: torch.device,
        weights: list[float],
    ) -> torch.Tensor:
        """Returns the weighted global token id-s for a range of sampled voices."""
        xs = []

        mel_transformer = self.model.mel_transformer
        speaker_encoder = self.model.speaker_encoder
        for path in audio_sample_paths:
            _, ref_wav = self.process_audio(path)
            mels = mel_transformer(ref_wav.to(device)).squeeze(1).transpose(1, 2)
            _, features = speaker_encoder.speaker_encoder(mels, True)
            x = speaker_encoder.perceiver_sampler(features.transpose(1, 2)).transpose(
                1, 2
            )
            xs.append(x)

        x = combine_tensors(xs, weights=weights, as_int=False).to(device)
        _, indices = speaker_encoder.quantizer(x)
        return indices


def add_safe_globals() -> None:
    """Ensures torch can load the classes for a pickled model or tokenizer."""
    torch.serialization.add_safe_globals(MODEL_TOKENIZER_CLASSES)


def get_model_and_tokenizers(
    model_dir: Path,
    device: torch.device | None = None,
    cache_path: Path | None = None,
) -> tuple[PreTrainedModel, PreTrainedTokenizerFast, BiCodecTokenizerBlend]:
    """Returns the loaded causal LM, tokenizer and audio tokenizer for a pretrained model."""
    identifier = get_id_from_path(model_dir)

    load_from_cache = False
    if cache_path:
        model_path = cache_path / f"{identifier}_model.pt"
        tokenizer_path = cache_path / f"{identifier}_tokenizer.pt"

        load_from_cache = model_path.exists() and tokenizer_path.exists()

    if load_from_cache:
        add_safe_globals()
        model = torch.load(model_path)
        tokenizer = torch.load(tokenizer_path)
    else:
        model = AutoModelForCausalLM.from_pretrained(f"{model_dir}/LLM")
        tokenizer = AutoTokenizer.from_pretrained(f"{model_dir}/LLM")

        if cache_path:
            torch.save(model, model_path)
            torch.save(tokenizer, tokenizer_path)

    model.to(device)
    audio_tokenizer = BiCodecTokenizerBlend(model_dir, device=device)

    return model, tokenizer, audio_tokenizer


# TODO: Deprecate
def load_sample_token_ids(
    audio_sample_path: Path,
    audio_tokenizer: BiCodecTokenizer | BiCodecTokenizerBlend,
    cache_path: Path | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Returns the global and semantic token id-s for the audio sample at sample_path."""
    identifier = get_id_from_path(audio_sample_path)

    load_from_cache = False
    if cache_path:
        global_token_ids_path = cache_path / f"{identifier}_global_token_ids.pt"
        semantic_token_ids_path = cache_path / f"{identifier}_semantic_token_ids.pt"
        load_from_cache = (
            global_token_ids_path.exists() and semantic_token_ids_path.exists()
        )

    if load_from_cache:
        global_token_ids = torch.load(global_token_ids_path)
        semantic_token_ids = torch.load(semantic_token_ids_path)
    else:
        global_token_ids, semantic_token_ids = audio_tokenizer.tokenize(
            audio_sample_path
        )

        if cache_path:
            torch.save(global_token_ids, global_token_ids_path)
            torch.save(semantic_token_ids, semantic_token_ids_path)

    return global_token_ids, semantic_token_ids


def load_global_token_ids(
    audio_sample_paths: list[Path],
    audio_tokenizer: BiCodecTokenizerBlend,
    device: torch.device,
    weights: list[float] | None = None,
    cache_path: Path | None = None,
) -> torch.Tensor:
    """Returns the global and semantic token id-s for the audio sample at sample_path."""
    weights = sanitize_weights(weights or len(audio_sample_paths))
    identifier = get_id_from_paths_and_weights(audio_sample_paths, weights)

    load_from_cache = False
    if cache_path:
        global_token_ids_path = cache_path / f"{identifier}_global_token_ids.pt"
        load_from_cache = global_token_ids_path.exists()

    if load_from_cache:
        global_token_ids = torch.load(global_token_ids_path)
    else:
        global_token_ids = audio_tokenizer.tokenize_blend(
            audio_sample_paths, device, weights
        )

        if cache_path:
            torch.save(global_token_ids, global_token_ids_path)

    return global_token_ids


__all__ = [
    "BiCodecTokenizerBlend",
    "get_model_and_tokenizers",
    "load_global_token_ids",
]
