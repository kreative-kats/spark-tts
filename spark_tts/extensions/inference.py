"""Provides functionality to run inference on audio speech prompts."""

import re

import numpy as np
import torch
from transformers import (
    BatchEncoding,
    PreTrainedModel,
    PreTrainedTokenizerFast,
)

from spark_tts.models.audio_tokenizer import BiCodecTokenizer
from spark_tts.utils.token_parser import TASK_TOKEN_MAP


def get_voice_prompt(
    prompt: str,
    global_token_ids: torch.Tensor,
) -> str:
    """Returns a complete Spark inference prompt, including voice characteristics."""
    global_tokens = "".join(
        [f"<|bicodec_global_{i}|>" for i in global_token_ids.squeeze()]
    )

    inputs = [
        TASK_TOKEN_MAP["tts"],
        "<|start_content|>",
        prompt,
        "<|end_content|>",
        "<|start_global_token|>",
        global_tokens,
        "<|end_global_token|>",
    ]

    return "".join(inputs)


def generate_output_tokens(
    model: PreTrainedModel,
    model_inputs: BatchEncoding,
    temperature: float,
    top_k: float,
    top_p: float,
) -> list[torch.Tensor]:
    """Generates output speech tokens using the model."""
    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=3000,
        do_sample=True,
        top_k=top_k,
        top_p=top_p,
        temperature=temperature,
    )

    # Trim the output tokens to remove the input tokens
    return [
        output_ids[len(input_ids) :]
        for input_ids, output_ids in zip(
            model_inputs.input_ids, generated_ids, strict=False
        )
    ]


def extract_output_semantic_tokens(
    tokenizer: PreTrainedTokenizerFast, output_tokens: list[torch.Tensor]
) -> torch.LongTensor:
    """Returns a Tensor with extracted semantic token id-s from output tokens."""
    strings = tokenizer.batch_decode(output_tokens, skip_special_tokens=True)[0]

    regex = r"bicodec_semantic_(\d+)"
    string_tokens = [int(token) for token in re.findall(regex, strings)]

    return torch.tensor(string_tokens).long().unsqueeze(0)


def run_inference(
    prompt: str,
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerFast,
    audio_tokenizer: BiCodecTokenizer,
    global_token_ids: torch.Tensor,
    device: torch.device,
    model_params: dict[str, float],
) -> np.ndarray:
    """Returns a waveform for predicted audio output."""
    voice_prompt = get_voice_prompt(prompt, global_token_ids)
    model_inputs = tokenizer([voice_prompt], return_tensors="pt").to(device)

    output_tokens = generate_output_tokens(model, model_inputs, **model_params)
    semantic_token_ids = extract_output_semantic_tokens(tokenizer, output_tokens)

    return audio_tokenizer.detokenize(
        global_token_ids.to(device).squeeze(0),
        semantic_token_ids.to(device),
    )


__all__ = ["run_inference"]
