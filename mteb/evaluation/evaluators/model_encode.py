from __future__ import annotations

import logging
from typing import Sequence

import numpy as np
import torch

from mteb.encoder_interface import Encoder

logger = logging.getLogger(__name__)

HF_TOKEN = "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"


def model_encode(
    sentences: Sequence[str], *, model: Encoder, prompt_name: str | None, **kwargs
) -> np.ndarray:
    """A wrapper function around the model.encode method that handles the prompt_name argument and standardizes the output to a numpy array.

    Args:
        sentences: The sentences to encode
        model: The model to use for encoding
        prompt_name: The prompt name to use for encoding
        **kwargs: Additional arguments to pass to the model.encode method
    """
    print('enter model encode kwargs: ', kwargs)
    kwargs["prompt_name"] = prompt_name
    if hasattr(model, "prompts"):
        # check if prompts is an empty dict
        if not model.prompts:  # type: ignore
            logger.info(
                "Model does not support prompts. Removing prompt_name argument."
            )
            kwargs.pop("prompt_name")
        elif prompt_name not in model.prompts:  # type: ignore
            logger.info(
                f"Prompt {prompt_name} not found in model prompts. Removing prompt_name argument."
            )
            kwargs.pop("prompt_name")
    logger.info(f"Encoding {len(sentences)} sentences.")

    embeddings, default, moe_rw = model.encode(sentences, **kwargs)
    if isinstance(embeddings, torch.Tensor):
        embeddings = embeddings.cpu().detach().float()
    if isinstance(default, torch.Tensor):
        default = default.cpu().detach().float()
    if isinstance(moe_rw, torch.Tensor):
        moe_rw = moe_rw.cpu().detach().float()
        
    print('embeddings.shape: ', embeddings.shape)

    return np.asarray(embeddings), np.asarray(default), np.asarray(moe_rw) 
