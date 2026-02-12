"""Perplexity evaluation for language models.

Computes standard sliding-window perplexity on text datasets, supporting
evaluation on WikiText-2 (via HuggingFace datasets) and synthetic data.
"""

from typing import Callable, Optional

import jax
import jax.numpy as jnp
import numpy as np


def load_wikitext2(split: str = "test") -> list[str]:
    """Load the WikiText-2 dataset from HuggingFace.

    Args:
        split: Dataset split to load ('train', 'validation', or 'test').

    Returns:
        List of non-empty text strings from the dataset.

    Raises:
        RuntimeError: If the datasets library is not installed.
    """
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError(
            "The 'datasets' library is required for WikiText-2 evaluation. "
            "Install with: pip install datasets"
        ) from exc

    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split=split)
    return [row["text"] for row in dataset if row["text"].strip()]


def evaluate_perplexity(
    model_apply_fn: Callable,
    params: dict,
    dataset_texts: list[str],
    tokenizer_encode_fn: Callable[[str], list[int]],
    max_seq_len: int = 64,
    stride: int = 512,
    max_samples: Optional[int] = None,
) -> float:
    """Evaluate perplexity on a text dataset using sliding window.

    Computes perplexity by sliding a window of max_seq_len over the
    concatenated text, using stride-sized steps, and accumulating the
    negative log-likelihood.

    Args:
        model_apply_fn: The model's apply function.
        params: Model parameters.
        dataset_texts: List of text strings to evaluate on.
        tokenizer_encode_fn: Function mapping text to list of token IDs.
        max_seq_len: Maximum sequence length per window.
        stride: Step size for the sliding window.
        max_samples: Maximum number of text samples to process.

    Returns:
        Perplexity value (lower is better).
    """
    all_token_ids = []
    texts_to_process = dataset_texts[:max_samples] if max_samples else dataset_texts
    for text in texts_to_process:
        if text.strip():
            all_token_ids.extend(tokenizer_encode_fn(text))

    if not all_token_ids:
        return float("inf")

    total_nll = 0.0
    total_tokens = 0

    for begin_idx in range(0, len(all_token_ids) - 1, stride):
        end_idx = min(begin_idx + max_seq_len, len(all_token_ids))
        input_ids = jnp.array(all_token_ids[begin_idx:end_idx])[None, :]
        target_ids = jnp.array(all_token_ids[begin_idx + 1 : end_idx + 1])

        actual_len = min(end_idx - begin_idx, len(all_token_ids) - begin_idx - 1)
        if actual_len <= 0:
            break

        input_ids = input_ids[:, :actual_len]
        target_ids = target_ids[:actual_len]

        logits = model_apply_fn(params, input_ids)
        log_probs = jax.nn.log_softmax(logits[0], axis=-1)

        target_len = min(target_ids.shape[0], log_probs.shape[0])
        target_ids = target_ids[:target_len]
        log_probs = log_probs[:target_len]

        token_log_probs = log_probs[jnp.arange(target_len), target_ids]

        eval_start = 0 if begin_idx == 0 else (end_idx - begin_idx - stride)
        eval_start = max(0, min(eval_start, target_len))

        total_nll -= jnp.sum(token_log_probs[eval_start:]).item()
        total_tokens += target_len - eval_start

    if total_tokens == 0:
        return float("inf")

    avg_nll = total_nll / total_tokens
    perplexity = float(np.exp(min(avg_nll, 100.0)))
    return perplexity


def evaluate_perplexity_simple(
    model_apply_fn: Callable,
    params: dict,
    token_ids: jnp.ndarray,
    max_seq_len: int = 64,
) -> float:
    """Simplified perplexity evaluation on pre-tokenized data.

    Args:
        model_apply_fn: The model's apply function.
        params: Model parameters.
        token_ids: 1D array of token IDs.
        max_seq_len: Maximum sequence length per chunk.

    Returns:
        Perplexity value.
    """
    total_nll = 0.0
    total_tokens = 0

    for start in range(0, len(token_ids) - 1, max_seq_len):
        end = min(start + max_seq_len, len(token_ids) - 1)
        if end <= start:
            break

        input_chunk = token_ids[start:end][None, :]
        target_chunk = token_ids[start + 1 : end + 1]

        logits = model_apply_fn(params, input_chunk)
        log_probs = jax.nn.log_softmax(logits[0], axis=-1)

        chunk_len = min(target_chunk.shape[0], log_probs.shape[0])
        token_log_probs = log_probs[jnp.arange(chunk_len), target_chunk[:chunk_len]]

        total_nll -= jnp.sum(token_log_probs).item()
        total_tokens += chunk_len

    if total_tokens == 0:
        return float("inf")

    avg_nll = total_nll / total_tokens
    return float(np.exp(min(avg_nll, 100.0)))
