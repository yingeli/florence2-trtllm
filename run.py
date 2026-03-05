# SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""End-to-end inference script for Florence2 with TRT-LLM.

Pipeline:
  1. Load HF Florence2 model (for DaViT vision encoder + image projection)
     OR load DaViT TRT engine (if --vision_engine_dir is provided)
  2. Load TRT-LLM encoder + decoder engines via EncDecModelRunner
  3. Process image → DaViT → image_features  (PyTorch or TRT)
  4. Construct encoder input_ids with virtual tokens for image positions
  5. Run TRT encoder with prompt_embedding_table=image_features
  6. Run TRT decoder for autoregressive generation
  7. Optionally compare output with HF reference

The key insight is that EncoderModel supports ``prompt_embedding_table``
(p-tuning): tokens with ID >= vocab_size are "virtual tokens" whose
embeddings come from the prompt table. This lets us inject DaViT image
features without modifying any core TRT-LLM model code.

Usage:
    python run.py \
        --model_dir /path/to/Florence-2-large-ft \
        --engine_dir /tmp/florence2_engine \
        --task "<CAPTION>" \
        --image /path/to/image.jpg \
        --compare_hf

    # With TRT vision engine (no HF model needed for vision):
    python run.py \
        --model_dir /path/to/Florence-2-large-ft \
        --engine_dir /tmp/florence2_engine \
        --vision_engine_dir /tmp/florence2_engine/vision \
        --task "<CAPTION>"
"""
import argparse
import json
import os
import pathlib
import re
import sys
import time
import typing

import torch
from PIL import Image

import transformers

import tensorrt_llm
from tensorrt_llm import _utils
from tensorrt_llm import runtime
from tensorrt_llm.runtime import generation
from tensorrt_llm.runtime import session


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Florence2 TRT-LLM inference")
    parser.add_argument("--model_dir",
                        type=str,
                        required=True,
                        help="Path to Florence2 HF model directory")
    parser.add_argument("--engine_dir",
                        type=str,
                        required=True,
                        help="Path to TRT-LLM engine directory")
    parser.add_argument("--vision_engine_dir",
                        type=str,
                        default=None,
                        help="Path to DaViT vision TRT engine directory. "
                        "If provided, uses TRT engine instead of PyTorch "
                        "for vision encoding.")
    parser.add_argument("--task",
                        type=str,
                        default="<CAPTION>",
                        help="Florence2 task prompt (default: <CAPTION>)")
    parser.add_argument("--image",
                        type=str,
                        default=None,
                        help="Path to input image (optional, uses test "
                        "image if not provided)")
    parser.add_argument("--max_new_tokens",
                        type=int,
                        default=128,
                        help="Maximum new tokens to generate")
    parser.add_argument("--num_beams",
                        type=int,
                        default=3,
                        help="Number of beams for beam search")
    parser.add_argument(
        "--length_penalty",
        type=float,
        default=None,
        help="Length penalty for beam search (overrides default if set).",
    )
    parser.add_argument("--compare_hf",
                        action="store_true",
                        help="Compare TRT-LLM output with HF model output")
    parser.add_argument("--debug_mode",
                        action="store_true",
                        help="Enable debug mode")
    parser.add_argument(
        "--time_breakdown",
        action="store_true",
        help=
        "Print vision/text timing breakdown (adds CUDA synchronizations).",
    )
    parser.add_argument(
        "--output_log_probs",
        action="store_true",
        help="Return per-token log probabilities (higher memory usage).",
    )
    parser.add_argument(
        "--debug_topk",
        action="store_true",
        help=
        "Print top-k next-token logits/prob at the first HF/TRT divergence step (requires --compare_hf).",
    )
    parser.add_argument(
        "--debug_token_index",
        type=int,
        default=None,
        help=
        "Print per-beam selected token prob + top-k at this output token index (0-based; requires --debug_topk).",
    )
    parser.add_argument(
        "--debug_topk_k",
        type=int,
        default=10,
        help="Top-k value for --debug_topk (default: 10).",
    )
    parser.add_argument(
        "--post_process",
        action="store_true",
        help=
        "Run Florence2 processor post-processing. Required for non-text tasks like <OD>/<OCR>.",
    )
    parser.add_argument(
        "--penalty_length_offset",
        type=int,
        default=0,
        help="Subtract this value from context_lengths when computing beam "
        "search length penalty. Default: 0 (no adjustment).",
    )
    parser.add_argument("--log_level",
                        type=str,
                        default="error",
                        help="Logging level")
    return parser.parse_args()


def _load_json(path: pathlib.Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _extract_task_token(task: str) -> str:
    match = re.match(r"\s*(<[^>]+>)", task)
    if match is None:
        return task
    return match.group(1)


def _get_image_hw(image: Image.Image) -> typing.Tuple[int, int]:
    # Florence2 processor expects image size as (height, width).
    width, height = image.size
    return height, width


def _load_florence2_generation_defaults(model_dir: str) -> dict:
    """Load generation defaults from local Florence2 checkpoint files."""
    config = _load_json(pathlib.Path(model_dir) / "config.json")
    text_config = config.get("text_config", {})

    generation_config_path = pathlib.Path(model_dir) / "generation_config.json"
    generation_config = (_load_json(generation_config_path)
                         if generation_config_path.exists() else {})

    def _get(key: str, default=None):
        if key in generation_config:
            return generation_config[key]
        return text_config.get(key, default)

    return {
        "forced_bos_token_id": _get("forced_bos_token_id", None),
        "no_repeat_ngram_size": _get("no_repeat_ngram_size", None),
        "early_stopping": _get("early_stopping", None),
        "num_beams": _get("num_beams", None),
    }


def _decode_and_post_process(processor, output_ids: torch.Tensor, task: str,
                             image: Image.Image, enable_post_process: bool):
    can_post_process = enable_post_process and hasattr(processor,
                                                       "post_process_generation")
    decoded = processor.batch_decode(output_ids,
                                     skip_special_tokens=not can_post_process,
                                     clean_up_tokenization_spaces=False)
    if not can_post_process:
        return decoded, decoded

    task_token = _extract_task_token(task)
    image_hw = _get_image_hw(image)
    post_processed = []
    for text in decoded:
        post_processed.append(
            processor.post_process_generation(text=text,
                                              task=task_token,
                                              image_size=image_hw))
    return decoded, post_processed


def load_hf_model(model_dir):
    """Load Florence2 HF model for vision encoding (and optionally reference)."""
    sys.path.insert(0, model_dir)
    processor = transformers.AutoProcessor.from_pretrained(
        model_dir, trust_remote_code=True)
    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_dir,
        trust_remote_code=True,
        torch_dtype=torch.float16,
        attn_implementation="eager",
    )
    model = model.to('cuda').eval()
    return model, processor


def _trim_to_first_eos(ids: torch.Tensor,
                       eos_token_id: int,
                       start_search: int = 0) -> torch.Tensor:
    """Trim a 1D token id tensor to the first eos (inclusive).

    Florence2 (BART) uses decoder_start_token_id == eos_token_id, so sequences
    commonly begin with eos. Use start_search=1 to skip the initial token when
    searching for the end-of-sequence eos.
    """
    eos_positions = (ids == eos_token_id).nonzero(as_tuple=True)[0]
    if eos_positions.numel() == 0:
        return ids
    for pos in eos_positions.tolist():
        if int(pos) >= int(start_search):
            return ids[:int(pos) + 1]
    return ids


def _trim_output_ids(ids: torch.Tensor, eos_token_id: int,
                     pad_token_id: int) -> torch.Tensor:
    """Trim a 1D token id tensor to eos (inclusive) or strip trailing pads."""
    start_search = 0
    if ids.numel() > 1 and int(ids[0].item()) == int(eos_token_id):
        start_search = 1
    trimmed = _trim_to_first_eos(ids, eos_token_id, start_search=start_search)
    if (trimmed == eos_token_id).any():
        return trimmed
    non_pad = (trimmed != pad_token_id).nonzero(as_tuple=True)[0]
    if non_pad.numel() == 0:
        return trimmed[:1]
    return trimmed[:int(non_pad[-1].item()) + 1]


def _first_mismatch_index(a: torch.Tensor,
                          b: torch.Tensor) -> typing.Optional[int]:
    """Return first index where a[i] != b[i], or None if identical."""
    n = min(a.numel(), b.numel())
    for i in range(n):
        if int(a[i].item()) != int(b[i].item()):
            return i
    if a.numel() != b.numel():
        return n
    return None


def _compare_token_by_token(
    processor,
    trt_ids: torch.Tensor,
    hf_ids: torch.Tensor,
    eos_token_id: int,
    pad_token_id: int,
) -> bool:
    """Print a per-position token comparison table. Return True if identical."""
    trt_seq = _trim_output_ids(trt_ids.to('cpu').view(-1), eos_token_id,
                               pad_token_id)
    hf_seq = _trim_output_ids(hf_ids.to('cpu').view(-1), eos_token_id,
                              pad_token_id)
    max_len = max(trt_seq.numel(), hf_seq.numel())
    tokenizer = getattr(processor, "tokenizer", None)

    def _tok(tid):
        if tokenizer is None:
            return str(tid)
        try:
            return tokenizer.decode([tid], clean_up_tokenization_spaces=False)
        except Exception:
            return str(tid)

    mismatches = 0
    print(f"\n{'='*80}")
    print("Token-by-token comparison  (TRT-LLM vs HF)")
    print(f"  TRT seq_len={trt_seq.numel()}  HF seq_len={hf_seq.numel()}")
    print(f"{'='*80}")
    print(f"{'pos':>4}  {'match':>5}  {'TRT id':>7}  {'HF id':>7}  "
          f"{'TRT token':<20}  {'HF token':<20}")
    print(f"{'-'*4}  {'-'*5}  {'-'*7}  {'-'*7}  {'-'*20}  {'-'*20}")

    for i in range(max_len):
        trt_id = int(trt_seq[i].item()) if i < trt_seq.numel() else None
        hf_id = int(hf_seq[i].item()) if i < hf_seq.numel() else None
        match = trt_id == hf_id
        if not match:
            mismatches += 1
        trt_str = _tok(trt_id) if trt_id is not None else "<end>"
        hf_str = _tok(hf_id) if hf_id is not None else "<end>"
        trt_id_str = str(trt_id) if trt_id is not None else "-"
        hf_id_str = str(hf_id) if hf_id is not None else "-"
        marker = "  OK" if match else "DIFF"
        print(f"{i:>4}  {marker:>5}  {trt_id_str:>7}  {hf_id_str:>7}  "
              f"{trt_str!r:<20}  {hf_str!r:<20}")

    print(f"{'='*80}")
    if mismatches == 0:
        print(f"RESULT: IDENTICAL  ({max_len} tokens)")
    else:
        first = _first_mismatch_index(trt_seq, hf_seq)
        print(f"RESULT: DIFFERENT  ({mismatches} mismatches out of {max_len} "
              f"positions, first at pos {first})")
    print(f"{'='*80}\n")
    return mismatches == 0


def _token_id_to_text(processor, token_id: int) -> str:
    tokenizer = getattr(processor, "tokenizer", None)
    if tokenizer is None:
        return str(token_id)
    try:
        return tokenizer.decode([token_id],
                                clean_up_tokenization_spaces=False)
    except Exception:
        return str(token_id)


def _topk_from_logits(logits: torch.Tensor, k: int) -> typing.List[dict]:
    logits = logits.float()
    topk = min(k, int(logits.numel()))
    values, indices = torch.topk(logits, k=topk)
    log_z = torch.logsumexp(logits, dim=-1)
    probs = torch.exp(values - log_z)
    results = []
    for rank in range(topk):
        token_id = int(indices[rank].item())
        results.append({
            "rank": rank + 1,
            "token_id": token_id,
            "logit": float(values[rank].item()),
            "prob": float(probs[rank].item()),
        })
    return results


def _print_topk(
    title: str,
    processor,
    topk: typing.List[dict],
    highlight_token_ids: typing.Optional[typing.List[int]] = None,
):
    if highlight_token_ids is None:
        highlight_token_ids = []
    highlight_token_ids = set(int(x) for x in highlight_token_ids)
    print(title)
    for row in topk:
        token_id = int(row["token_id"])
        token_text = _token_id_to_text(processor, token_id)
        highlight = "*" if token_id in highlight_token_ids else " "
        print(
            f"{highlight}#{row['rank']:2d} id={token_id:5d} token={token_text!r} "
            f"logit={row['logit']:.4f} prob={row['prob']:.6f}")


def _debug_print_trt_token_index_topk(processor, args, trt_debug: dict,
                                      token_index: int) -> None:
    """Print per-beam selected token prob + top-k at a given TRT output token index."""
    if tensorrt_llm.mpi_rank() != 0:
        return
    if trt_debug is None:
        print("WARNING: TRT debug buffers are missing.")
        return
    if token_index is None:
        return

    output_ids = trt_debug.get("output_ids", None)
    parent_ids = trt_debug.get("parent_ids", None)
    generation_logits = trt_debug.get("generation_logits", None)
    decoder_max_input_length = int(trt_debug.get("decoder_max_input_length", 0))
    eos_token_id = int(trt_debug.get("eos_token_id", 0))
    pad_token_id = int(trt_debug.get("pad_token_id", 0))
    vocab_size = int(trt_debug.get("vocab_size", 0))
    num_beams = int(args.num_beams)

    if output_ids is None or parent_ids is None or generation_logits is None:
        print(
            "WARNING: Missing TRT buffers (need output_ids, parent_ids, generation_logits)."
        )
        return
    if output_ids.dim() != 3:
        raise ValueError(
            f"Unexpected TRT output_ids shape: {tuple(output_ids.shape)}")
    if parent_ids.dim() != 3:
        raise ValueError(
            f"Unexpected TRT parent_ids shape: {tuple(parent_ids.shape)}")
    if output_ids.size(0) != 1:
        print(
            f"WARNING: Only batch_size=1 debug is supported; got batch_size={output_ids.size(0)}"
        )
        return

    seq_len = int(output_ids.size(-1))
    if token_index < 0 or token_index >= seq_len:
        print(
            f"WARNING: token_index out of range: token_index={token_index}, seq_len={seq_len}"
        )
        return

    step = token_index - decoder_max_input_length
    print(
        f"\n=== TRT per-beam debug at token_index={token_index} (step={step}, ctx_len={decoder_max_input_length}) ==="
    )

    # Summarize current tokens across beams.
    beam_token_ids = []
    for beam_idx in range(num_beams):
        beam_token_ids.append(int(output_ids[0, beam_idx, token_index].item()))
    counts: typing.Dict[int, int] = {}
    for tid in beam_token_ids:
        counts[tid] = counts.get(tid, 0) + 1
    dup = [(tid, c) for tid, c in counts.items() if c > 1]
    if dup:
        dup_str = ", ".join(
            f"id={tid} token={_token_id_to_text(processor, tid)!r} count={c}"
            for tid, c in sorted(dup, key=lambda x: (-x[1], x[0])))
        print(f"Duplicate tokens at index {token_index}: {dup_str}")

    if step < 0:
        print(
            "Token index is inside the decoder prompt; no generation logits are available."
        )
        return
    if step >= len(generation_logits):
        print(
            f"WARNING: step out of range for generation_logits: step={step}, len={len(generation_logits)}"
        )
        return

    step_logits = generation_logits[step]
    if step_logits.dim() == 2:
        step_logits = step_logits.reshape(1, num_beams, -1)
    elif step_logits.dim() != 3:
        raise ValueError(
            f"Unexpected TRT generation_logits tensor shape at step={step}: {tuple(step_logits.shape)}"
        )

    def _backtrack_beam_index(final_beam_idx: int, start_pos: int,
                              target_pos: int) -> int:
        if target_pos > start_pos:
            raise ValueError(
                f"target_pos ({target_pos}) must be <= start_pos ({start_pos})")
        cur = int(final_beam_idx)
        for pos in range(int(start_pos), int(target_pos), -1):
            cur = int(parent_ids[0, cur, pos].item())
        return cur

    for beam_idx in range(num_beams):
        final_beam_idx = beam_idx
        final_seq = output_ids[0, final_beam_idx]
        trimmed = _trim_output_ids(final_seq,
                                   eos_token_id=eos_token_id,
                                   pad_token_id=pad_token_id)
        end_pos = int(trimmed.numel() - 1)
        if token_index > end_pos:
            print(f"beam={final_beam_idx}: token_index beyond eos "
                  f"(token_index={token_index}, eos_pos={end_pos}); skipping")
            continue

        beam_at_pos = _backtrack_beam_index(final_beam_idx,
                                            start_pos=end_pos,
                                            target_pos=token_index)
        parent_beam = int(parent_ids[0, beam_at_pos, token_index].item())
        token_id = int(output_ids[0, final_beam_idx, token_index].item())
        token_text = _token_id_to_text(processor, token_id)

        prev_token_id = None
        repeat = False
        if token_index > 0:
            prev_token_id = int(
                output_ids[0, final_beam_idx, token_index - 1].item())
            repeat = prev_token_id == token_id

        prefix = trimmed[:min(int(token_index + 1), int(trimmed.numel()))]
        prefix_text = processor.batch_decode(prefix.unsqueeze(0),
                                             skip_special_tokens=False,
                                             clean_up_tokenization_spaces=False)[0]

        parent_logits = step_logits[0, parent_beam, :vocab_size].float()
        if torch.isnan(parent_logits).any():
            nan_count = int(torch.isnan(parent_logits).sum().item())
            print(
                f"beam={final_beam_idx} beam_at_pos={beam_at_pos} parent={parent_beam} "
                f"token_id={token_id} token={token_text!r} "
                f"repeat={repeat} prob=<nan> (nan_count={nan_count})"
            )
            print(f"  prefix: {prefix_text!r}")
            continue

        log_z = torch.logsumexp(parent_logits, dim=-1)
        logit_value = float(parent_logits[token_id].item()
                            ) if token_id < vocab_size else float("nan")
        prob_value = float(torch.exp(parent_logits[token_id] - log_z).item()
                           ) if token_id < vocab_size else 0.0
        print(
            f"beam={final_beam_idx} beam_at_pos={beam_at_pos} parent={parent_beam} "
            f"token_id={token_id} token={token_text!r} "
            f"repeat={repeat} prob={prob_value:.6f} logit={logit_value:.4f}"
        )
        if prev_token_id is not None and repeat:
            prev_text = _token_id_to_text(processor, prev_token_id)
            print(
                f"  repeat_of: token_id={prev_token_id} token={prev_text!r}"
            )
        print(f"  prefix: {prefix_text!r}")

        topk = _topk_from_logits(parent_logits, k=args.debug_topk_k)
        _print_topk(
            f"  topk(parent={parent_beam})",
            processor,
            topk,
            highlight_token_ids=[token_id],
        )


def _debug_print_divergence_topk(processor, args, trt_ids: torch.Tensor,
                                 trt_debug: typing.Optional[dict],
                                 hf_ids: torch.Tensor,
                                 hf_debug: typing.Optional[dict]) -> None:
    """Print top-k distribution at the first divergence step for HF vs TRT."""
    if tensorrt_llm.mpi_rank() != 0:
        return
    if trt_debug is None or hf_debug is None:
        print(
            "WARNING: --debug_topk requires both TRT and HF debug buffers; ensure --compare_hf is set."
        )
        return
    if hf_debug.get("scores", None) is None:
        print("WARNING: HF debug scores are missing.")
        return
    if trt_debug.get("generation_logits", None) is None:
        print("WARNING: TRT debug generation_logits are missing.")
        return

    eos_token_id = int(trt_debug["eos_token_id"])
    pad_token_id = int(trt_debug["pad_token_id"])
    vocab_size = int(trt_debug["vocab_size"])
    num_beams = int(args.num_beams)

    trt_seq = _trim_output_ids(trt_ids[0].to('cpu'), eos_token_id, pad_token_id)
    hf_seq = _trim_output_ids(hf_ids[0].to('cpu'), eos_token_id, pad_token_id)
    mismatch_pos = _first_mismatch_index(trt_seq, hf_seq)
    if mismatch_pos is None:
        print("\n=== Debug top-k ===")
        print("No divergence found between best TRT/HF sequences.")
        return

    trt_token_id = int(trt_seq[mismatch_pos].item()
                       ) if mismatch_pos < trt_seq.numel() else None
    hf_token_id = int(hf_seq[mismatch_pos].item()
                      ) if mismatch_pos < hf_seq.numel() else None

    print("\n=== Debug top-k (first divergence) ===")
    print(f"First mismatch at token index={mismatch_pos}")
    if trt_token_id is not None:
        print(
            f"  TRT token: id={trt_token_id} token={_token_id_to_text(processor, trt_token_id)!r}"
        )
    else:
        print("  TRT token: <sequence ended>")
    if hf_token_id is not None:
        print(
            f"  HF  token: id={hf_token_id} token={_token_id_to_text(processor, hf_token_id)!r}"
        )
    else:
        print("  HF  token: <sequence ended>")

    # HF step index: position p>=1 maps to scores[p-1] (decoder_start at p=0).
    hf_step = mismatch_pos - 1
    # TRT step index: decoder_input_ids=[decoder_start, bos],
    # so position p>=2 maps to generation_logits[p-2].
    trt_step = mismatch_pos - 1
    print(
        f"  HF step={hf_step}, TRT step={trt_step} (k={args.debug_topk_k}, num_beams={num_beams})"
    )

    hf_scores = hf_debug["scores"]
    trt_logits = trt_debug["generation_logits"]
    if hf_step < 0 or hf_step >= len(hf_scores):
        print(
            f"WARNING: HF step index out of range: hf_step={hf_step}, len(scores)={len(hf_scores)}"
        )
        return
    if trt_step < 0 or trt_step >= len(trt_logits):
        print(
            f"WARNING: TRT step index out of range: trt_step={trt_step}, len(generation_logits)={len(trt_logits)}"
        )
        return

    highlight = []
    if trt_token_id is not None:
        highlight.append(trt_token_id)
    if hf_token_id is not None:
        highlight.append(hf_token_id)

    hf_step_scores = hf_scores[hf_step]
    if hf_step_scores.dim() == 2:
        hf_step_scores = hf_step_scores.reshape(1, num_beams, -1)
    elif hf_step_scores.dim() != 3:
        raise ValueError(
            f"Unexpected HF scores tensor shape at step={hf_step}: {hf_step_scores.shape}"
        )

    trt_step_logits = trt_logits[trt_step]
    if trt_step_logits.dim() == 2:
        trt_step_logits = trt_step_logits.reshape(1, num_beams, -1)
    elif trt_step_logits.dim() != 3:
        raise ValueError(
            f"Unexpected TRT generation_logits tensor shape at step={trt_step}: {trt_step_logits.shape}"
        )

    print("\n[HF] top-k next-token scores (often log-prob in beam search)")
    for beam_idx in range(num_beams):
        beam_scores = hf_step_scores[0, beam_idx, :vocab_size]
        if torch.isnan(beam_scores).any():
            nan_count = int(torch.isnan(beam_scores).sum().item())
            print(
                f"HF step={hf_step} beam={beam_idx}: scores contain NaN (count={nan_count}); skipping top-k"
            )
            continue
        topk = _topk_from_logits(beam_scores, k=args.debug_topk_k)
        _print_topk(f"HF step={hf_step} beam={beam_idx}",
                    processor,
                    topk,
                    highlight_token_ids=highlight)

    print("\n[TRT] top-k next-token logits")
    for beam_idx in range(num_beams):
        beam_logits = trt_step_logits[0, beam_idx, :vocab_size]
        if torch.isnan(beam_logits).any():
            nan_count = int(torch.isnan(beam_logits).sum().item())
            print(
                f"TRT step={trt_step} beam={beam_idx}: logits contain NaN (count={nan_count}); skipping top-k"
            )
            continue
        topk = _topk_from_logits(beam_logits, k=args.debug_topk_k)
        _print_topk(f"TRT step={trt_step} beam={beam_idx}",
                    processor,
                    topk,
                    highlight_token_ids=highlight)


class VisionTRTRunner:
    """Run the DaViT vision encoder via a TensorRT engine."""

    def __init__(self,
                 engine_dir,
                 stream: typing.Optional[torch.cuda.Stream] = None):
        engine_path = os.path.join(engine_dir, "model.engine")
        with open(engine_path, 'rb') as f:
            engine_buffer = f.read()
        self.session = session.Session.from_serialized_engine(engine_buffer)
        self.stream = stream
        if self.stream is None:
            self.stream = torch.cuda.current_stream()

        self._input_dtype = self.session.engine.get_tensor_dtype(
            "pixel_values")
        self._output_dtype = self.session.engine.get_tensor_dtype(
            "image_features")
        self._torch_input_dtype = _utils.trt_dtype_to_torch(self._input_dtype)
        self._torch_output_dtype = _utils.trt_dtype_to_torch(self._output_dtype)
        self._output_info_cache = {}

    @property
    def torch_input_dtype(self) -> torch.dtype:
        return self._torch_input_dtype

    def run(self, pixel_values: torch.Tensor):
        """Run vision encoder.

        Args:
            pixel_values: [B, 3, 768, 768] tensor on CUDA

        Returns:
            image_features: [B, 577, 1024] tensor on CUDA
        """
        if pixel_values.dtype != self._torch_input_dtype:
            raise ValueError(
                f"pixel_values.dtype ({pixel_values.dtype}) does not match vision engine input "
                f"dtype ({self._torch_input_dtype}). "
                "Convert inputs to the correct dtype or rebuild the vision engine."
            )

        inputs = {'pixel_values': pixel_values}
        batch_size = int(pixel_values.shape[0])
        output_info = self._output_info_cache.get(batch_size)
        if output_info is None:
            output_info = self.session.infer_shapes([
                session.TensorInfo('pixel_values', self._input_dtype,
                                   pixel_values.shape)
            ])
            self._output_info_cache[batch_size] = output_info

        outputs = {}
        for t in output_info:
            outputs[t.name] = torch.empty(tuple(t.shape),
                                          dtype=_utils.trt_dtype_to_torch(
                                              t.dtype),
                                          device=pixel_values.device)

        ok = self.session.run(inputs, outputs, self.stream.cuda_stream)
        if not ok:
            raise RuntimeError("Vision TRT engine execution failed")
        return outputs['image_features']


def encode_image(pixel_values, hf_model=None, vision_trt_runner=None):
    """Run DaViT vision encoder + projection to get image features.

    Args:
        pixel_values: [B, C, H, W] tensor
        hf_model: Florence2ForConditionalGeneration model (PyTorch path)
        vision_trt_runner: VisionTRTRunner instance (TRT path)

    Returns:
        image_features: [B, N_img, d_model] tensor (e.g., [1, 577, 1024])
    """
    if vision_trt_runner is not None:
        return vision_trt_runner.run(pixel_values)
    if hf_model is not None:
        with torch.no_grad():
            return hf_model._encode_image(pixel_values)
    raise ValueError("Either hf_model or vision_trt_runner must be provided")


def create_test_image(size=(768, 768)):
    """Create a simple test image if none is provided."""
    img = Image.new('RGB', size, color=(128, 128, 128))
    return img


def run_trt_florence2(args, tllm_model, processor, image, gen_defaults,
                      hf_model=None, vision_trt_runner=None):
    """Run Florence2 inference using TRT-LLM engines.

    The pipeline:
    1. Process image with DaViT (PyTorch) -> image_features [B, N_img, 1024]
    2. Tokenize task prompt -> task_token_ids [B, N_text]
    3. Create virtual token IDs for image positions:
       [vocab_size, vocab_size+1, ..., vocab_size+N_img-1]
    4. Concatenate: encoder_input_ids = [virtual_ids, task_token_ids]
    5. Run TRT encoder with prompt_embedding_table=image_features
    6. Run TRT decoder autoregressively

    We call encoder and decoder separately (instead of using
    EncDecModelRunner.generate()) to replicate HF Florence2's generation
    config: ``forced_bos_token_id=0`` (via decoder prefix ``[2, 0]``) and
    ``no_repeat_ngram_size=3`` (via SamplingConfig). Without these, the BART
    decoder's ~69% bos probability causes all beams to degenerate into
    infinite bos sequences.
    """
    # --- Process image ---
    inputs = processor(text=args.task, images=image, return_tensors="pt")
    pixel_values = inputs["pixel_values"].to(device='cuda')
    if vision_trt_runner is not None:
        pixel_values = pixel_values.to(dtype=vision_trt_runner.torch_input_dtype)
    else:
        pixel_values = pixel_values.to(dtype=torch.float16)
    task_ids = inputs["input_ids"].to(dtype=torch.int32, device='cuda')

    # Optional: timing breakdown (adds syncs, useful for stable measurements).
    vision_time_ms = None
    text_time_ms = None
    if args.time_breakdown:
        torch.cuda.synchronize()
        vision_tik = time.time()

    # ===================================================================
    # Generate timing: vision encode + encoder/decoder (excludes tokenize)
    # ===================================================================
    generate_tik = time.time()

    # Encode image with DaViT (TRT engine or PyTorch)
    image_features = encode_image(
        pixel_values, hf_model=hf_model,
        vision_trt_runner=vision_trt_runner)  # [B, N_img, 1024]
    if args.time_breakdown:
        torch.cuda.synchronize()
        vision_time_ms = (time.time() - vision_tik) * 1000
        text_tik = time.time()
    batch_size = image_features.shape[0]
    n_img_tokens = image_features.shape[1]

    if tensorrt_llm.mpi_rank() == 0:
        print(f"Image features shape: {image_features.shape}")
        print(f"Task input_ids shape: {task_ids.shape}")
        print(f"Task: {args.task}")

    # --- Construct encoder input_ids with virtual tokens ---
    vocab_size = tllm_model.encoder_model_config.vocab_size
    virtual_ids = torch.arange(vocab_size,
                               vocab_size + n_img_tokens,
                               dtype=torch.int32,
                               device=task_ids.device).unsqueeze(0)
    virtual_ids = virtual_ids.expand(batch_size, -1)
    encoder_input_ids = torch.cat([virtual_ids, task_ids], dim=1)

    if tensorrt_llm.mpi_rank() == 0:
        print(f"Encoder input_ids shape: {encoder_input_ids.shape}")
        print(
            f"  Virtual image tokens: {n_img_tokens} (IDs {vocab_size}..{vocab_size + n_img_tokens - 1})"
        )
        print(f"  Task text tokens: {task_ids.shape[1]}")

    # --- Decoder config (prefer HF model_dir config for parity) ---
    florence2_config = _load_json(pathlib.Path(args.model_dir) / "config.json")
    text_config = florence2_config.get("text_config", {})
    decoder_start_token_id = text_config.get("decoder_start_token_id", 2)
    eos_token_id = florence2_config.get("eos_token_id",
                                        text_config.get("eos_token_id", 2))
    bos_token_id = florence2_config.get("bos_token_id",
                                        text_config.get("bos_token_id", 0))
    pad_token_id = florence2_config.get("pad_token_id",
                                        text_config.get("pad_token_id", 1))
    if gen_defaults.get("forced_bos_token_id") is not None:
        bos_token_id = gen_defaults["forced_bos_token_id"]

    # Decoder prefix: [decoder_start_token_id, bos_token_id] = [2, 0].
    # This matches HF Florence2's forced_bos_token_id behavior.
    decoder_input_ids = torch.IntTensor(
        [[decoder_start_token_id, bos_token_id]]).to('cuda')
    decoder_input_ids = decoder_input_ids.repeat((batch_size, 1))

    max_prompt_table = tllm_model.encoder_model_config.max_prompt_embedding_table_size
    required_prompt_table = batch_size * n_img_tokens
    if required_prompt_table > max_prompt_table:
        raise ValueError(
            "Prompt embedding table too small for batch size. "
            f"Required {required_prompt_table}, engine supports {max_prompt_table}. "
            "Rebuild encoder engine with "
            f"--max_prompt_embedding_table_size {required_prompt_table}.")

    # --- Prompt embedding table setup ---
    prompt_dtype = _utils.str_dtype_to_torch(tllm_model.encoder_model_config.dtype)
    prompt_embedding_table = image_features.to(dtype=prompt_dtype).contiguous(
    ).reshape(-1, image_features.shape[-1])
    enc_remove_pad = tllm_model.encoder_model_config.remove_input_padding
    if enc_remove_pad:
        # In remove-padding mode, tasks must align with the flattened input ids
        # produced by EncDecModelRunner.process_input().
        input_lengths = 1 + (encoder_input_ids[:, 1:] != pad_token_id).sum(
            dim=1, dtype=torch.int32)
        prompt_tasks = torch.repeat_interleave(
            torch.arange(batch_size, dtype=torch.int32, device='cuda'),
            input_lengths,
        )
    else:
        # In padding mode, tasks is [batch_size, 1] and will be broadcast to [batch_size, seq_len].
        prompt_tasks = torch.arange(batch_size, dtype=torch.int32,
                                    device='cuda').unsqueeze(1)
    prompt_vocab_size = torch.tensor([n_img_tokens],
                                     dtype=torch.int32,
                                     device='cuda')

    # ===================================================================
    # Run encoder and decoder separately for full beam search control
    # ===================================================================
    # --- 1. Process encoder input ---
    (encoder_input_ids_p, encoder_input_lengths, encoder_max_input_length,
     prompt_tasks_p, _) = tllm_model.process_input(encoder_input_ids,
                                                    enc_remove_pad,
                                                    pad_token_id,
                                                    prompt_tasks, None)

    # --- 2. Run encoder ---
    encoder_output = tllm_model.encoder_run(
        encoder_input_ids_p,
        encoder_input_lengths,
        encoder_max_input_length,
        debug_mode=args.debug_mode,
        prompt_embedding_table=prompt_embedding_table,
        prompt_tasks=prompt_tasks_p,
        prompt_vocab_size=prompt_vocab_size,
    )

    # --- 3. Process decoder input ---
    dec_remove_pad = tllm_model.decoder_model_config.remove_input_padding
    (decoder_input_ids_p, decoder_input_lengths, decoder_max_input_length, _,
     _) = tllm_model.process_input(decoder_input_ids, dec_remove_pad,
                                   pad_token_id, None, None)

    # --- 4. Create SamplingConfig ---
    sampling_config = generation.SamplingConfig(end_id=eos_token_id,
                                                pad_id=pad_token_id,
                                                num_beams=args.num_beams,
                                                min_length=1,
                                                return_dict=True)

    sampling_kwargs = {}
    sampling_kwargs["output_cum_log_probs"] = True
    if args.output_log_probs:
        sampling_kwargs["output_log_probs"] = True
    if args.length_penalty is not None:
        sampling_kwargs["length_penalty"] = float(args.length_penalty)
    if gen_defaults.get("no_repeat_ngram_size") is not None:
        sampling_kwargs["no_repeat_ngram_size"] = gen_defaults[
            "no_repeat_ngram_size"]
    if gen_defaults.get("early_stopping") is not None:
        sampling_kwargs["early_stopping"] = int(gen_defaults["early_stopping"])
    sampling_config.update(**sampling_kwargs)

    # --- 5. Setup decoder session ---
    tllm_model.decoder_session.setup(
        decoder_input_lengths.size(0),
        decoder_max_input_length,
        args.max_new_tokens,
        args.num_beams,
        max_attention_window_size=None,
        encoder_max_input_length=encoder_max_input_length,
    )

    # --- 5b. Patch length penalty to match HF ---
    #
    # To fix: subtract penalty_length_offset from context_lengths before it
    # reaches the C++ beam search and gather_tree kernels. We monkey-patch
    # two methods on the session:
    #   1. dynamic_decoder.forward() — per-step beam scoring
    #   2. finalize_decoder() — final beam re-ranking
    #
    # context_lengths is still used unchanged for position_ids,
    # sequence_length_buffer, and other generation mechanics.
    _penalty_offset = args.penalty_length_offset
    if _penalty_offset != 0:
        if tensorrt_llm.mpi_rank() == 0:
            print(f"[penalty_length_offset={_penalty_offset}] "
                  f"Adjusting beam search inputLengths by -{_penalty_offset} "
                  f"to match HF length penalty computation.")
        _session = tllm_model.decoder_session

        # Patch dynamic_decoder.forward: context_lengths is positional arg #9
        # (0-indexed) = inputLengthsOpt in dynamicDecodeOp.cpp.
        # dynamic_decoder is a TorchScript custom class (DynamicDecodeOp), so
        # we cannot set attributes on it directly.  Instead, replace the
        # session attribute with a thin Python wrapper.
        class _DynamicDecoderProxy:
            """Proxy that adjusts inputLengthsOpt before forwarding."""

            def __init__(self, original, offset):
                object.__setattr__(self, '_original', original)
                object.__setattr__(self, '_offset', offset)

            def forward(self, *args):
                args = list(args)
                if args[9] is not None:
                    args[9] = args[9] - self._offset
                return self._original.forward(*args)

            def __getattr__(self, name):
                return getattr(self._original, name)

        _session.dynamic_decoder = _DynamicDecoderProxy(
            _session.dynamic_decoder, _penalty_offset)

        # Patch finalize_decoder: context_lengths is the first positional arg,
        # passed as tiled_input_lengths to gather_tree in gatherTreeOp.cpp.
        _orig_finalize = _session.finalize_decoder

        def _patched_finalize(context_lengths, *args, **kwargs):
            return _orig_finalize(context_lengths - _penalty_offset,
                                  *args, **kwargs)

        _session.finalize_decoder = _patched_finalize

    # --- 6. Run decoder ---
    # Default decoding behavior (HF parity):
    # decoder_input_ids=[decoder_start, bos] = [2, 0].
    # no_repeat_ngram_size=3 prevents degenerate repetitions (same as HF).
    tllm_output = tllm_model.decoder_session.decode(
        decoder_input_ids_p,
        decoder_input_lengths,
        sampling_config,
        encoder_output=encoder_output,
        encoder_input_lengths=encoder_input_lengths,
        output_generation_logits=args.debug_topk,
        return_dict=True,
    )
    torch.cuda.synchronize()
    generate_tok = time.time()
    if args.time_breakdown:
        text_time_ms = (generate_tok - text_tik) * 1000

    tllm_output_ids = tllm_output['output_ids']
    tllm_cum_log_probs = tllm_output.get('cum_log_probs', None)
    # TRT-LLM's finalize_decoder (gather_tree) already ranks beams using
    # length_penalty, so beam 0 is the best beam.  Do NOT re-rank by raw
    # cum_log_probs — that ignores length penalty and biases towards shorter
    # sequences.
    output_ids = tllm_output_ids[:, 0, :]  # [B, max_output_len]
    # Verify that the output starts with [decoder_start, bos, ...].
    if tensorrt_llm.mpi_rank() == 0:
        first_two = output_ids[0, :2].tolist()
        expected = [decoder_start_token_id, bos_token_id]
        assert first_two == expected, (
            f"Expected output_ids to start with {expected}, got {first_two}")

    # Debug: show all beams
    if args.debug_mode and tensorrt_llm.mpi_rank() == 0:
        for beam_idx in range(tllm_output_ids.shape[1]):
            beam_ids = tllm_output_ids[0, beam_idx, :]
            beam_text = processor.batch_decode(beam_ids.unsqueeze(0),
                                               skip_special_tokens=False,
                                               clean_up_tokenization_spaces=False)
            has_eos = (beam_ids == eos_token_id).any().item()
            print(f"  Beam {beam_idx}: text={beam_text}, "
                  f"has_eos={has_eos}")
            print(f"    Tokens: {beam_ids[:30].tolist()}")
        print("  Best beam: 0 (ranked by gather_tree with length penalty)")

    decoded_text = None
    final_answer = None
    if tensorrt_llm.mpi_rank() == 0:
        output_ids_cpu = output_ids.to('cpu')
        decoded_text, final_answer = _decode_and_post_process(
            processor,
            output_ids_cpu,
            args.task,
            image,
            enable_post_process=args.post_process,
        )

    if tensorrt_llm.mpi_rank() == 0:
        print("\n--------------------------------------")
        print(f"TRT-LLM decoded text: {decoded_text}")
        if args.post_process:
            print(f"TRT-LLM post-processed: {final_answer}")
        print(f"TRT-LLM output_ids: {output_ids_cpu}")
        if tllm_cum_log_probs is not None:
            cum_lp = tllm_cum_log_probs.detach().cpu()
            for bi in range(cum_lp.size(0)):
                for beam_idx in range(cum_lp.size(1)):
                    beam_ids = tllm_output_ids[bi, beam_idx, :]
                    trimmed = _trim_output_ids(beam_ids.cpu(), eos_token_id,
                                               pad_token_id)
                    seq_len = int(trimmed.numel())
                    beam_text = processor.batch_decode(
                        trimmed.unsqueeze(0),
                        skip_special_tokens=True,
                        clean_up_tokenization_spaces=False)[0]
                    print(
                        f"TRT-LLM beam {beam_idx}: cum_log_prob={cum_lp[bi, beam_idx].item():.4f} "
                        f"seq_len={seq_len} text={beam_text!r}"
                    )
        if args.time_breakdown:
            print(f"TRT vision time: {vision_time_ms:.1f}ms")
            print(f"TRT text time:   {text_time_ms:.1f}ms")
        print(
            f"TRT generate time (vision+text): {(generate_tok - generate_tik) * 1000:.1f}ms"
        )
        print(f"Precision: {tllm_model.encoder_model_config.dtype}")
        print("--------------------------------------")

    trt_debug = None
    if args.debug_topk:
        trt_debug = {
            "generation_logits": tllm_output.get("generation_logits", None),
            "parent_ids": tllm_model.decoder_session.parent_ids.detach().to('cpu'),
            "output_ids": tllm_output_ids.detach().to('cpu'),
            "decoder_max_input_length": decoder_max_input_length,
            "eos_token_id": eos_token_id,
            "bos_token_id": bos_token_id,
            "pad_token_id": pad_token_id,
            "vocab_size": vocab_size,
        }

    return final_answer, output_ids, trt_debug


def run_hf_reference(hf_model, processor, image, args, gen_defaults):
    """Run HF Florence2 model for reference comparison."""
    inputs = processor(text=args.task, images=image, return_tensors="pt")
    inputs = {
        k: v.to('cuda') if isinstance(v, torch.Tensor) else v
        for k, v in inputs.items()
    }
    # Match model dtype for pixel_values (model loaded in fp16)
    if "pixel_values" in inputs:
        inputs["pixel_values"] = inputs["pixel_values"].to(
            dtype=torch.float16)

    # Disable dynamic cache creation for transformers >=4.57 compat.
    # Florence2's custom BART code uses legacy tuple-of-tuples cache format
    # which is incompatible with EncoderDecoderCache.
    orig_supports_cache = None
    if hasattr(hf_model.language_model, "_supports_default_dynamic_cache"):
        orig_supports_cache = hf_model.language_model._supports_default_dynamic_cache
        hf_model.language_model._supports_default_dynamic_cache = lambda: False

    with torch.no_grad():
        tik = time.time()
        generate_kwargs = dict(
            input_ids=inputs["input_ids"],
            pixel_values=inputs["pixel_values"],
            max_new_tokens=args.max_new_tokens,
            num_beams=args.num_beams,
        )
        if args.length_penalty is not None:
            generate_kwargs["length_penalty"] = float(args.length_penalty)
        if gen_defaults.get("early_stopping") is not None:
            generate_kwargs["early_stopping"] = gen_defaults["early_stopping"]
        if gen_defaults.get("no_repeat_ngram_size") is not None:
            generate_kwargs["no_repeat_ngram_size"] = gen_defaults[
                "no_repeat_ngram_size"]
        if gen_defaults.get("forced_bos_token_id") is not None:
            generate_kwargs["forced_bos_token_id"] = gen_defaults[
                "forced_bos_token_id"]
        generate_kwargs["return_dict_in_generate"] = True
        generate_kwargs["output_scores"] = True
        generate_kwargs["num_return_sequences"] = args.num_beams
        hf_output = hf_model.generate(**generate_kwargs)
        torch.cuda.synchronize()
        tok = time.time()

    if orig_supports_cache is not None:
        hf_model.language_model._supports_default_dynamic_cache = orig_supports_cache

    output_ids = hf_output.sequences
    hf_output_text, hf_final_answer = _decode_and_post_process(
        processor,
        output_ids[:1],
        args.task,
        image,
        enable_post_process=args.post_process,
    )
    if tensorrt_llm.mpi_rank() == 0:
        print("\n--------------------------------------")
        print(f"HF decoded text: {hf_output_text}")
        if args.post_process:
            print(f"HF post-processed: {hf_final_answer}")
        print(f"HF output_ids: {output_ids}")

        # Compute per-token transition scores and sum to get raw cum_log_probs.
        beam_indices = getattr(hf_output, "beam_indices", None)
        hf_scores = getattr(hf_output, "scores", None)
        hf_cum_log_probs = None
        if hf_scores is not None and beam_indices is not None:
            transition_scores = hf_model.compute_transition_scores(
                output_ids, hf_scores, beam_indices, normalize_logits=True)
            hf_cum_log_probs = transition_scores.sum(dim=-1)

        hf_sequences_scores = getattr(hf_output, "sequences_scores", None)
        num_return = output_ids.size(0)
        for idx in range(num_return):
            seq = output_ids[idx]
            seq_text = processor.batch_decode(
                seq.unsqueeze(0),
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False)[0]
            parts = [f"HF beam {idx}:"]
            if hf_cum_log_probs is not None:
                parts.append(
                    f"cum_log_prob={hf_cum_log_probs[idx].item():.4f}")
            if hf_sequences_scores is not None:
                parts.append(
                    f"sequences_score={hf_sequences_scores[idx].item():.4f}")
            parts.append(f"seq_len={seq.numel()}")
            parts.append(f"text={seq_text!r}")
            print(" ".join(parts))

        print(f"HF generate time: {(tok - tik) * 1000:.1f}ms")
        print("--------------------------------------")

    hf_debug = None
    if args.debug_topk:
        hf_debug = {
            "scores": list(hf_output.scores) if hf_output.scores is not None else None,
        }

    return hf_final_answer, output_ids, hf_debug


def main():
    args = parse_arguments()
    tensorrt_llm.logger.set_level(args.log_level)

    if args.debug_token_index is not None and not args.debug_topk:
        raise ValueError("--debug_token_index requires --debug_topk")

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is required to run Florence2 with TensorRT-LLM engines.")

    # Create TRT-LLM runner first so it sets the CUDA device for this MPI rank.
    tllm_model = runtime.EncDecModelRunner.from_engine(
        "enc_dec", args.engine_dir, debug_mode=args.debug_mode)
    torch.cuda.set_device(tllm_model.device)
    if (tllm_model.encoder_model_config is not None
            and tllm_model.encoder_model_config.remove_input_padding
            != tllm_model.decoder_model_config.remove_input_padding):
        raise RuntimeError(
            "Encoder and decoder engines must be built with the same "
            f"--remove_input_padding setting, but got encoder.remove_input_padding="
            f"{tllm_model.encoder_model_config.remove_input_padding} and "
            f"decoder.remove_input_padding={tllm_model.decoder_model_config.remove_input_padding}. "
            "Rebuild both engines with matching settings."
        )

    gen_defaults = _load_florence2_generation_defaults(args.model_dir)
    if tensorrt_llm.mpi_rank() == 0:
        print(f"Generation defaults: {gen_defaults}")

    # Load vision TRT engine if provided
    vision_trt_runner = None
    if args.vision_engine_dir is not None:
        if tensorrt_llm.mpi_rank() == 0:
            print(f"Loading vision TRT engine from {args.vision_engine_dir}")
        vision_trt_runner = VisionTRTRunner(args.vision_engine_dir,
                                            stream=tllm_model.stream)

    # Load HF model: needed for PyTorch DaViT (when no vision engine) or
    # for --compare_hf reference. When vision engine is provided and
    # --compare_hf is not set, we only need the processor (not the full model).
    hf_model = None
    need_hf_model = (vision_trt_runner is None) or args.compare_hf
    if need_hf_model:
        if tensorrt_llm.mpi_rank() == 0:
            print(f"Loading Florence2 model from {args.model_dir}")
        hf_model, processor = load_hf_model(args.model_dir)
    else:
        if tensorrt_llm.mpi_rank() == 0:
            print(f"Loading Florence2 processor from {args.model_dir} "
                  "(skipping full model — using vision TRT engine)")
        sys.path.insert(0, args.model_dir)
        processor = transformers.AutoProcessor.from_pretrained(
            args.model_dir, trust_remote_code=True)

    if not args.post_process:
        task_token = _extract_task_token(args.task)
        post_types = getattr(processor, "tasks_answer_post_processing_type", {})
        post_type = post_types.get(task_token, "pure_text")
        if post_type != "pure_text":
            args.post_process = True
            if tensorrt_llm.mpi_rank() == 0:
                print(
                    f"Auto-enabling --post_process for task {task_token} (type={post_type})"
                )

    # Load or create image
    if args.image is not None:
        image = Image.open(args.image).convert("RGB")
        if tensorrt_llm.mpi_rank() == 0:
            print(f"Loaded image: {args.image} ({image.size})")
    else:
        image = create_test_image()
        if tensorrt_llm.mpi_rank() == 0:
            print("Using test image (768x768 gray)")

    # Run TRT-LLM inference
    trt_answer, trt_ids, trt_debug = run_trt_florence2(
        args, tllm_model, processor, image, gen_defaults,
        hf_model=hf_model, vision_trt_runner=vision_trt_runner)

    if args.debug_topk and trt_debug is not None and tensorrt_llm.mpi_rank() == 0:
        if args.debug_token_index is not None:
            _debug_print_trt_token_index_topk(processor, args, trt_debug,
                                              args.debug_token_index)
        else:
            # Dump per-beam top-k for ALL generated token indices.
            gen_logits = trt_debug.get("generation_logits", None)
            dml = int(trt_debug.get("decoder_max_input_length", 0))
            if gen_logits is not None:
                n_steps = len(gen_logits)
                print(f"\n{'='*60}")
                print(f"Per-step per-beam top-k dump  (steps=0..{n_steps-1}, "
                      f"token_indices={dml}..{dml + n_steps - 1})")
                print(f"{'='*60}")
                for step_idx in range(n_steps):
                    token_idx = dml + step_idx
                    _debug_print_trt_token_index_topk(
                        processor, args, trt_debug, token_idx)

    # Optionally compare with HF
    if args.compare_hf:
        if hf_model is None:
            raise RuntimeError(
                "--compare_hf requires the HF model but it was not loaded")
        if tensorrt_llm.mpi_rank() == 0:
            hf_answer, hf_ids, hf_debug = run_hf_reference(
                hf_model, processor, image, args, gen_defaults)

            if args.debug_topk:
                _debug_print_divergence_topk(processor, args, trt_ids, trt_debug,
                                             hf_ids, hf_debug)

            # --- Token-by-token comparison ---
            florence2_config = _load_json(
                pathlib.Path(args.model_dir) / "config.json")
            text_config = florence2_config.get("text_config", {})
            eos_id = florence2_config.get(
                "eos_token_id", text_config.get("eos_token_id", 2))
            pad_id = florence2_config.get(
                "pad_token_id", text_config.get("pad_token_id", 1))

            identical = _compare_token_by_token(
                processor, trt_ids[0], hf_ids[0], eos_id, pad_id)

            # Also show string-level comparison
            import difflib

            def _to_compare_str(obj) -> str:
                try:
                    return json.dumps(obj,
                                      sort_keys=True,
                                      ensure_ascii=False)
                except TypeError:
                    return str(obj)

            match_rate = difflib.SequenceMatcher(
                None, _to_compare_str(trt_answer),
                _to_compare_str(hf_answer)).ratio()
            print(f"=== String comparison ===")
            print(f"TRT-LLM: {trt_answer}")
            print(f"HF:      {hf_answer}")
            print(f"Match rate: {match_rate:.4f}")

            if identical:
                print("PASS: TRT-LLM and HF sequences are token-identical")
            elif match_rate > 0.8:
                print("PARTIAL: strings similar but tokens differ")
            else:
                print(
                    f"FAIL: sequences differ (string match {match_rate:.4f})")


if __name__ == "__main__":
    main()
