#!/bin/bash
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

# Build Florence2 TRT-LLM engines in one shot:
#   1) convert_checkpoint.py
#   2) trtllm-build for encoder/decoder
#   3) build_vision.py
#
# Usage:
#   bash build.sh \
#       --model_dir <model_dir> \
#       --engine_dir <engine_dir> \
#       [--max_batch_size N] \
#       [--max_seq_len N]
#
# Required args:
#   --model_dir DIR      Path to Florence2 HF model directory
#   --engine_dir DIR     Output directory for encoder/decoder/vision engines
#
# Optional args:
#   --max_batch_size N   Default 1
#   --max_seq_len N      Default 256 (decoder max sequence length)
#
# Notes:
#   - Effective max_output_len is derived as
#     (MAX_SEQ_LEN - decoder max input len).
#   - MAX_INPUT_LEN and MAX_ENCODER_INPUT_LEN are fixed to 608.

set -euo pipefail

usage() {
    echo "Usage: bash build.sh --model_dir <model_dir> --engine_dir <engine_dir> [--max_batch_size N] [--max_seq_len N]"
    exit 1
}

if [[ $# -lt 1 ]]; then
    usage
fi

MODEL_DIR=""
ENGINE_DIR=""

MAX_BATCH_SIZE=1
MAX_SEQ_LEN=256

# Fixed build settings
MAX_INPUT_LEN=608
MAX_ENCODER_INPUT_LEN=608
DECODER_MAX_INPUT_LEN=2
MAX_BEAM_WIDTH=3
DTYPE=float16
GPT_ATTENTION_PLUGIN="$DTYPE"

LEGACY_MAX_OUTPUT_LEN=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --model_dir)
            [[ $# -ge 2 ]] || usage
            MODEL_DIR="$2"
            shift 2
            ;;
        --engine_dir)
            [[ $# -ge 2 ]] || usage
            ENGINE_DIR="$2"
            shift 2
            ;;
        --max_batch_size)
            [[ $# -ge 2 ]] || usage
            MAX_BATCH_SIZE="$2"
            shift 2
            ;;
        --max_seq_len)
            [[ $# -ge 2 ]] || usage
            MAX_SEQ_LEN="$2"
            shift 2
            ;;
        --max_output_len)
            [[ $# -ge 2 ]] || usage
            LEGACY_MAX_OUTPUT_LEN="$2"
            shift 2
            ;;
        -h|--help)
            usage
            ;;
        *)
            echo "ERROR: unknown argument: $1"
            usage
            ;;
    esac
done

if [[ -z "$MODEL_DIR" || -z "$ENGINE_DIR" ]]; then
    echo "ERROR: --model_dir and --engine_dir are required."
    usage
fi

is_positive_int() {
    local v="$1"
    [[ "$v" =~ ^[0-9]+$ ]] && (( v > 0 ))
}

if [[ ! -d "$MODEL_DIR" ]]; then
    echo "ERROR: model_dir does not exist: $MODEL_DIR"
    exit 1
fi

if ! is_positive_int "$MAX_BATCH_SIZE"; then
    echo "ERROR: max_batch_size must be a positive integer, got: $MAX_BATCH_SIZE"
    exit 1
fi

if [[ -n "${LEGACY_MAX_OUTPUT_LEN}" ]]; then
    if ! is_positive_int "${LEGACY_MAX_OUTPUT_LEN}"; then
        echo "ERROR: max_output_len must be a positive integer, got: ${LEGACY_MAX_OUTPUT_LEN}"
        exit 1
    fi
    if [[ "${MAX_SEQ_LEN}" != "256" ]]; then
        echo "ERROR: --max_seq_len and deprecated --max_output_len cannot be used together."
        exit 1
    fi
    MAX_SEQ_LEN=$((LEGACY_MAX_OUTPUT_LEN + DECODER_MAX_INPUT_LEN))
    echo "WARNING: --max_output_len is deprecated; converted to --max_seq_len=${MAX_SEQ_LEN}."
fi

if ! is_positive_int "$MAX_SEQ_LEN"; then
    echo "ERROR: max_seq_len must be a positive integer, got: $MAX_SEQ_LEN"
    exit 1
fi

if (( MAX_SEQ_LEN <= DECODER_MAX_INPUT_LEN )); then
    echo "ERROR: max_seq_len must be > ${DECODER_MAX_INPUT_LEN}, got: ${MAX_SEQ_LEN}"
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODEL_DIR="$(cd "$MODEL_DIR" && pwd)"
ENGINE_DIR="$(mkdir -p "$ENGINE_DIR" && cd "$ENGINE_DIR" && pwd)"
CHECKPOINT_DIR="$(mktemp -d -p /tmp florence2_ckpt.XXXXXX)"

cleanup() {
    rm -rf "$CHECKPOINT_DIR"
}
trap cleanup EXIT

MAX_OUTPUT_LEN=$((MAX_SEQ_LEN - DECODER_MAX_INPUT_LEN))
MAX_PROMPT_EMBEDDING_TABLE_SIZE=$((577 * MAX_BATCH_SIZE))

echo "============================================"
echo "Building Florence2 TRT-LLM Engines"
echo "============================================"
echo "Model dir:      ${MODEL_DIR}"
echo "Engine dir:     ${ENGINE_DIR}"
echo "Checkpoint dir: ${CHECKPOINT_DIR}"
echo "Max batch size: ${MAX_BATCH_SIZE}"
echo "Max seq len:    ${MAX_SEQ_LEN}"
echo "Max output len: ${MAX_OUTPUT_LEN} (computed as max_seq_len - ${DECODER_MAX_INPUT_LEN})"
echo "Decoder in len: ${DECODER_MAX_INPUT_LEN}"
echo "Max input len:  ${MAX_INPUT_LEN}"
echo "Max encoder in: ${MAX_ENCODER_INPUT_LEN}"
echo "Prompt table:   ${MAX_PROMPT_EMBEDDING_TABLE_SIZE}"
echo "Max beam width: ${MAX_BEAM_WIDTH}"
echo "Dtype:          ${DTYPE}"
echo "============================================"

echo ""
echo ">>> Converting checkpoint..."
(cd /tmp && \
    python3 "${SCRIPT_DIR}/convert_checkpoint.py" \
        --model_dir "${MODEL_DIR}" \
        --output_dir "${CHECKPOINT_DIR}" \
        --dtype "${DTYPE}")

echo ""
echo ">>> Building ENCODER engine..."
(cd /tmp && \
    trtllm-build \
        --checkpoint_dir "${CHECKPOINT_DIR}/encoder" \
        --output_dir "${ENGINE_DIR}/encoder" \
        --gpt_attention_plugin "${GPT_ATTENTION_PLUGIN}" \
        --gemm_plugin "${DTYPE}" \
        --moe_plugin disable \
        --max_batch_size "${MAX_BATCH_SIZE}" \
        --max_input_len "${MAX_INPUT_LEN}" \
        --max_prompt_embedding_table_size "${MAX_PROMPT_EMBEDDING_TABLE_SIZE}" \
        --multiple_profiles enable)

echo ""
echo ">>> Building DECODER engine..."
(cd /tmp && \
    trtllm-build \
        --checkpoint_dir "${CHECKPOINT_DIR}/decoder" \
        --output_dir "${ENGINE_DIR}/decoder" \
        --gpt_attention_plugin "${GPT_ATTENTION_PLUGIN}" \
        --gemm_plugin "${DTYPE}" \
        --moe_plugin disable \
        --max_batch_size "${MAX_BATCH_SIZE}" \
        --max_input_len "${DECODER_MAX_INPUT_LEN}" \
        --max_seq_len "${MAX_SEQ_LEN}" \
        --max_encoder_input_len "${MAX_ENCODER_INPUT_LEN}" \
        --max_beam_width "${MAX_BEAM_WIDTH}" \
        --multiple_profiles enable)

echo ""
echo ">>> Building VISION engine..."
(cd /tmp && \
    python3 "${SCRIPT_DIR}/build_vision.py" \
        --model_dir "${MODEL_DIR}" \
        --output_dir "${ENGINE_DIR}/vision" \
        --max_batch_size "${MAX_BATCH_SIZE}" \
        --dtype "${DTYPE}")

echo ""
echo "============================================"
echo "Build complete!"
echo "  Encoder: ${ENGINE_DIR}/encoder"
echo "  Decoder: ${ENGINE_DIR}/decoder"
echo "  Vision:  ${ENGINE_DIR}/vision"
echo "============================================"
echo ""
echo "Run inference with:"
echo "  python3 run.py \\"
echo "      --model_dir ${MODEL_DIR} \\"
echo "      --engine_dir ${ENGINE_DIR} \\"
echo "      --vision_engine_dir ${ENGINE_DIR}/vision \\"
echo "      --task \"<CAPTION>\" --compare_hf"
