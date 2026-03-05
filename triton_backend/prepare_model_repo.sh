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

set -euo pipefail

MAX_BATCH_SIZE=64
MAX_INPUT_LEN=608
MAX_SEQ_LEN=256
MAX_BEAM_WIDTH=3
MAX_QUEUE_DELAY_MS=0
APPLY_FLORENCE2_POST_PROCESS=true
KV_CACHE_FREE_GPU_MEM_FRACTION=0.5
CROSS_KV_CACHE_FRACTION=0.5
BATCH_SCHEDULER_POLICY="guaranteed_no_evict"
ENABLE_CHUNKED_CONTEXT="false"
LENGTH_PENALTY=1

MODEL_REPO="/opt/tritonserver/model_repo"

usage() {
    cat <<'EOF'
Usage: prepare_model_repo.sh [options]

Options:
  --output_dir DIR              Model repository root path (default: /opt/tritonserver/model_repo)
  --max_batch_size N            Max batch size for Triton config (default: 64)
  --max_input_len N             Max encoder input length for preprocessing check (default: 608)
  --max_seq_len N               Max decoder sequence length / max output length (default: 256)
  --max_beam_width N            Max beam width for Triton config (default: 3)
  --kv_cache_free_gpu_mem_fraction F  Fraction of free GPU memory for KV cache (default: 0.5)
  --cross_kv_cache_fraction F         Split between self/cross-attention KV cache (default: 0.5)
  --batch_scheduler_policy V    TRT-LLM batch scheduler policy (default: guaranteed_no_evict)
  --enable_chunked_context V    TRT-LLM chunked context toggle (default: false)
  --help                        Show this help
EOF
    exit 1
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --output_dir)           MODEL_REPO="$2"; shift 2 ;;
        --max_batch_size)       MAX_BATCH_SIZE="$2"; shift 2 ;;
        --max_input_len)        MAX_INPUT_LEN="$2"; shift 2 ;;
        --max_seq_len)          MAX_SEQ_LEN="$2"; shift 2 ;;
        --max_beam_width)       MAX_BEAM_WIDTH="$2"; shift 2 ;;
        --kv_cache_free_gpu_mem_fraction) KV_CACHE_FREE_GPU_MEM_FRACTION="$2"; shift 2 ;;
        --cross_kv_cache_fraction) CROSS_KV_CACHE_FRACTION="$2"; shift 2 ;;
        --batch_scheduler_policy) BATCH_SCHEDULER_POLICY="$2"; shift 2 ;;
        --enable_chunked_context) ENABLE_CHUNKED_CONTEXT="$2"; shift 2 ;;
        --help|-h)              usage ;;
        *) echo "Unknown option: $1"; usage ;;
    esac
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FILL_TEMPLATE_SCRIPT="${FILL_TEMPLATE_SCRIPT:-/app/tools/fill_template.py}"
TEMPLATE_REPO="${SCRIPT_DIR}/template"

MAX_QUEUE_DELAY_US=$(( MAX_QUEUE_DELAY_MS * 1000 ))

echo ">>> Preparing Triton model repository"
echo ""

mkdir -p "${MODEL_REPO}"

# Keep MODEL_REPO itself, only replace model directories under repo root.
echo "  Cleaning existing model directories in ${MODEL_REPO} ..."
rm -rf "${MODEL_REPO}/ensemble" \
       "${MODEL_REPO}/preprocessing" \
       "${MODEL_REPO}/vision" \
       "${MODEL_REPO}/tensorrt_llm" \
       "${MODEL_REPO}/postprocessing"
cp -r "${TEMPLATE_REPO}/." "${MODEL_REPO}/"

echo "  Copied template models to ${MODEL_REPO}"

TOKENIZER_DIR_FOR_PY_CONFIG="../tensorrt_llm/1/tokenizer"
TOKENIZER_DIR_FOR_TRTLLM_CONFIG="1/tokenizer"
DECODER_ENGINE_PATH_FOR_CONFIG="1/decoder"
ENCODER_ENGINE_PATH_FOR_CONFIG="1/encoder"
VISION_ENGINE_PATH_FOR_MOUNT="${MODEL_REPO}/vision/1/model.engine"

# Version directory is still required by tensorrt_plan. The engine file must
# be provided at runtime (e.g. bind-mount to ${MODEL_REPO}/vision/1/model.engine).
mkdir -p "${MODEL_REPO}/vision/1" "${MODEL_REPO}/tensorrt_llm/1"
# Python wrapper is not used for tensorrt_plan vision backend.
rm -f "${MODEL_REPO}/vision/1/model.py"

echo "  Filling preprocessing/config.pbtxt ..."
python3 "${FILL_TEMPLATE_SCRIPT}" -i "${MODEL_REPO}/preprocessing/config.pbtxt" \
    "tokenizer_dir:${TOKENIZER_DIR_FOR_PY_CONFIG},\
max_input_len:${MAX_INPUT_LEN},\
max_output_len:${MAX_SEQ_LEN},\
length_penalty:${LENGTH_PENALTY}"

echo "  Filling vision/config.pbtxt ..."
python3 "${FILL_TEMPLATE_SCRIPT}" -i "${MODEL_REPO}/vision/config.pbtxt" \
    "vision_triton_max_batch_size:${MAX_BATCH_SIZE}"

echo "  Filling tensorrt_llm/config.pbtxt ..."
python3 "${FILL_TEMPLATE_SCRIPT}" -i "${MODEL_REPO}/tensorrt_llm/config.pbtxt" \
    "triton_max_batch_size:${MAX_BATCH_SIZE},\
max_queue_delay_microseconds:${MAX_QUEUE_DELAY_US},\
max_beam_width:${MAX_BEAM_WIDTH},\
decoder_engine_dir:${DECODER_ENGINE_PATH_FOR_CONFIG},\
encoder_engine_dir:${ENCODER_ENGINE_PATH_FOR_CONFIG},\
kv_cache_free_gpu_mem_fraction:${KV_CACHE_FREE_GPU_MEM_FRACTION},\
cross_kv_cache_fraction:${CROSS_KV_CACHE_FRACTION},\
batch_scheduler_policy:${BATCH_SCHEDULER_POLICY},\
enable_chunked_context:${ENABLE_CHUNKED_CONTEXT},\
tokenizer_dir:${TOKENIZER_DIR_FOR_TRTLLM_CONFIG}"

echo "  Filling postprocessing/config.pbtxt ..."
python3 "${FILL_TEMPLATE_SCRIPT}" -i "${MODEL_REPO}/postprocessing/config.pbtxt" \
    "tokenizer_dir:${TOKENIZER_DIR_FOR_PY_CONFIG},\
apply_florence2_post_process:${APPLY_FLORENCE2_POST_PROCESS}"

echo ""
echo "============================================================"
echo "Prepare Done!"
echo "============================================================"
echo ""
echo "  Model repo root: ${MODEL_REPO}"
echo "    ensemble/"
echo "    preprocessing/    (tokenizer_dir in config: ${TOKENIZER_DIR_FOR_PY_CONFIG})"
echo "    vision/           (mount engine to ${VISION_ENGINE_PATH_FOR_MOUNT})"
echo "    tensorrt_llm/     (tokenizer_dir in config: ${TOKENIZER_DIR_FOR_TRTLLM_CONFIG}, encoder: ${ENCODER_ENGINE_PATH_FOR_CONFIG}, decoder: ${DECODER_ENGINE_PATH_FOR_CONFIG})"
echo "    postprocessing/   (tokenizer_dir in config: ${TOKENIZER_DIR_FOR_PY_CONFIG})"
echo ""
echo "  NOTE: this script does not create symlinks for external assets."
echo "        Ensure runtime mounts expose:"
echo "          - tokenizer dir: ${MODEL_REPO}/tensorrt_llm/1/tokenizer"
echo "          - encoder dir:   ${MODEL_REPO}/tensorrt_llm/1/encoder"
echo "          - decoder dir:   ${MODEL_REPO}/tensorrt_llm/1/decoder"
echo "          - vision engine: ${VISION_ENGINE_PATH_FOR_MOUNT}"
echo ""
echo "Launch tritonserver:"
echo "  tritonserver \\"
echo "      --model-repository=${MODEL_REPO} \\"
echo "      --disable-auto-complete-config \\"
echo "      --backend-config=python,shm-region-prefix-name=florence2_"
echo ""
