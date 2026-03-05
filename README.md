# Florence-2 with TensorRT-LLM

Florence-2 is a multimodal encoder-decoder model (DaViT vision encoder + BART language model) from Microsoft. This guide shows how to accelerate the BART text backbone with TRT-LLM, with an optional TRT engine for the DaViT vision encoder.

## Architecture

```
Image → DaViT → image_projection → LayerNorm → image_features [B, 577, 1024]
Task prompt → tokenizer → text_ids [B, N_text]

[virtual_image_ids, text_ids] → TRT Encoder (prompt_embedding_table=image_features) → encoder_output
[decoder_start, bos] → TRT Decoder (cross-attention to encoder_output) → generated text
```

Image features are injected into the encoder via the **prompt embedding table** (p-tuning mechanism): tokens with ID >= vocab_size are "virtual tokens" whose embeddings come from the DaViT output. No core TRT-LLM code modifications are needed.

## Quick Start

### 0. Download weights

```bash
git clone https://huggingface.co/microsoft/Florence-2-large-ft
```

### 1. Build all engines (convert + encoder/decoder + vision)

```bash
bash build.sh \
    --model_dir /workspaces/florence2/Florence-2-large-ft \
    --engine_dir /tmp/florence2_engine \
    --max_batch_size 1 \
    --max_seq_len 256
```

`build.sh` now takes `--max_seq_len` directly. The effective generation
`max_output_len` is computed as `max_seq_len - 2` (decoder prefix length is 2:
`[decoder_start, bos]`).

### 2. (Optional) Build manually

If you want full control over each stage, run convert/build commands manually:

```bash
python convert_checkpoint.py \
    --model_dir /workspaces/florence2/Florence-2-large-ft \
    --output_dir /tmp/florence2_ckpt \
    --dtype float16
```

Then build encoder/decoder:

```bash
# Encoder (max_prompt_embedding_table_size=577 for 768x768 images)
trtllm-build \
    --checkpoint_dir /tmp/florence2_ckpt/encoder \
    --output_dir /tmp/florence2_engine/encoder \
    --gpt_attention_plugin float16 \
    --gemm_plugin float16 \
    --max_batch_size 1 \
    --max_input_len 608 \
    --max_prompt_embedding_table_size 577

# Decoder (max_input_len=2 for [decoder_start, bos] prefix)
trtllm-build \
    --checkpoint_dir /tmp/florence2_ckpt/decoder \
    --output_dir /tmp/florence2_engine/decoder \
    --gpt_attention_plugin float16 \
    --gemm_plugin float16 \
    --max_batch_size 1 \
    --max_input_len 2 \
    --max_seq_len 256 \
    --max_encoder_input_len 608 \
    --max_beam_width 3
```

### 3. Run inference

```bash
python run.py \
    --model_dir /workspaces/florence2/Florence-2-large-ft \
    --engine_dir /tmp/florence2_engine \
    --task "<CAPTION>" \
    --image /workspaces/florence2/images/car.jpg \
    --compare_hf
```

Supported task tokens: `<CAPTION>`, `<DETAILED_CAPTION>`, `<MORE_DETAILED_CAPTION>`, `<OD>`, `<OCR>`, etc. For non-text tasks like `<OD>`/`<OCR>`, pass `--post_process` to parse structured outputs (bboxes, polygons, etc.).

### 4. (Optional) Build vision TRT engine

Export the DaViT vision encoder as a TRT engine. This replaces the PyTorch DaViT path with a TRT engine that takes `pixel_values [B, 3, 768, 768]` and outputs `image_features [B, 577, 1024]`.

The engine is built with FP32 internal precision by default (ONNX I/O stays FP16). This matches PyTorch's behaviour where LayerNorm and softmax internally upcast to FP32, avoiding numerical divergence that can affect beam search output.

```bash
python build_vision.py \
    --model_dir /workspaces/florence2/Florence-2-large-ft \
    --output_dir /tmp/florence2_engine/vision \
    --max_batch_size 1
```

Or include vision build in the one-shot shell script:

```bash
bash build.sh \
    --model_dir /path/to/Florence-2-large-ft \
    --engine_dir /tmp/florence2_engine \
    --max_batch_size 1 \
    --max_seq_len 256
```

### 5. Run with vision TRT engine

When `--vision_engine_dir` is provided, the HF model is not loaded for vision encoding (saves ~2GB memory and avoids PyTorch overhead). The HF model is still loaded if `--compare_hf` is set.

```bash
python run.py \
    --model_dir /workspaces/florence2/Florence-2-large-ft \
    --engine_dir /tmp/florence2_engine \
    --vision_engine_dir /tmp/florence2_engine/vision \
    --task "<CAPTION>" \
    --image /workspaces/florence2/images/car.jpg \
    --compare_hf
```

## Beam Search Configuration

Florence-2's BART decoder assigns ~69% probability to `<s>` (bos, token 0) at every step. Without mitigation, beam search degenerates into infinite bos sequences. The HF model handles this via `text_config`/`generation_config` settings that `run.py` replicates:

| HF Setting | Value | TRT-LLM Equivalent |
|------------|-------|---------------------|
| `forced_bos_token_id` | 0 | `decoder_input_ids = [decoder_start, bos]` |
| `no_repeat_ngram_size` | 3 | `SamplingConfig.no_repeat_ngram_size = 3` |
| `early_stopping` | False (from `generation_config.json`) | `SamplingConfig.early_stopping = 0` |

After generating 3 bos tokens, `no_repeat_ngram_size=3` prevents further bos, forcing the model to produce content tokens.

## Files

| File | Description |
|------|-------------|
| `convert_checkpoint.py` | Convert HF Florence-2 weights to TRT-LLM checkpoint format |
| `build.sh` | One-shot build: convert checkpoint + build encoder/decoder + vision TRT engines |
| `build_vision.py` | Build TRT engine for DaViT vision encoder |
| `run.py` | End-to-end inference (DaViT in PyTorch or TRT + BART in TRT-LLM) |

## Troubleshooting

### Python path: `ModuleNotFoundError: No module named 'helper'`

`convert_checkpoint.py` imports the shared `helper` module from the enc_dec examples. Add it to `PYTHONPATH`:

```bash
export PYTHONPATH=/path/to/TensorRT-LLM/examples/models/core/enc_dec:$PYTHONPATH
```

### Python path: `ModuleNotFoundError: No module named 'tensorrt_llm.bindings'`

If you run scripts with CWD set to the TensorRT-LLM source tree root, Python may import the source `tensorrt_llm/` package (which lacks compiled C++ bindings) instead of the pip-installed one. Run scripts from outside the source tree:

```bash
cd /tmp
python /path/to/TensorRT-LLM/examples/models/contrib/florence2/run.py ...
```

### scipy / numpy ABI incompatibility

**Symptom**: `ValueError: All ufuncs must have type numpy.ufunc` or infinite recursion in `numpy.core._dtype` when importing `tensorrt_llm` or `transformers`.

**Root cause**: `scipy >= 1.14` is built against the numpy 2.x ABI. If your environment pins `numpy < 2` (as `tensorrt-llm` requires), the ABI mismatch causes crashes at import time. The problem is amplified when a user-level (`~/.local/lib/python3.12/site-packages/`) scipy shadows the system-level one.

**Fix**:

1. Remove any user-level scipy that may shadow the system installation:

   ```bash
   rm -rf ~/.local/lib/python3.12/site-packages/scipy*
   ```

2. Install a scipy version compatible with numpy 1.x at the system level:

   ```bash
   pip install 'scipy==1.13.1'
   ```

3. Verify both packages load from the system path without errors:

   ```bash
   python -c "import numpy; print(numpy.__version__, numpy.__file__)"
   # Expected: 1.26.4 /usr/local/.../numpy/__init__.py

   python -c "import scipy; print(scipy.__version__, scipy.__file__)"
   # Expected: 1.13.1 /usr/local/.../scipy/__init__.py
   ```

> **Do not** upgrade numpy to 2.x — `tensorrt_llm` C++ bindings are compiled against the numpy 1.x ABI and will fail to import with `ModuleNotFoundError: No module named 'tensorrt_llm.bindings'`.

### Missing `timm` package

**Symptom**: `ImportError: This modeling file requires the following packages that were not found in your environment: timm`

Florence-2's HF model code (`modeling_florence2.py`) uses `timm` for the DaViT vision backbone:

```bash
pip install timm
```

## Performance

Tested on a single GPU with Florence-2-large-ft, `<CAPTION>` task, `num_beams=3`, `max_new_tokens=128`:

| Runtime | Latency | Notes |
|---------|---------|-------|
| HF (FP16, eager attention) | ~820ms | Full PyTorch |
| TRT-LLM (FP16) | ~240ms | ~3.4x speedup |

Output match rate vs HF: 92-100% (minor differences from attention kernel implementations).
