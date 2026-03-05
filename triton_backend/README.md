# Florence2 Triton Ensemble Deployment Guide

This guide covers building TensorRT engines for Florence-2-large-ft and
deploying the Triton Inference Server ensemble via Kubernetes + Helm.

## Architecture

```
Client (gRPC/HTTP)
  │
  └─► ensemble
        ├─► preprocessing   (Python backend, CPU)
        │     Tokenize task prompt, build virtual token IDs, load image
        ├─► vision           (TensorRT plan backend, GPU)
        │     DaViT TRT engine → image_features as prompt_embedding_table
        ├─► tensorrt_llm     (C++ tensorrtllm backend, GPU)
        │     BART encoder/decoder TRT-LLM engines, beam search
        └─► postprocessing   (Python backend, CPU)
              Decode tokens, Florence2 structured output parsing
```

## Prerequisites

| Component | Version |
|-----------|---------|
| TensorRT-LLM | 1.3.x |
| Triton Inference Server | 25.12+ (`nvcr.io/nvidia/tritonserver:25.12-trtllm-python-py3`) |
| GPU | NVIDIA Ampere or newer (A10, A100, H100, L40S, Blackwell, etc.) |
| CUDA | 12.x+ |
| Kubernetes | 1.27+ |
| Helm | 3.x |
| [NVIDIA GPU Operator](https://docs.nvidia.com/datacenter/cloud-native/gpu-operator/latest/index.html) | Installed on k8s cluster |

## Step 1: Build TensorRT Engines

Run these on a machine with the target GPU. All commands use the
TRT-LLM container image.

```bash
# Launch build container
docker run --rm -it --gpus all \
  -v /path/to/Florence-2-large-ft:/model \
  -v /path/to/output:/engines \
  -v /path/to/TensorRT-LLM:/workspace/TensorRT-LLM \
  nvcr.io/nvidia/tritonserver:25.12-trtllm-python-py3 bash
```

### 1a. Convert HF weights to TRT-LLM checkpoint

```bash
python /workspace/TensorRT-LLM/examples/models/core/enc_dec/convert_florence2.py \
    --model_dir /model \
    --output_dir /engines/ckpt \
    --dtype float16
```

### 1b. Build BART encoder + decoder engines

```bash
MAX_BATCH_SIZE=64 \
MAX_PROMPT_EMBEDDING_TABLE_SIZE=36928 \
  bash /workspace/TensorRT-LLM/examples/models/core/enc_dec/build_florence2.sh \
  /engines/ckpt /engines
```

Key build parameters:
- `MAX_BATCH_SIZE=64` — max concurrent requests
- `MAX_PROMPT_EMBEDDING_TABLE_SIZE=36928` — must be `MAX_BATCH_SIZE * 577` (577 = DaViT image tokens)
- `MAX_BEAM_WIDTH=3` — Florence2 uses beam search with 3 beams (hardcoded in build script)

### 1c. Build DaViT vision engine

```bash
python /workspace/TensorRT-LLM/examples/models/core/enc_dec/build_florence2_vision.py \
    --model_dir /model \
    --output_dir /engines/vision \
    --max_batch_size 64 \
    --dtype float16
```

### 1d. Verify engines (optional)

```bash
python /workspace/TensorRT-LLM/examples/models/core/enc_dec/run_florence2.py \
    --model_dir /model \
    --engine_dir /engines \
    --vision_engine_dir /engines/vision \
    --task "<CAPTION>" \
    --image /path/to/test.jpg \
    --compare_hf
```

Expected output for the standard car.jpg test image:
```
TRT-LLM decoded text: ['A green car parked in front of a yellow building.']
```

### Engine directory structure

After building, you should have:

```
/engines/
├── encoder/
│   ├── config.json
│   └── rank0.engine
├── decoder/
│   ├── config.json
│   └── rank0.engine
└── vision/
    ├── config.json
    └── model.engine
```

### Typical two-step workflow (recommended for iteration)

When you frequently tweak Triton configs, use this split workflow:
build engines once, then rerun model-repo preparation only.

```bash
# Step A: Build engines (one-time or when engine settings change)
bash build.sh \
    --model_dir /path/to/Florence-2-large-ft \
    --engine_dir /tmp/florence2_engines \
    --max_batch_size 64 \
    --max_seq_len 256

# Step B: Prepare Triton model repository (rerun after config/template changes)
bash triton_backend/prepare_model_repo.sh
```

## Step 2: Prepare Model Repository

Use `prepare_model_repo.sh` when engines are already built and you only want
to generate the final Triton model repository.
By default, the script writes to `/opt/tritonserver/model_repo`.
You can override it with `--output_dir`.
In the sections below, `<model_repo>` means the same repo root path.
For `tensorrt_llm/config.pbtxt`, it writes relative engine/tokenizer paths:
`decoder_engine_dir=1/decoder`, `encoder_engine_dir=1/encoder`, `tokenizer_dir=1/tokenizer`.
For `preprocessing/postprocessing`, it writes
`tokenizer_dir=../tensorrt_llm/1/tokenizer`.
No symlink/copy for external assets is performed.

```bash
bash triton_backend/prepare_model_repo.sh
```

Common options:

```bash
bash triton_backend/prepare_model_repo.sh \
    --output_dir /opt/tritonserver/model_repo \
    --max_batch_size 64 \
    --max_input_len 608 \
    --max_seq_len 256 \
    --max_beam_width 3 \
    --kv_cache_free_gpu_mem_fraction 0.5 \
    --cross_kv_cache_fraction 0.5 \
    --batch_scheduler_policy guaranteed_no_evict \
    --enable_chunked_context false
```

### Filled model repository layout

```
<model_repo>/
├── ensemble/
│   └── config.pbtxt
├── preprocessing/
│   ├── config.pbtxt
│   └── 1/
│       └── model.py
├── vision/
│   ├── config.pbtxt
│   └── 1/
│       └── model.engine   # must be mounted here at runtime
├── tensorrt_llm/
│   ├── config.pbtxt
│   └── 1/
│       ├── encoder         # mount external encoder engine dir
│       ├── decoder         # mount external decoder engine dir
│       └── tokenizer       # mount external tokenizer dir
└── postprocessing/
    ├── config.pbtxt
    └── 1/
        └── model.py
```

### Runtime mount requirements

Container target paths are fixed by current `prepare_model_repo.sh` output and
config values. Inference will fail if these paths are not mounted exactly.

| Asset | Must be mounted to container path | Type | Required content |
|------|-----------------------------------|------|------------------|
| Tokenizer directory | `<model_repo>/tensorrt_llm/1/tokenizer` | Directory | `config.json`, `tokenizer.json`, `tokenizer_config.json`, `vocab.json`, Florence2 custom python files |
| Encoder engine | `<model_repo>/tensorrt_llm/1/encoder` | Directory | TRT-LLM encoder engine dir (`config.json`, `rank0.engine`, etc.) |
| Decoder engine | `<model_repo>/tensorrt_llm/1/decoder` | Directory | TRT-LLM decoder engine dir (`config.json`, `rank0.engine`, etc.) |
| Vision engine | `<model_repo>/vision/1/model.engine` | Single file | DaViT TensorRT plan file |

`platform: "tensorrt_plan"` for `vision` strictly requires:
`<model_repo>/vision/1/model.engine`.

### Minimal required mounts (production)

If you only keep the essentials, these 4 mounts are mandatory:

1. tokenizer dir -> `<model_repo>/tensorrt_llm/1/tokenizer` (directory)
2. encoder dir -> `<model_repo>/tensorrt_llm/1/encoder` (directory)
3. decoder dir -> `<model_repo>/tensorrt_llm/1/decoder` (directory)
4. vision engine -> `<model_repo>/vision/1/model.engine` (single file)

If you use a custom `--output_dir`, replace `/opt/tritonserver/model_repo`
in the examples below with that same path.

Example (`docker run`):

```bash
docker run --rm --gpus all \
  -v /tmp/florence2_model_repo:/opt/tritonserver/model_repo \
  -v /path/to/Florence-2-large-ft:/opt/tritonserver/model_repo/tensorrt_llm/1/tokenizer:ro \
  -v /path/to/florence2_engines/encoder:/opt/tritonserver/model_repo/tensorrt_llm/1/encoder:ro \
  -v /path/to/florence2_engines/decoder:/opt/tritonserver/model_repo/tensorrt_llm/1/decoder:ro \
  -v /path/to/florence2_engines/vision/model.engine:/opt/tritonserver/model_repo/vision/1/model.engine:ro \
  nvcr.io/nvidia/tritonserver:25.12-trtllm-python-py3 \
  tritonserver \
    --model-repository=/opt/tritonserver/model_repo \
    --disable-auto-complete-config \
    --backend-config=python,shm-region-prefix-name=florence2_
```

Quick check inside container:

```bash
MODEL_REPO=/opt/tritonserver/model_repo
ls -l ${MODEL_REPO}/tensorrt_llm/1/tokenizer
ls -l ${MODEL_REPO}/tensorrt_llm/1/encoder
ls -l ${MODEL_REPO}/tensorrt_llm/1/decoder
ls -l ${MODEL_REPO}/vision/1/model.engine
```

## Step 3: Build Serving Container Image

Create a Docker image that bundles runtime tokenizer files, engines, and
filled model repo.

```dockerfile
# Dockerfile.florence2
ARG BASE_IMAGE=nvcr.io/nvidia/tritonserver:25.12-trtllm-python-py3

FROM ${BASE_IMAGE}

# Copy filled model repository
COPY triton_model_repo /opt/tritonserver/model_repo

# Copy tokenizer files into configured runtime path
RUN mkdir -p /opt/tritonserver/model_repo/tensorrt_llm/1/tokenizer
COPY Florence-2-large-ft/config.json /opt/tritonserver/model_repo/tensorrt_llm/1/tokenizer/
COPY Florence-2-large-ft/generation_config.json /opt/tritonserver/model_repo/tensorrt_llm/1/tokenizer/
COPY Florence-2-large-ft/preprocessor_config.json /opt/tritonserver/model_repo/tensorrt_llm/1/tokenizer/
COPY Florence-2-large-ft/tokenizer.json /opt/tritonserver/model_repo/tensorrt_llm/1/tokenizer/
COPY Florence-2-large-ft/tokenizer_config.json /opt/tritonserver/model_repo/tensorrt_llm/1/tokenizer/
COPY Florence-2-large-ft/vocab.json /opt/tritonserver/model_repo/tensorrt_llm/1/tokenizer/
COPY Florence-2-large-ft/processing_florence2.py /opt/tritonserver/model_repo/tensorrt_llm/1/tokenizer/
COPY Florence-2-large-ft/configuration_florence2.py /opt/tritonserver/model_repo/tensorrt_llm/1/tokenizer/
COPY Florence-2-large-ft/modeling_florence2.py /opt/tritonserver/model_repo/tensorrt_llm/1/tokenizer/

# Copy pre-built TRT engines into configured runtime paths
COPY engines/encoder /opt/tritonserver/model_repo/tensorrt_llm/1/encoder
COPY engines/decoder /opt/tritonserver/model_repo/tensorrt_llm/1/decoder
COPY engines/vision/model.engine /opt/tritonserver/model_repo/vision/1/model.engine

# Install Florence2 dependencies (AutoProcessor uses transformers)
RUN pip install --no-cache-dir timm einops

ENTRYPOINT ["tritonserver", \
    "--model-repository=/opt/tritonserver/model_repo", \
    "--disable-auto-complete-config", \
    "--backend-config=python,shm-region-prefix-name=florence2_"]
```

Build and push:

```bash
docker build -t <registry>/florence2-triton:latest -f Dockerfile.florence2 .
docker push <registry>/florence2-triton:latest
```

## Step 4: Deploy with Helm on Kubernetes

Use the [NVIDIA Triton Inference Server Helm chart](https://github.com/triton-inference-server/server/tree/main/deploy/k8s-onprem).

### 4a. Add Helm repository

```bash
# Option A: use NVIDIA's published chart
helm fetch https://github.com/triton-inference-server/server/releases/download/v2.53.0/tritonserver-2.53.0.tgz
tar xzf tritonserver-2.53.0.tgz

# Option B: clone and use from source
git clone https://github.com/triton-inference-server/server.git
cd server/deploy/k8s-onprem
```

### 4b. Create values file

```yaml
# values-florence2.yaml

image:
  imageName: <registry>/florence2-triton:latest
  pullPolicy: IfNotPresent

# GPU resources
resources:
  limits:
    nvidia.com/gpu: 1
  requests:
    nvidia.com/gpu: 1

# Shared memory for Python backend IPC
shmSize: 4Gi

# Tritonserver command-line args (override entrypoint if needed)
args:
  - --model-repository=/opt/tritonserver/model_repo
  - --disable-auto-complete-config
  - --backend-config=python,shm-region-prefix-name=florence2_
  - --log-verbose=0

# Runtime mounts (example)
volumeMounts:
  - name: florence2-model-repo
    mountPath: /opt/tritonserver/model_repo
  - name: florence2-tokenizer
    mountPath: /opt/tritonserver/model_repo/tensorrt_llm/1/tokenizer
    readOnly: true
  - name: florence2-encoder
    mountPath: /opt/tritonserver/model_repo/tensorrt_llm/1/encoder
    readOnly: true
  - name: florence2-decoder
    mountPath: /opt/tritonserver/model_repo/tensorrt_llm/1/decoder
    readOnly: true
  - name: florence2-vision
    mountPath: /opt/tritonserver/model_repo/vision/1/model.engine
    subPath: model.engine
    readOnly: true

volumes:
  - name: florence2-model-repo
    persistentVolumeClaim:
      claimName: pvc-florence2-model-repo
  - name: florence2-tokenizer
    persistentVolumeClaim:
      claimName: pvc-florence2-tokenizer
  - name: florence2-encoder
    persistentVolumeClaim:
      claimName: pvc-florence2-encoder
  - name: florence2-decoder
    persistentVolumeClaim:
      claimName: pvc-florence2-decoder
  - name: florence2-vision
    persistentVolumeClaim:
      claimName: pvc-florence2-vision

# Service ports
service:
  type: ClusterIP
  ports:
    - name: http
      port: 8000
      targetPort: 8000
    - name: grpc
      port: 8001
      targetPort: 8001
    - name: metrics
      port: 8002
      targetPort: 8002

# Health checks
readinessProbe:
  httpGet:
    path: /v2/health/ready
    port: 8000
  initialDelaySeconds: 30
  periodSeconds: 10

livenessProbe:
  httpGet:
    path: /v2/health/live
    port: 8000
  initialDelaySeconds: 30
  periodSeconds: 15

# Optional: node selector for specific GPU type
nodeSelector:
  nvidia.com/gpu.product: "NVIDIA-A100-SXM4-80GB"

# Optional: tolerations for GPU nodes
tolerations:
  - key: nvidia.com/gpu
    operator: Exists
    effect: NoSchedule
```

### 4c. Deploy

```bash
helm install florence2 ./tritonserver \
    -f values-florence2.yaml \
    --namespace florence2 \
    --create-namespace
```

### 4d. Verify deployment

```bash
# Check pod status
kubectl get pods -n florence2

# Wait for ready
kubectl wait --for=condition=ready pod -l app=florence2 -n florence2 --timeout=300s

# Check server logs
kubectl logs -n florence2 -l app=florence2 --tail=50

# Port-forward for local testing
kubectl port-forward -n florence2 svc/florence2-tritonserver 8001:8001
```

## Step 5: Send Inference Requests

### Python gRPC client

```python
import numpy as np
import tritonclient.grpc as grpcclient
from tritonclient.utils import np_to_triton_dtype

def florence2_infer(image_path: str, task: str = "<CAPTION>",
                    url: str = "localhost:8001"):
    client = grpcclient.InferenceServerClient(url=url)

    # Inputs
    text_input = np.array([[task]], dtype=object)
    with open(image_path, "rb") as f:
        image_bytes = np.frombuffer(f.read(), dtype=np.uint8)

    inputs = []
    for name, data in [("text_input", text_input),
                        ("image_bytes_input", image_bytes)]:
        inp = grpcclient.InferInput(name, list(data.shape),
                                    np_to_triton_dtype(data.dtype))
        inp.set_data_from_numpy(data)
        inputs.append(inp)

    outputs = [grpcclient.InferRequestedOutput("text_output")]
    result = client.infer("ensemble", inputs=inputs, outputs=outputs)

    text = result.as_numpy("text_output")[0]
    return text.decode("utf-8") if isinstance(text, bytes) else text


# Example usage
print(florence2_infer("car.jpg", "<CAPTION>"))
# => A green car parked in front of a yellow building.
```

### curl (HTTP)

```bash
# Florence2 ensemble expects binary image + text, which requires
# the Triton HTTP/REST API with binary tensor extension.
# gRPC client (above) is recommended for production use.

# Health check
curl -s http://localhost:8000/v2/health/ready
# => 200 OK

# Model metadata
curl -s http://localhost:8000/v2/models/ensemble | python3 -m json.tool
```

### Supported Florence2 tasks

| Task prompt | Output |
|-------------|--------|
| `<CAPTION>` | Short caption |
| `<DETAILED_CAPTION>` | Detailed caption |
| `<MORE_DETAILED_CAPTION>` | Very detailed caption |
| `<OD>` | Object detection (bboxes + labels) |
| `<DENSE_REGION_CAPTION>` | Dense region captions |
| `<REGION_PROPOSAL>` | Region proposals |
| `<OCR>` | OCR text |
| `<OCR_WITH_REGION>` | OCR with bounding boxes |
| `<CAPTION_TO_PHRASE_GROUNDING> text` | Phrase grounding |
| `<REFERRING_EXPRESSION_SEGMENTATION> text` | Referring segmentation |
| `<OPEN_VOCABULARY_DETECTION> text` | Open-vocabulary detection |

## Configuration Reference

### Key tensorrt_llm parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `triton_max_batch_size` | 64 | Max concurrent requests in TRT-LLM backend |
| `batching_strategy` | `inflight_fused_batching` | In-flight batching for optimal throughput |
| `kv_cache_free_gpu_mem_fraction` | 0.5 | Fraction of free GPU memory for KV cache |
| `cross_kv_cache_fraction` | 0.5 | Split between self/cross-attention KV cache |
| `max_beam_width` | 3 | Beam search width (hardcoded for Florence2) |
| `batch_scheduler_policy` | `guaranteed_no_evict` | Never evict running requests |

### Preprocessing defaults

| Value | Setting | Notes |
|-------|---------|-------|
| `beam_width` | 3 | Prevents degenerate BART beam search |
| `no_repeat_ngram_size` | 3 | Prevents repetition in beam search |
| `len_penalty` | 1.0 | Beam-search length normalization (default in `prepare_model_repo.sh`) |
| `n_img_tokens` | 577 | Fixed for Florence2 DaViT at 768x768 |
| `vocab_size` | 51289 | Florence-2-large-ft |

## Scaling and Performance

### Horizontal scaling

Deploy multiple replicas behind a Kubernetes Service. Each replica
needs its own GPU.

```yaml
# In values-florence2.yaml
replicaCount: 4
```

### GPU memory estimation (Florence-2-large-ft, float16)

| Component | Approximate GPU memory |
|-----------|----------------------|
| DaViT vision TRT engine | ~700 MB |
| BART encoder TRT engine | ~400 MB |
| BART decoder TRT engine + KV cache | ~700 MB (varies with batch/seq) |
| CUDA/framework overhead | ~2 GB |
| **Total (batch_size=1)** | **~4 GB** |

Florence-2-large-ft is a relatively small model (~0.77B parameters).
A single A10 (24 GB) or L4 (24 GB) GPU can serve batch_size=64
comfortably.

## Troubleshooting

| Symptom | Cause | Fix |
|---------|-------|-----|
| `UNAVAILABLE: model 'tensorrt_llm' is not ready` | Missing engine files or wrong paths | Check `decoder_engine_dir` / `encoder_engine_dir` in config |
| `UNAVAILABLE: model 'vision' is not ready` | Vision engine not found | Check `/opt/tritonserver/model_repo/vision/1/model.engine` is mounted and readable |
| `Python backend: ModuleNotFoundError` | Missing pip packages | Install `timm`, `einops`, `transformers` in the container |
| `CUDA out of memory` | KV cache too large | Reduce `kv_cache_free_gpu_mem_fraction` (e.g. 0.3) |
| Degenerate repetitive output | Missing `no_repeat_ngram_size` | Verify `no_repeat_ngram_size` input is declared in `tensorrt_llm/config.pbtxt` |
| `shm` errors | Insufficient shared memory | Increase `shmSize` in Helm values (default: 4Gi) |

## Files

| File | Purpose |
|------|---------|
| `ensemble/config.pbtxt` | 4-step ensemble wiring |
| `preprocessing/1/model.py` | Image loading, tokenization, virtual token ID construction |
| `vision/1/model.engine` | DaViT TensorRT plan file (mounted at runtime) |
| `tensorrt_llm/config.pbtxt` | C++ tensorrtllm backend configuration |
| `postprocessing/1/model.py` | Token decoding + Florence2 structured output parsing |
| `test_pipeline.py` | Standalone integration test (no Triton server needed) |
| `test_triton_server.py` | End-to-end test that launches `tritonserver` |
