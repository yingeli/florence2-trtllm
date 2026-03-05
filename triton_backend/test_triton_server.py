#!/usr/bin/env python3
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
"""End-to-end Triton Inference Server test for the Florence2 ensemble.

This script:
  1. Copies the model repository to a working directory
  2. Fills config.pbtxt template variables (engine paths, tokenizer, etc.)
  3. Launches tritonserver in the background
  4. Waits for the server to become ready
  5. Sends inference requests via gRPC (image bytes + task prompt)
  6. Verifies the output against the expected reference
  7. Shuts down the server and cleans up

Prerequisites:
  - tritonserver binary (typically /opt/tritonserver/bin/tritonserver)
  - TensorRT-LLM backend (libtriton_tensorrtllm.so)
  - tritonclient[grpc] Python package
  - Pre-built Florence2 TRT engines (encoder, decoder, vision)

Usage:
    python test_triton_server.py \
        --model_dir /path/to/Florence-2-large-ft \
        --engine_dir /tmp/florence2_engine \
        --vision_engine_dir /tmp/florence2_engine/vision \
        --image /path/to/car.jpg \
        --task "<CAPTION>"
"""
import argparse
import os
import re
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path
from string import Template

import numpy as np

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
EXPECTED_OUTPUT = "A green car parked in front of a yellow building."
GRPC_PORT = 18001
HTTP_PORT = 18000
METRICS_PORT = 18002
STARTUP_TIMEOUT = 120  # seconds
REQUEST_TIMEOUT = 60  # seconds
DEFAULT_MAX_BATCH_SIZE = 64
DEFAULT_MAX_INPUT_LEN = 608
DEFAULT_MAX_SEQ_LEN = 256
DEFAULT_MAX_BEAM_WIDTH = 3
DEFAULT_LENGTH_PENALTY = 1.0


# ---------------------------------------------------------------------------
# 1. Prepare model repository
# ---------------------------------------------------------------------------
def prepare_model_repo(src_dir: Path, work_dir: Path, args) -> Path:
    """Copy the template model repo and fill in config variables."""
    repo = work_dir / "model_repo"
    if repo.exists():
        shutil.rmtree(repo)

    template_dir = src_dir / "template"
    if not template_dir.is_dir():
        raise FileNotFoundError(f"Template directory not found: {template_dir}")

    # Copy only template model repo content.
    shutil.copytree(template_dir, repo)

    tokenizer_files = [
        "config.json",
        "generation_config.json",
        "preprocessor_config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "vocab.json",
        "processing_florence2.py",
        "configuration_florence2.py",
        "modeling_florence2.py",
    ]

    # Match prepare_model_repo.sh layout:
    #   tensorrt_llm/1/{encoder,decoder,tokenizer}
    # with relative paths in model configs.
    trtllm_v1 = repo / "tensorrt_llm" / "1"
    trtllm_v1.mkdir(parents=True, exist_ok=True)

    # Copy tokenizer files under tensorrt_llm/1/tokenizer.
    tokenizer_dst = trtllm_v1 / "tokenizer"
    tokenizer_dst.mkdir(parents=True, exist_ok=True)
    for filename in tokenizer_files:
        src = Path(args.model_dir) / filename
        if src.is_file():
            shutil.copy2(src, tokenizer_dst / filename)
        else:
            print(f"  [WARN] tokenizer file not found, skipping: {src}")

    # Copy encoder/decoder engine directories under tensorrt_llm/1/.
    engine_root = Path(args.engine_dir).resolve()
    for name in ("encoder", "decoder"):
        src = engine_root / name
        if not src.is_dir():
            raise FileNotFoundError(f"{name} engine dir not found: {src}")
        dst = trtllm_v1 / name
        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(src, dst)

    # Vision backend is tensorrt_plan and expects vision/1/model.engine.
    vision_engine = Path(args.vision_engine_dir) / "model.engine"
    if not vision_engine.is_file():
        raise FileNotFoundError(
            f"Vision engine not found: {vision_engine}")
    vision_dst = repo / "vision" / "1" / "model.engine"
    vision_dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(vision_engine, vision_dst)
    # Python wrapper is unused for tensorrt_plan.
    (repo / "vision" / "1" / "model.py").unlink(missing_ok=True)

    tokenizer_dir_for_prepost = "../tensorrt_llm/1/tokenizer"
    tokenizer_dir_for_trtllm = "1/tokenizer"
    decoder_engine_dir_for_trtllm = "1/decoder"
    encoder_engine_dir_for_trtllm = "1/encoder"

    # preprocessing
    _fill(repo / "preprocessing" / "config.pbtxt", {
        "tokenizer_dir": tokenizer_dir_for_prepost,
        "max_input_len": str(DEFAULT_MAX_INPUT_LEN),
        "max_output_len": str(DEFAULT_MAX_SEQ_LEN),
        "length_penalty": str(DEFAULT_LENGTH_PENALTY),
    })

    # vision
    _fill(repo / "vision" / "config.pbtxt", {
        "vision_triton_max_batch_size": str(DEFAULT_MAX_BATCH_SIZE),
    })

    # postprocessing
    _fill(repo / "postprocessing" / "config.pbtxt", {
        "tokenizer_dir": tokenizer_dir_for_prepost,
        "apply_florence2_post_process": "true",
    })

    # tensorrt_llm (C++ backend)
    _fill(repo / "tensorrt_llm" / "config.pbtxt", {
        "triton_max_batch_size": str(DEFAULT_MAX_BATCH_SIZE),
        "max_queue_delay_microseconds": "0",
        "max_beam_width": str(DEFAULT_MAX_BEAM_WIDTH),
        "decoder_engine_dir": decoder_engine_dir_for_trtllm,
        "encoder_engine_dir": encoder_engine_dir_for_trtllm,
        "tokenizer_dir": tokenizer_dir_for_trtllm,
        "batch_scheduler_policy": "guaranteed_no_evict",
        "kv_cache_free_gpu_mem_fraction": "0.5",
        "cross_kv_cache_fraction": "0.5",
        "enable_chunked_context": "false",
    })

    # Fail fast if any template placeholder remains unresolved.
    unresolved = []
    for pbtxt in repo.rglob("*.pbtxt"):
        text = pbtxt.read_text(encoding="utf-8")
        if re.search(r"\$\{[a-z_]+\}", text):
            unresolved.append(str(pbtxt))
    if unresolved:
        raise RuntimeError(
            "Unresolved template variables in: " + ", ".join(unresolved))

    print(f"  Model repo prepared at {repo}")
    return repo


def _fill(config_path: Path, substitutions: dict):
    """Apply template substitutions to a config.pbtxt file in-place."""
    with open(config_path) as f:
        content = f.read()
    tmpl = Template(content)
    # safe_substitute leaves unknown ${vars} as-is
    result = tmpl.safe_substitute(substitutions)
    with open(config_path, "w") as f:
        f.write(result)


# ---------------------------------------------------------------------------
# 2. Launch tritonserver
# ---------------------------------------------------------------------------
def find_tritonserver() -> str:
    """Find the tritonserver binary."""
    candidates = [
        os.environ.get("TRITONSERVER", ""),
        "/opt/tritonserver/bin/tritonserver",
        shutil.which("tritonserver") or "",
    ]
    for c in candidates:
        if c and os.path.isfile(c) and os.access(c, os.X_OK):
            return c
    return ""


def launch_server(model_repo: Path, tritonserver: str,
                  grpc_port: int = GRPC_PORT,
                  http_port: int = HTTP_PORT,
                  metrics_port: int = METRICS_PORT) -> subprocess.Popen:
    """Start tritonserver and return the process handle."""
    cmd = [
        tritonserver,
        f"--model-repository={model_repo}",
        f"--grpc-port={grpc_port}",
        f"--http-port={http_port}",
        f"--metrics-port={metrics_port}",
        "--disable-auto-complete-config",
        "--backend-config=python,shm-region-prefix-name=florence2_test_",
    ]
    print(f"  Starting tritonserver:")
    print(f"    {' '.join(cmd)}")
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        preexec_fn=os.setsid,  # new process group for clean kill
    )
    return proc


def wait_for_ready(grpc_url: str, timeout: int, proc: subprocess.Popen):
    """Poll the server until it reports READY or timeout is reached."""
    import tritonclient.grpc as grpcclient

    print(f"  Waiting for server to be ready (timeout={timeout}s) ...")
    start = time.time()
    while time.time() - start < timeout:
        # Check if process died
        if proc.poll() is not None:
            stdout = proc.stdout.read().decode("utf-8", errors="replace")
            print(f"\n  [ERROR] tritonserver exited with code {proc.returncode}")
            print("  --- server output (last 3000 chars) ---")
            print(stdout[-3000:])
            raise RuntimeError("tritonserver exited unexpectedly")
        try:
            client = grpcclient.InferenceServerClient(url=grpc_url)
            if client.is_server_ready():
                elapsed = time.time() - start
                print(f"  Server ready in {elapsed:.1f}s")
                return
        except Exception:
            pass
        time.sleep(2)
    # Timeout — dump logs
    stdout = proc.stdout.read(8192).decode("utf-8", errors="replace")
    print(f"\n  [ERROR] Server did not become ready within {timeout}s")
    print("  --- server output (partial) ---")
    print(stdout[-3000:])
    raise TimeoutError("tritonserver startup timeout")


# ---------------------------------------------------------------------------
# 3. Send inference request
# ---------------------------------------------------------------------------
def send_request(grpc_url: str, image_path: str, task: str) -> str:
    """Send a single request to the Florence2 ensemble and return text."""
    import tritonclient.grpc as grpcclient
    from tritonclient.utils import np_to_triton_dtype

    client = grpcclient.InferenceServerClient(url=grpc_url)

    # Prepare inputs
    text_input = np.array([[task]], dtype=object)
    inputs = []
    inputs.append(_make_input("text_input", text_input))

    # Read image as bytes
    with open(image_path, "rb") as f:
        image_bytes = f.read()
    image_bytes_np = np.frombuffer(image_bytes, dtype=np.uint8)
    inputs.append(_make_input("image_bytes_input", image_bytes_np))

    outputs = [grpcclient.InferRequestedOutput("text_output")]

    print(f"  Sending request: task={task}, image={image_path}")
    result = client.infer(
        model_name="ensemble",
        inputs=inputs,
        outputs=outputs,
        client_timeout=REQUEST_TIMEOUT,
    )

    text_output = result.as_numpy("text_output")
    # text_output is an array of bytes
    decoded = []
    for item in text_output.flat:
        if isinstance(item, bytes):
            decoded.append(item.decode("utf-8"))
        else:
            decoded.append(str(item))
    return decoded[0] if decoded else ""


def _make_input(name, data):
    import tritonclient.grpc as grpcclient
    from tritonclient.utils import np_to_triton_dtype

    inp = grpcclient.InferInput(name, list(data.shape),
                                np_to_triton_dtype(data.dtype))
    inp.set_data_from_numpy(data)
    return inp


# ---------------------------------------------------------------------------
# 4. Cleanup
# ---------------------------------------------------------------------------
def kill_server(proc: subprocess.Popen):
    """Kill the tritonserver process group."""
    if proc and proc.poll() is None:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
            proc.wait(timeout=10)
        except Exception:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            proc.wait(timeout=5)
        print("  Server stopped.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="End-to-end Triton Server test for Florence2 ensemble")
    parser.add_argument("--model_dir", required=True,
                        help="Path to Florence-2-large-ft HF checkpoint")
    parser.add_argument("--engine_dir", required=True,
                        help="Path to TRT-LLM engine dir (encoder/ decoder/)")
    parser.add_argument("--vision_engine_dir", required=True,
                        help="Path to DaViT vision TRT engine dir")
    parser.add_argument("--image", required=True,
                        help="Path to test image")
    parser.add_argument("--task", default="<CAPTION>",
                        help="Florence2 task prompt (default: <CAPTION>)")
    parser.add_argument("--expected", default=EXPECTED_OUTPUT,
                        help="Expected output string for verification")
    parser.add_argument("--work_dir", default="/tmp/florence2_triton_test",
                        help="Working directory for model repo copy")
    parser.add_argument("--tritonserver", default="",
                        help="Path to tritonserver binary")
    parser.add_argument("--grpc_port", type=int, default=GRPC_PORT,
                        help="gRPC port")
    parser.add_argument("--keep_repo", action="store_true",
                        help="Keep the working model repo after test")
    args = parser.parse_args()

    grpc_port = args.grpc_port
    grpc_url = f"localhost:{grpc_port}"

    print("=" * 60)
    print("Florence2 Triton Server End-to-End Test")
    print("=" * 60)

    # Find tritonserver
    tritonserver = args.tritonserver or find_tritonserver()
    if not tritonserver:
        print("\n[ERROR] tritonserver binary not found.")
        print("  Set --tritonserver or install Triton Inference Server.")
        print("  In an NGC container: /opt/tritonserver/bin/tritonserver")
        sys.exit(1)
    print(f"  tritonserver: {tritonserver}")

    # Validate paths
    for name, path in [("model_dir", args.model_dir),
                       ("engine_dir", args.engine_dir),
                       ("vision_engine_dir", args.vision_engine_dir),
                       ("image", args.image)]:
        if not os.path.exists(path):
            print(f"\n[ERROR] {name} does not exist: {path}")
            sys.exit(1)

    proc = None
    work_dir = Path(args.work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    try:
        # Step 1: Prepare model repo
        print("\n[Step 1] Preparing model repository")
        model_repo = prepare_model_repo(SCRIPT_DIR, work_dir, args)

        # Step 2: Launch server
        print("\n[Step 2] Launching tritonserver")
        proc = launch_server(model_repo, tritonserver, grpc_port=grpc_port)

        # Step 3: Wait for ready
        print("\n[Step 3] Waiting for server")
        wait_for_ready(grpc_url, STARTUP_TIMEOUT, proc)

        # Step 4: Send request
        print("\n[Step 4] Sending inference request")
        output = send_request(grpc_url, args.image, args.task)
        print(f"  Output: {output}")

        # Step 5: Verify
        print("\n" + "=" * 60)
        print("VERIFICATION")
        print("=" * 60)
        print(f"Expected: {args.expected}")
        print(f"Actual:   {output}")
        if output.strip() == args.expected.strip():
            print(f"Match:    PASS")
        else:
            print(f"Match:    FAIL")
            sys.exit(1)
        print("=" * 60)

    except KeyboardInterrupt:
        print("\n  Interrupted by user.")
        sys.exit(130)
    except Exception as e:
        print(f"\n[ERROR] {e}")
        sys.exit(1)
    finally:
        # Step 6: Cleanup
        print("\n[Cleanup]")
        kill_server(proc)
        if not args.keep_repo and work_dir.exists():
            shutil.rmtree(work_dir, ignore_errors=True)
            print(f"  Removed {work_dir}")


if __name__ == "__main__":
    main()
