# Copyright 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
#  * Neither the name of NVIDIA CORPORATION nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
# OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""Florence2 DaViT vision encoder Triton model.

Runs the DaViT TensorRT engine to convert pixel_values [B, 3, 768, 768]
into image_features [B, 577, 1024], which are output as the
prompt_embedding_table for the BART encoder.

Reference: examples/models/core/enc_dec/run_florence2.py (VisionTRTRunner)
Reference: triton_backend/all_models/multimodal/multimodal_encoders/1/model.py
"""

import json
import os

# Ensure HOME is writable before importing packages that use JIT caches
# (e.g., flashinfer). In Triton containers the triton-server user may not
# have write access to its HOME directory.
if not os.access(os.environ.get('HOME', '/nonexistent'), os.W_OK):
    os.environ['HOME'] = '/tmp'

import numpy as np
import torch
import triton_python_backend_utils as pb_utils
from torch.utils.dlpack import from_dlpack, to_dlpack

import tensorrt_llm
import tensorrt_llm.logger as logger
from tensorrt_llm._utils import trt_dtype_to_torch, torch_dtype_to_trt
from tensorrt_llm.runtime import Session, TensorInfo

logger.set_level('info')


class TritonPythonModel:

    def initialize(self, args):
        # Only initialize on rank 0 (non-LLM engines only run on GPU 0).
        self.rank = tensorrt_llm.mpi_rank()
        if self.rank != 0:
            return

        model_config = json.loads(args['model_config'])

        # Resolve engine directory: prefer local 'engine/' subdir next to
        # this model.py (works with Triton cloud storage where absolute
        # build-time paths are invalid), then fall back to the configured path.
        model_dir = os.path.dirname(os.path.abspath(__file__))
        local_engine_dir = os.path.join(model_dir, 'engine')
        if os.path.isdir(local_engine_dir):
            visual_model_path = local_engine_dir
        else:
            visual_model_path = model_config['parameters'][
                'visual_model_path']['string_value']

        # Load TRT engine.
        engine_path = os.path.join(visual_model_path, 'model.engine')
        with open(engine_path, 'rb') as f:
            engine_buffer = f.read()
        self.session = Session.from_serialized_engine(engine_buffer)

        # Read config for precision and max batch size.
        config_path = os.path.join(visual_model_path, 'config.json')
        with open(config_path, 'r') as f:
            visual_config = json.load(f)
        self.vision_dtype_str = visual_config['builder_config']['precision']
        self.vision_max_batch_size = visual_config['builder_config'][
            'max_batch_size']

        self.stream = torch.cuda.current_stream()

        # Determine output dtype from Triton config.
        self.vision_output_dtype = self._triton_string_to_torch(
            pb_utils.get_output_config_by_name(
                model_config, 'OUT_PROMPT_EMBEDDING_TABLE')['data_type'])

    def execute(self, requests):
        if self.rank != 0:
            return [
                pb_utils.InferenceResponse(output_tensors=[])
                for _ in requests
            ]

        responses = []
        for request in requests:
            # Receive pixel_values from preprocessing.
            pixel_values_tensor = pb_utils.get_input_tensor_by_name(
                request, 'PIXEL_VALUES')
            pixel_values = from_dlpack(pixel_values_tensor.to_dlpack())
            pixel_values = pixel_values.to(
                'cuda',
                dtype=self._str_dtype_to_torch(self.vision_dtype_str))

            batch_size = pixel_values.shape[0]

            # Sub-batch loop respecting vision_max_batch_size.
            all_embeddings = []
            for start_idx in range(0, batch_size,
                                   self.vision_max_batch_size):
                end_idx = min(start_idx + self.vision_max_batch_size,
                              batch_size)
                batch = pixel_values[start_idx:end_idx]

                input_info = [
                    TensorInfo('pixel_values',
                               torch_dtype_to_trt(batch.dtype),
                               batch.shape)
                ]
                output_info = self.session.infer_shapes(input_info)

                outputs = {}
                for t in output_info:
                    outputs[t.name] = torch.empty(
                        tuple(t.shape),
                        dtype=trt_dtype_to_torch(t.dtype),
                        device='cuda')

                ok = self.session.run(
                    {'pixel_values': batch}, outputs,
                    self.stream.cuda_stream)
                if not ok:
                    raise RuntimeError(
                        "Vision TRT engine execution failed")

                # DaViT output tensor is named "image_features"
                # (from build_florence2_vision.py).
                all_embeddings.append(outputs['image_features'])

            embeddings = torch.cat(all_embeddings, dim=0)  # [B, 577, 1024]

            # Synchronize to ensure TRT engine writes are complete
            # before the output tensors are consumed by the next
            # ensemble step.
            self.stream.synchronize()

            # Output as prompt_embedding_table.
            embeddings = embeddings.to(self.vision_output_dtype)
            prompt_embedding_table_tensor = pb_utils.Tensor.from_dlpack(
                'OUT_PROMPT_EMBEDDING_TABLE', to_dlpack(embeddings))

            # prompt_vocab_size = 577 (number of image tokens).
            prompt_vocab_size = np.array(
                [[embeddings.shape[1]]] * batch_size, dtype=np.int32)
            prompt_vocab_size_tensor = pb_utils.Tensor(
                'OUT_PROMPT_VOCAB_SIZE', prompt_vocab_size)

            inference_response = pb_utils.InferenceResponse(
                output_tensors=[
                    prompt_embedding_table_tensor,
                    prompt_vocab_size_tensor,
                ])
            responses.append(inference_response)
        return responses

    def finalize(self):
        logger.info('Cleaning up florence2_vision...')

    @staticmethod
    def _str_dtype_to_torch(dtype_str):
        _map = {
            'bfloat16': torch.bfloat16,
            'float16': torch.float16,
            'float32': torch.float32,
            'int32': torch.int32,
            'int64': torch.int64,
        }
        return _map[dtype_str]

    @staticmethod
    def _triton_string_to_torch(dtype):
        _map = {
            "TYPE_BOOL": torch.bool,
            "TYPE_UINT8": torch.uint8,
            "TYPE_INT8": torch.int8,
            "TYPE_INT16": torch.int16,
            "TYPE_INT32": torch.int32,
            "TYPE_INT64": torch.int64,
            "TYPE_FP16": torch.float16,
            "TYPE_FP32": torch.float32,
            "TYPE_FP64": torch.float64,
            "TYPE_BF16": torch.bfloat16,
        }
        return _map[dtype]
