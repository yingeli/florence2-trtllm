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
"""Florence2 Triton postprocessing model.

Decodes output token IDs from the BART decoder and optionally applies
Florence2's structured post-processing (for tasks like <OD>, <OCR>, etc.).

Reference: examples/models/core/enc_dec/run_florence2.py (_decode_and_post_process)
Reference: triton_backend/all_models/inflight_batcher_llm/postprocessing/1/model.py
"""

import json
import os
import re
import sys

# Ensure HOME is writable before importing packages that create caches there.
# In Triton containers the triton-server user may not have write access to its
# HOME directory, which breaks transformers trust_remote_code module caching.
if not os.access(os.environ.get('HOME', '/nonexistent'), os.W_OK):
    os.environ['HOME'] = '/tmp'

import numpy as np
import triton_python_backend_utils as pb_utils


class TritonPythonModel:

    def initialize(self, args):
        model_config = json.loads(args['model_config'])

        # Resolve tokenizer directory: prefer local 'tokenizer/' subdir
        # next to this model.py (works with Triton cloud storage where
        # absolute build-time paths are invalid), then fall back to the
        # configured path.
        model_dir = os.path.dirname(os.path.abspath(__file__))
        local_tokenizer_dir = os.path.join(model_dir, 'tokenizer')
        if os.path.isdir(local_tokenizer_dir):
            tokenizer_dir = local_tokenizer_dir
        else:
            tokenizer_dir = model_config['parameters']['tokenizer_dir'][
                'string_value']

        # Check whether to apply Florence2's structured post-processing.
        apply_post_process = model_config['parameters'].get(
            'apply_florence2_post_process')
        if apply_post_process is not None:
            val = apply_post_process['string_value'].lower()
            self.apply_florence2_post_process = val in [
                'true', '1', 't', 'y', 'yes'
            ]
        else:
            self.apply_florence2_post_process = False

        # Florence2 uses custom code in the HF checkpoint directory.
        sys.path.insert(0, tokenizer_dir)

        if self.apply_florence2_post_process:
            from transformers import AutoProcessor
            self.processor = AutoProcessor.from_pretrained(
                tokenizer_dir, trust_remote_code=True)
        else:
            from transformers import AutoTokenizer
            self.processor = AutoTokenizer.from_pretrained(
                tokenizer_dir, legacy=False, padding_side='left',
                trust_remote_code=True)

        # Florence2 post-processing needs special tokens to identify
        # task boundaries, so skip_special_tokens defaults to False.
        skip_special_tokens = model_config['parameters'].get(
            'skip_special_tokens')
        if skip_special_tokens is not None:
            val = skip_special_tokens['string_value'].lower()
            self.skip_special_tokens = val in [
                'true', '1', 't', 'y', 'yes'
            ]
        else:
            self.skip_special_tokens = False

        # Parse output dtype.
        output_config = pb_utils.get_output_config_by_name(
            model_config, "OUTPUT")
        self.output_dtype = pb_utils.triton_string_to_numpy(
            output_config['data_type'])

    def execute(self, requests):
        responses = []
        for request in requests:
            # Read token output from tensorrt_llm backend.
            tokens_batch = pb_utils.get_input_tensor_by_name(
                request, 'TOKENS_BATCH').as_numpy()
            sequence_lengths = pb_utils.get_input_tensor_by_name(
                request, 'SEQUENCE_LENGTH').as_numpy()

            # The C++ tensorrtllm backend includes the internal batch
            # dimension in the output tensor.  Depending on Triton's
            # batch-dim handling the tensor may arrive as:
            #   3D [batch, beams, seq_len]  /  2D [beams, seq_len]
            # Handle both cases, and always take beam 0 (best beam).
            if tokens_batch.ndim == 2:
                tokens_batch = tokens_batch[np.newaxis, ...]
            if sequence_lengths.ndim == 1:
                sequence_lengths = sequence_lengths[np.newaxis, ...]

            batch_size = tokens_batch.shape[0]
            list_of_tokens = []
            for batch_idx in range(batch_size):
                seq_len = int(sequence_lengths[batch_idx][0])
                tokens = tokens_batch[batch_idx][0][:seq_len]
                list_of_tokens.append(tokens)

            # Decode tokens to text.
            decoded = self.processor.batch_decode(
                list_of_tokens,
                skip_special_tokens=self.skip_special_tokens)

            # Optionally apply Florence2 structured post-processing.
            task_query = pb_utils.get_input_tensor_by_name(
                request, 'TASK_QUERY')
            if (self.apply_florence2_post_process and task_query is not None
                    and hasattr(self.processor, 'post_process_generation')):
                task_text = task_query.as_numpy().flat[0]
                if isinstance(task_text, bytes):
                    task_text = task_text.decode('utf-8')
                task_token = self._extract_task_token(task_text)

                output_texts = []
                for text in decoded:
                    result = self.processor.post_process_generation(
                        text=text,
                        task=task_token,
                        image_size=(768, 768))
                    output_texts.append(
                        json.dumps(result).encode('utf-8'))
            else:
                output_texts = [text.encode('utf-8') for text in decoded]

            output_tensor = pb_utils.Tensor(
                'OUTPUT',
                np.array(output_texts).astype(self.output_dtype))

            inference_response = pb_utils.InferenceResponse(
                output_tensors=[output_tensor])
            responses.append(inference_response)
        return responses

    @staticmethod
    def _extract_task_token(task):
        """Extract the task token (e.g., '<CAPTION>', '<OD>') from the query."""
        match = re.match(r"\s*(<[^>]+>)", task)
        if match is None:
            return task
        return match.group(1)

    def finalize(self):
        print('Cleaning up florence2_postprocessing...')
