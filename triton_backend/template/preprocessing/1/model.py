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
"""Florence2 Triton preprocessing model.

Loads images, tokenizes task prompts, and constructs encoder input_ids
with virtual tokens for the DaViT image positions (p-tuning mechanism).

Reference: examples/models/core/enc_dec/run_florence2.py
"""

import base64
import io
import json
import os
import sys

# Ensure HOME is writable before importing packages that create caches there.
# In Triton containers the triton-server user may not have write access to its
# HOME directory, which breaks transformers trust_remote_code module caching.
if not os.access(os.environ.get('HOME', '/nonexistent'), os.W_OK):
    os.environ['HOME'] = '/tmp'

import numpy as np
import requests
import triton_python_backend_utils as pb_utils
from PIL import Image


class TritonPythonModel:

    def initialize(self, args):
        model_config = json.loads(args['model_config'])

        def _get_int_param(name, default=None):
            param = model_config['parameters'].get(name)
            if param is None:
                return default
            value = param.get('string_value')
            if value is None or value == "":
                return default
            return int(value)

        def _get_float_param(name, default=None):
            param = model_config['parameters'].get(name)
            if param is None:
                return default
            value = param.get('string_value')
            if value is None or value == "":
                return default
            return float(value)

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

        # Florence2 uses custom code in the HF checkpoint directory.
        sys.path.insert(0, tokenizer_dir)
        from transformers import AutoProcessor

        self.processor = AutoProcessor.from_pretrained(
            tokenizer_dir, trust_remote_code=True)

        # Read vocab_size from the Florence2 config.json.
        config_path = os.path.join(tokenizer_dir, 'config.json')
        with open(config_path, 'r') as f:
            florence2_config = json.load(f)
        text_config = florence2_config.get('text_config', {})
        self.vocab_size = text_config.get('vocab_size', 51289)

        # Token IDs for decoder prefix and EOS/PAD.
        self.decoder_start_token_id = text_config.get(
            'decoder_start_token_id', 2)
        self.eos_token_id = florence2_config.get(
            'eos_token_id', text_config.get('eos_token_id', 2))
        self.pad_token_id = florence2_config.get(
            'pad_token_id', text_config.get('pad_token_id', 1))

        # Defaults from generation_config.json.
        gen_config_path = os.path.join(tokenizer_dir,
                                       'generation_config.json')
        if os.path.exists(gen_config_path):
            with open(gen_config_path, 'r') as f:
                generation_config = json.load(f)
        else:
            generation_config = {}
        self.forced_bos_token_id = generation_config.get(
            'forced_bos_token_id', 0)

        # Beam ranking control: default to backend behavior (0.0) unless
        # explicitly provided via preprocessing model parameter.
        self.length_penalty = _get_float_param(
            'length_penalty',
            generation_config.get('length_penalty', 0.0))
        if self.length_penalty is None:
            self.length_penalty = 0.0

        # Fixed number of image tokens for Florence2 DaViT at 768x768.
        self.n_img_tokens = 577

        # Max lengths for request validation.
        self.max_input_len = _get_int_param('max_input_len')
        self.max_output_len = _get_int_param('max_output_len')

        # Parse output dtypes from Triton config.
        output_names = [
            "INPUT_ID", "REQUEST_INPUT_LEN", "DECODER_INPUT_ID",
            "REQUEST_DECODER_INPUT_LEN", "PIXEL_VALUES",
            "OUT_PROMPT_VOCAB_SIZE", "REQUEST_OUTPUT_LEN", "OUT_END_ID",
            "OUT_PAD_ID", "BEAM_WIDTH", "NO_REPEAT_NGRAM_SIZE", "LEN_PENALTY"
        ]
        for output_name in output_names:
            setattr(
                self, output_name.lower() + "_dtype",
                pb_utils.triton_string_to_numpy(
                    pb_utils.get_output_config_by_name(
                        model_config, output_name)['data_type']))

    def execute(self, requests):
        responses = []
        for request in requests:
            # --- Load image ---
            image = self._load_image(request)

            # --- Get task text ---
            query = pb_utils.get_input_tensor_by_name(
                request, 'QUERY').as_numpy()
            task_text = query[0][0]
            if isinstance(task_text, bytes):
                task_text = task_text.decode('utf-8')

            # --- Get requested output length ---
            request_output_len_tensor = pb_utils.get_input_tensor_by_name(
                request, 'REQUEST_OUTPUT_LEN')
            if request_output_len_tensor is None:
                if self.max_output_len is None:
                    raise ValueError(
                        "REQUEST_OUTPUT_LEN is not provided and max_output_len "
                        "parameter is not configured.")
                request_output_len = np.array(
                    [[self.max_output_len]], dtype=np.int32)
            else:
                request_output_len = request_output_len_tensor.as_numpy()

            requested_output_len = int(request_output_len.flat[0])
            if requested_output_len <= 0:
                raise ValueError(
                    f"Requested output length must be > 0, got "
                    f"{requested_output_len}.")
            if (self.max_output_len is not None
                    and requested_output_len > self.max_output_len):
                raise ValueError(
                    f"Requested output length is too large: "
                    f"{requested_output_len} "
                    f"(max_output_len={self.max_output_len}). "
                    "Reduce REQUEST_OUTPUT_LEN or rebuild engines with a larger "
                    "max_seq_len.")

            # --- Run Florence2 processor ---
            inputs = self.processor(
                text=task_text, images=image, return_tensors='np')
            pixel_values = inputs['pixel_values']  # [1, 3, 768, 768]
            task_ids = inputs['input_ids']  # [1, N_text]

            # --- Construct encoder input_ids with virtual tokens ---
            # Virtual token IDs: [vocab_size, vocab_size+1, ...,
            #                     vocab_size+n_img_tokens-1]
            virtual_ids = np.arange(
                self.vocab_size,
                self.vocab_size + self.n_img_tokens,
                dtype=np.int32).reshape(1, -1)
            encoder_input_ids = np.concatenate(
                [virtual_ids, task_ids.astype(np.int32)], axis=1)

            if (self.max_input_len is not None
                    and encoder_input_ids.shape[1] > self.max_input_len):
                raise ValueError(
                    f"Input is too long: {encoder_input_ids.shape[1]} tokens "
                    f"(max_input_len={self.max_input_len}). "
                    "Reduce the prompt or rebuild engines with a larger "
                    "max_input_len.")

            input_length = np.array(
                [[encoder_input_ids.shape[1]]], dtype=np.int32)

            # --- Decoder prefix: [decoder_start_token_id, forced_bos_token_id] ---
            # Matches HF Florence2's forced_bos_token_id behavior: [2, 0].
            decoder_input_ids = np.array(
                [[self.decoder_start_token_id, self.forced_bos_token_id]],
                dtype=np.int32)
            decoder_input_length = np.array(
                [[decoder_input_ids.shape[1]]], dtype=np.int32)

            # --- Build output tensors ---
            input_id_tensor = pb_utils.Tensor(
                'INPUT_ID',
                encoder_input_ids.astype(self.input_id_dtype))
            request_input_len_tensor = pb_utils.Tensor(
                'REQUEST_INPUT_LEN',
                input_length.astype(self.request_input_len_dtype))
            decoder_input_id_tensor = pb_utils.Tensor(
                'DECODER_INPUT_ID',
                decoder_input_ids.astype(self.decoder_input_id_dtype))
            request_decoder_input_len_tensor = pb_utils.Tensor(
                'REQUEST_DECODER_INPUT_LEN',
                decoder_input_length.astype(
                    self.request_decoder_input_len_dtype))
            pixel_values_tensor = pb_utils.Tensor(
                'PIXEL_VALUES',
                pixel_values.astype(self.pixel_values_dtype))
            prompt_vocab_size_tensor = pb_utils.Tensor(
                'OUT_PROMPT_VOCAB_SIZE',
                np.array(
                    [[self.n_img_tokens]],
                    dtype=np.int32).astype(self.out_prompt_vocab_size_dtype))
            request_output_len_tensor = pb_utils.Tensor(
                'REQUEST_OUTPUT_LEN',
                request_output_len.astype(self.request_output_len_dtype))
            end_id_tensor = pb_utils.Tensor(
                'OUT_END_ID',
                np.array([[self.eos_token_id]], dtype=np.int32))
            pad_id_tensor = pb_utils.Tensor(
                'OUT_PAD_ID',
                np.array([[self.pad_token_id]], dtype=np.int32))

            # Hardcode beam_width=3 and no_repeat_ngram_size=3 to prevent
            # degenerate beam search (BART's ~69% bos probability).
            beam_width_tensor = pb_utils.Tensor(
                'BEAM_WIDTH',
                np.array([[3]], dtype=np.int32))
            no_repeat_ngram_size_tensor = pb_utils.Tensor(
                'NO_REPEAT_NGRAM_SIZE',
                np.array([[3]], dtype=np.int32))
            len_penalty_tensor = pb_utils.Tensor(
                'LEN_PENALTY',
                np.array(
                    [[self.length_penalty]],
                    dtype=np.float32).astype(self.len_penalty_dtype))

            inference_response = pb_utils.InferenceResponse(
                output_tensors=[
                    input_id_tensor,
                    request_input_len_tensor,
                    decoder_input_id_tensor,
                    request_decoder_input_len_tensor,
                    pixel_values_tensor,
                    prompt_vocab_size_tensor,
                    request_output_len_tensor,
                    end_id_tensor,
                    pad_id_tensor,
                    beam_width_tensor,
                    no_repeat_ngram_size_tensor,
                    len_penalty_tensor,
                ])
            responses.append(inference_response)
        return responses

    def _load_image(self, request):
        """Load image from IMAGE_BYTES or IMAGE_URL input."""
        image_bytes = pb_utils.get_input_tensor_by_name(
            request, 'IMAGE_BYTES')
        image_url = pb_utils.get_input_tensor_by_name(request, 'IMAGE_URL')

        if image_bytes is not None:
            raw = image_bytes.as_numpy().tobytes()
            return Image.open(io.BytesIO(raw)).convert('RGB')
        elif image_url is not None:
            url = image_url.as_numpy()[0][0]
            if isinstance(url, bytes):
                url = url.decode('utf-8')
            if url.startswith('data:image'):
                image_base64 = url.split(',')[1]
                image_data = base64.b64decode(image_base64)
                return Image.open(io.BytesIO(image_data)).convert('RGB')
            else:
                return Image.open(
                    requests.get(url, stream=True).raw).convert('RGB')
        else:
            raise ValueError(
                "Either IMAGE_BYTES or IMAGE_URL must be provided")

    def finalize(self):
        print('Cleaning up florence2_preprocessing...')
