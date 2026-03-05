#!/usr/bin/env python3
# Copyright 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""
Standalone integration test for the Florence2 Triton ensemble pipeline.

Simulates the 4-step pipeline WITHOUT Triton Server by calling the same
logic that each model.py uses:
    Step 1: Preprocessing  (image load, tokenize, virtual token IDs)
    Step 2: Vision         (DaViT TRT engine → image_features)
    Step 3: BART enc/dec   (TRT-LLM EncDecModelRunner)
    Step 4: Postprocessing (decode + Florence2 post-process)

Compares output against run_florence2.py reference.

Usage:
    cd /tmp && python3 /path/to/test_pipeline.py \
        --model_dir /workspaces/florence2/Florence-2-large-ft \
        --engine_dir /tmp/florence2_engine \
        --vision_engine_dir /tmp/florence2_engine/vision \
        --image /workspaces/florence2/images/car.jpg
"""
import argparse
import json
import os
import re
import sys
import time

import numpy as np
import torch

# ---------------------------------------------------------------
# Step 1: Preprocessing (same logic as florence2_preprocessing/1/model.py)
# ---------------------------------------------------------------
def step_preprocessing(processor, config, image_path, task_text):
    """Simulate preprocessing model."""
    from PIL import Image

    image = Image.open(image_path).convert('RGB')
    inputs = processor(text=task_text, images=image, return_tensors='np')
    pixel_values = inputs['pixel_values']  # [1, 3, 768, 768]
    task_ids = inputs['input_ids']          # [1, N_text]

    text_config = config.get('text_config', {})
    vocab_size = text_config.get('vocab_size', 51289)
    n_img_tokens = 577

    # Virtual token IDs
    virtual_ids = np.arange(
        vocab_size, vocab_size + n_img_tokens, dtype=np.int32
    ).reshape(1, -1)
    encoder_input_ids = np.concatenate(
        [virtual_ids, task_ids.astype(np.int32)], axis=1)

    # Decoder prefix: [decoder_start_token_id, forced_bos_token_id]
    decoder_start = text_config.get('decoder_start_token_id', 2)
    gen_config_path = os.path.join(args.model_dir, 'generation_config.json')
    if os.path.exists(gen_config_path):
        with open(gen_config_path) as f:
            gen = json.load(f)
        forced_bos = gen.get('forced_bos_token_id', 0)
    else:
        forced_bos = 0
    decoder_input_ids = np.array([[decoder_start, forced_bos]], dtype=np.int32)

    eos_token_id = config.get('eos_token_id',
                               text_config.get('eos_token_id', 2))
    pad_token_id = config.get('pad_token_id',
                               text_config.get('pad_token_id', 1))

    print(f"  Preprocessing:")
    print(f"    pixel_values: {pixel_values.shape} dtype={pixel_values.dtype}")
    print(f"    encoder_input_ids: {encoder_input_ids.shape} "
          f"(virtual={n_img_tokens} + text={task_ids.shape[1]})")
    print(f"    decoder_input_ids: {decoder_input_ids.tolist()}")
    print(f"    eos_token_id={eos_token_id} pad_token_id={pad_token_id}")

    return {
        'pixel_values': pixel_values.astype(np.float16),
        'encoder_input_ids': encoder_input_ids,
        'input_length': encoder_input_ids.shape[1],
        'decoder_input_ids': decoder_input_ids,
        'decoder_input_length': decoder_input_ids.shape[1],
        'eos_token_id': eos_token_id,
        'pad_token_id': pad_token_id,
        'beam_width': 3,
        'no_repeat_ngram_size': 3,
    }


# ---------------------------------------------------------------
# Step 2: Vision (same logic as florence2_vision/1/model.py)
# ---------------------------------------------------------------
def step_vision(vision_engine_dir, pixel_values_np):
    """Simulate vision model using TRT Session."""
    from tensorrt_llm.runtime import Session, TensorInfo
    from tensorrt_llm._utils import trt_dtype_to_torch, torch_dtype_to_trt

    # Load engine
    engine_path = os.path.join(vision_engine_dir, 'model.engine')
    with open(engine_path, 'rb') as f:
        engine_buffer = f.read()
    session = Session.from_serialized_engine(engine_buffer)

    # Read config
    config_path = os.path.join(vision_engine_dir, 'config.json')
    with open(config_path) as f:
        vconfig = json.load(f)
    precision = vconfig['builder_config']['precision']
    dtype_map = {
        'float16': torch.float16,
        'float32': torch.float32,
        'bfloat16': torch.bfloat16,
    }
    vision_dtype = dtype_map[precision]

    # Run engine
    pixel_values = torch.from_numpy(pixel_values_np).to('cuda', dtype=vision_dtype)
    stream = torch.cuda.current_stream()

    input_info = [
        TensorInfo('pixel_values',
                   torch_dtype_to_trt(pixel_values.dtype),
                   pixel_values.shape)
    ]
    output_info = session.infer_shapes(input_info)

    outputs = {}
    for t in output_info:
        outputs[t.name] = torch.empty(
            tuple(t.shape),
            dtype=trt_dtype_to_torch(t.dtype),
            device='cuda')

    ok = session.run({'pixel_values': pixel_values}, outputs, stream.cuda_stream)
    assert ok, "Vision TRT engine execution failed"
    stream.synchronize()

    image_features = outputs['image_features']
    print(f"  Vision:")
    print(f"    image_features: {image_features.shape} dtype={image_features.dtype}")
    print(f"    prompt_vocab_size: {image_features.shape[1]}")

    return image_features  # [B, 577, 1024]


# ---------------------------------------------------------------
# Step 3: BART enc/dec (uses EncDecModelRunner from TRT-LLM)
# ---------------------------------------------------------------
def step_tensorrt_llm(engine_dir, prep_outputs, image_features):
    """Simulate tensorrt_llm model using EncDecModelRunner.

    Uses the same API as run_florence2.py: EncDecModelRunner.from_engine()
    + encoder_run() + decoder_session-based beam search.
    """
    from tensorrt_llm.runtime import EncDecModelRunner, SamplingConfig
    from tensorrt_llm._utils import str_dtype_to_torch

    # Load enc/dec runner (same as run_florence2.py:1218)
    runner = EncDecModelRunner.from_engine(
        "enc_dec", engine_dir, debug_mode=False)
    encoder_config = runner.encoder_model_config

    batch_size = image_features.shape[0]
    n_img_tokens = image_features.shape[1]

    # Prompt embedding table (same as run_florence2.py:870)
    prompt_dtype = str_dtype_to_torch(encoder_config.dtype)
    prompt_embedding_table = image_features.to(
        dtype=prompt_dtype).contiguous().reshape(-1, image_features.shape[-1])

    # Prompt tasks
    enc_remove_pad = encoder_config.remove_input_padding
    encoder_input_ids = torch.from_numpy(
        prep_outputs['encoder_input_ids']).to('cuda', dtype=torch.int32)
    pad_token_id = prep_outputs['pad_token_id']

    if enc_remove_pad:
        input_lengths = 1 + (encoder_input_ids[:, 1:] != pad_token_id).sum(
            dim=1, dtype=torch.int32)
        prompt_tasks = torch.repeat_interleave(
            torch.arange(batch_size, dtype=torch.int32, device='cuda'),
            input_lengths)
    else:
        prompt_tasks = torch.arange(
            batch_size, dtype=torch.int32, device='cuda').unsqueeze(1)

    prompt_vocab_size = torch.tensor(
        [n_img_tokens], dtype=torch.int32, device='cuda')

    # Process encoder input (same as run_florence2.py:894)
    (encoder_input_ids_p, encoder_input_lengths, encoder_max_input_length,
     prompt_tasks_p, _) = runner.process_input(
        encoder_input_ids, enc_remove_pad, pad_token_id, prompt_tasks, None)

    # Run encoder (same as run_florence2.py:901)
    encoder_output = runner.encoder_run(
        encoder_input_ids_p,
        encoder_input_lengths,
        encoder_max_input_length,
        debug_mode=False,
        prompt_embedding_table=prompt_embedding_table,
        prompt_tasks=prompt_tasks_p,
        prompt_vocab_size=prompt_vocab_size,
    )

    # Process decoder input (same as run_florence2.py:912)
    decoder_input_ids = torch.from_numpy(
        prep_outputs['decoder_input_ids']).to('cuda', dtype=torch.int32)
    eos_token_id = prep_outputs['eos_token_id']
    dec_remove_pad = runner.decoder_model_config.remove_input_padding

    (decoder_input_ids_p, decoder_input_lengths, decoder_max_input_length,
     _, _) = runner.process_input(
        decoder_input_ids, dec_remove_pad, pad_token_id, None, None)

    # SamplingConfig (same as run_florence2.py:918)
    sampling_config = SamplingConfig(
        end_id=eos_token_id,
        pad_id=pad_token_id,
        num_beams=prep_outputs['beam_width'],
        min_length=1,
        return_dict=True)
    sampling_config.update(
        output_cum_log_probs=True,
        no_repeat_ngram_size=prep_outputs['no_repeat_ngram_size'])

    # Setup decoder session (same as run_florence2.py:938)
    max_new_tokens = 128
    runner.decoder_session.setup(
        decoder_input_lengths.size(0),
        decoder_max_input_length,
        max_new_tokens,
        prep_outputs['beam_width'],
        max_attention_window_size=None,
        encoder_max_input_length=encoder_max_input_length,
    )

    # Run decoder (same as run_florence2.py via decoder_session.decode)
    output = runner.decoder_session.decode(
        decoder_input_ids_p,
        decoder_input_lengths,
        sampling_config,
        encoder_output=encoder_output,
        encoder_input_lengths=encoder_input_lengths,
        return_dict=True,
    )

    output_ids = output['output_ids']  # [B, beams, seq_len]
    torch.cuda.synchronize()

    # Compute sequence lengths by finding eos_token_id position.
    # The C++ backend outputs sequence_length separately, but the Python
    # GenerationSession return_dict may not include it.
    # Fallback: scan for eos or use full length.
    eos = prep_outputs['eos_token_id']
    pad = prep_outputs['pad_token_id']
    batch_size_out = output_ids.shape[0]
    n_beams = output_ids.shape[1]
    seq_dim = output_ids.shape[2]
    seq_lengths = np.zeros((batch_size_out, n_beams), dtype=np.int32)
    for bi in range(batch_size_out):
        for bm in range(n_beams):
            tokens = output_ids[bi, bm].cpu().numpy()
            # Find first eos after position 0
            eos_positions = np.where(tokens[1:] == eos)[0]
            if len(eos_positions) > 0:
                seq_lengths[bi, bm] = eos_positions[0] + 2  # +1 offset, +1 inclusive
            else:
                # No eos found; strip trailing pads
                non_pad = np.where(tokens != pad)[0]
                seq_lengths[bi, bm] = non_pad[-1] + 1 if len(non_pad) > 0 else seq_dim

    print(f"  TensorRT-LLM:")
    print(f"    output_ids: {output_ids.shape}")
    print(f"    sequence_lengths: {seq_lengths}")
    print(f"    output dict keys: {list(output.keys()) if isinstance(output, dict) else 'N/A'}")

    return output_ids.cpu().numpy(), seq_lengths


# ---------------------------------------------------------------
# Step 4: Postprocessing (same logic as florence2_postprocessing/1/model.py)
# ---------------------------------------------------------------
def step_postprocessing(processor, tokens_batch, sequence_lengths, task_text):
    """Simulate postprocessing model."""
    # Handle both 2D and 3D
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

    # Decode (skip_special_tokens=False for Florence2 post-processing)
    decoded = processor.batch_decode(list_of_tokens, skip_special_tokens=False)
    print(f"  Postprocessing:")
    print(f"    Raw decoded: {decoded}")

    # Florence2 structured post-processing
    match = re.match(r"\s*(<[^>]+>)", task_text)
    task_token = match.group(1) if match else task_text

    results = []
    for text in decoded:
        result = processor.post_process_generation(
            text=text, task=task_token, image_size=(768, 768))
        results.append(result)
        print(f"    Post-processed: {result}")

    return results


# ---------------------------------------------------------------
# Main
# ---------------------------------------------------------------
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Test Florence2 Triton ensemble pipeline')
    parser.add_argument('--model_dir', required=True,
                        help='Florence-2 HF checkpoint directory')
    parser.add_argument('--engine_dir', required=True,
                        help='TRT-LLM engine directory (encoder/decoder)')
    parser.add_argument('--vision_engine_dir', required=True,
                        help='DaViT vision TRT engine directory')
    parser.add_argument('--image', required=True,
                        help='Path to test image')
    parser.add_argument('--task', default='<CAPTION>',
                        help='Florence2 task (default: <CAPTION>)')
    args = parser.parse_args()

    # Load config and processor
    sys.path.insert(0, args.model_dir)
    from transformers import AutoProcessor

    with open(os.path.join(args.model_dir, 'config.json')) as f:
        config = json.load(f)
    processor = AutoProcessor.from_pretrained(
        args.model_dir, trust_remote_code=True)

    print("=" * 60)
    print("Florence2 Triton Ensemble Pipeline Test")
    print("=" * 60)
    print(f"Model: {args.model_dir}")
    print(f"Image: {args.image}")
    print(f"Task:  {args.task}")
    print()

    # Step 1: Preprocessing
    print("[Step 1] Preprocessing")
    t0 = time.time()
    prep = step_preprocessing(processor, config, args.image, args.task)
    print(f"  Time: {(time.time() - t0)*1000:.1f}ms\n")

    # Step 2: Vision
    print("[Step 2] Vision (DaViT TRT)")
    t0 = time.time()
    image_features = step_vision(args.vision_engine_dir, prep['pixel_values'])
    print(f"  Time: {(time.time() - t0)*1000:.1f}ms\n")

    # Step 3: BART enc/dec
    print("[Step 3] TensorRT-LLM (BART enc/dec)")
    t0 = time.time()
    output_ids, sequence_lengths = step_tensorrt_llm(
        args.engine_dir, prep, image_features)
    print(f"  Time: {(time.time() - t0)*1000:.1f}ms\n")

    # Step 4: Postprocessing
    print("[Step 4] Postprocessing")
    t0 = time.time()
    results = step_postprocessing(
        processor, output_ids, sequence_lengths, args.task)
    print(f"  Time: {(time.time() - t0)*1000:.1f}ms\n")

    # Verify against expected output
    print("=" * 60)
    print("VERIFICATION")
    print("=" * 60)
    expected_caption = "A green car parked in front of a yellow building."
    actual_caption = results[0].get(args.task, '')
    match = actual_caption == expected_caption
    print(f"Expected: {expected_caption}")
    print(f"Actual:   {actual_caption}")
    print(f"Match:    {'PASS' if match else 'FAIL'}")
    print("=" * 60)

    sys.exit(0 if match else 1)
