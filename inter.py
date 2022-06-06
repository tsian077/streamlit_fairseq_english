#!/usr/bin/env python3 -u
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
Translate raw text with a trained model. Batches data on-the-fly.
"""

# import ast
import fileinput
import logging
import math
import os
import sys
import time
from argparse import Namespace
from collections import namedtuple
import librosa

import numpy as np
# import torch

# from fairseq import checkpoint_utils, tasks, utils
# from fairseq.token_generation_constraints import  unpack_constraints
# from generate import get_symbols_to_strip_from_output
# import base64

import fairseq
from fairseq.checkpoint_utils import load_model_ensemble
from fairseq.token_generation_constraints import  unpack_constraints
from fairseq.utils import import_user_module,split_paths,load_align_dict,resolve_max_positions,strip_pad,post_process_prediction
from fairseq.tasks import setup_task
from generate import get_symbols_to_strip_from_output

Batch = namedtuple("Batch", "ids src_tokens src_lengths constraints")
Translation = namedtuple("Translation", "src_str hypos pos_scores alignments")


def buffered_read(input, buffer_size):
    buffer = []
    with fileinput.input(files=[input], openhook=fileinput.hook_encoded("utf-8")) as h:
        for src_str in h:
            buffer.append(src_str.strip())
            if len(buffer) >= buffer_size:
                yield buffer
                buffer = []

    if len(buffer) > 0:
        yield buffer


def make_batches(lines,  task, max_positions, encode_fn):
    def encode_fn_target(x):
        return encode_fn(x)
    

    constraints_tensor = None

    tokens, lengths = task.get_interactive_tokens_and_lengths(lines, encode_fn)

    itr = task.get_batch_iterator(
        dataset=task.build_dataset_for_inference(
            tokens, lengths, constraints=constraints_tensor
        ),
        max_tokens=50000,
        max_sentences=None,
        max_positions=max_positions,
        ignore_invalid_inputs=False,
    ).next_epoch_itr(shuffle=False)
    for batch in itr:
        ids = batch["id"]
        src_tokens = batch["net_input"]["src_tokens"]
        src_lengths = batch["net_input"]["src_lengths"]
        constraints = batch.get("constraints", None)

        yield Batch(
            ids=ids,
            src_tokens=src_tokens,
            src_lengths=src_lengths,
            constraints=constraints,
        )


# def main():
def s2t():
    
    y,sr = librosa.load('input.wav',sr=44100)
    y_16k = librosa.resample(y,orig_sr=sr,target_sr=8000)
    librosa.output.write_wav('input_16k.wav',y,sr)


    start_time = time.time()
    total_translate_time = 0

    
    common = {'_name': None, 'no_progress_bar': False, 'log_interval': 100, 'log_format': None, 'log_file': None, 'tensorboard_logdir': None, 'wandb_project': None, 'azureml_logging': False, 'seed': 1, 'cpu': False, 'tpu': False, 'bf16': False, 'memory_efficient_bf16': False, 'fp16': False, 'memory_efficient_fp16': False, 'fp16_no_flatten_grads': False, 'fp16_init_scale': 128, 'fp16_scale_window': None, 'fp16_scale_tolerance': 0.0, 'on_cpu_convert_precision': False, 'min_loss_scale': 0.0001, 'threshold_loss_scale': None, 'amp': False, 'amp_batch_retries': 2, 'amp_init_scale': 128, 'amp_scale_window': None, 'user_dir': None, 'empty_cache_freq': 0, 'all_gather_list_size': 16384, 'model_parallel_size': 1, 'quantization_config_path': None, 'profile': False, 'reset_logging': False, 'suppress_crashes': False, 'use_plasma_view': False, 'plasma_path': '/tmp/plasma'}
    import_user_module(common)


    use_cuda = False

    
    dic_task = {'no_progress_bar': False, 'log_interval': 100, 'log_format': None, 'log_file': None, 'tensorboard_logdir': None, 'wandb_project': None, 'azureml_logging': False, 'seed': 1, 'cpu': False, 'tpu': False, 'bf16': False, 'memory_efficient_bf16': False, 'fp16': False, 'memory_efficient_fp16': False, 'fp16_no_flatten_grads': False, 'fp16_init_scale': 128, 'fp16_scale_window': None, 'fp16_scale_tolerance': 0.0, 'on_cpu_convert_precision': False, 'min_loss_scale': 0.0001, 'threshold_loss_scale': None, 'amp': False, 'amp_batch_retries': 2, 'amp_init_scale': 128, 'amp_scale_window': None, 'user_dir': None, 'empty_cache_freq': 0, 'all_gather_list_size': 16384, 'model_parallel_size': 1, 'quantization_config_path': None, 'profile': False, 'reset_logging': False, 'suppress_crashes': False, 'use_plasma_view': False, 'plasma_path': '/tmp/plasma', 'criterion': 'cross_entropy', 'tokenizer': None, 'bpe': None, 'optimizer': None, 'lr_scheduler': 'fixed', 'simul_type': None, 'scoring': 'bleu', 'task': 'speech_to_text', 'num_workers': 1, 'skip_invalid_size_inputs_valid_test': False, 'max_tokens': 50000, 'batch_size': None, 'required_batch_size_multiple': 8, 'required_seq_len_multiple': 1, 'dataset_impl': None, 'data_buffer_size': 10, 'train_subset': 'train', 'valid_subset': 'valid', 'combine_valid_subsets': None, 'ignore_unused_valid_subsets': False, 'validate_interval': 1, 'validate_interval_updates': 0, 'validate_after_updates': 0, 'fixed_validation_seed': None, 'disable_validation': False, 'max_tokens_valid': 50000, 'batch_size_valid': None, 'max_valid_steps': None, 'curriculum': 0, 'gen_subset': 'test', 'num_shards': 1, 'shard_id': 0, 'grouped_shuffling': False, 'update_epoch_batch_itr': False, 'update_ordered_indices_seed': False, 'distributed_world_size': 1, 'distributed_num_procs': 1, 'distributed_rank': 0, 'distributed_backend': 'nccl', 'distributed_init_method': None, 'distributed_port': -1, 'device_id': 0, 'distributed_no_spawn': False, 'ddp_backend': 'pytorch_ddp', 'ddp_comm_hook': 'none', 'bucket_cap_mb': 25, 'fix_batches_to_gpus': False, 'find_unused_parameters': False, 'gradient_as_bucket_view': False, 'fast_stat_sync': False, 'heartbeat_timeout': -1, 'broadcast_buffers': False, 'slowmo_momentum': None, 'slowmo_base_algorithm': 'localsgd', 'localsgd_frequency': 3, 'nprocs_per_node': 1, 'pipeline_model_parallel': False, 'pipeline_balance': None, 'pipeline_devices': None, 'pipeline_chunks': 0, 'pipeline_encoder_balance': None, 'pipeline_encoder_devices': None, 'pipeline_decoder_balance': None, 'pipeline_decoder_devices': None, 'pipeline_checkpoint': 'never', 'zero_sharding': 'none', 'no_reshard_after_forward': False, 'fp32_reduce_scatter': False, 'cpu_offload': False, 'use_sharded_state': False, 'not_fsdp_flatten_parameters': False, 'path': 'model/checkpoint_best.pt', 'post_process': None, 'quiet': False, 'model_overrides': '{}', 'results_path': None, 'beam': 5, 'nbest': 1, 'max_len_a': 0, 'max_len_b': 200, 'min_len': 1, 'match_source_len': False, 'unnormalized': False, 'no_early_stop': False, 'no_beamable_mm': False, 'lenpen': 1, 'unkpen': 0, 'replace_unk': None, 'sacrebleu': False, 'score_reference': False, 'prefix_size': 0, 'no_repeat_ngram_size': 0, 'sampling': False, 'sampling_topk': -1, 'sampling_topp': -1.0, 'constraints': None, 'temperature': 1.0, 'diverse_beam_groups': -1, 'diverse_beam_strength': 0.5, 'diversity_rate': -1.0, 'print_alignment': None, 'print_step': False, 'lm_path': None, 'lm_weight': 0.0, 'iter_decode_eos_penalty': 0.0, 'iter_decode_max_iter': 10, 'iter_decode_force_max_iter': False, 'iter_decode_with_beam': 1, 'iter_decode_with_external_reranker': False, 'retain_iter_history': False, 'retain_dropout': False, 'retain_dropout_modules': None, 'decoding_format': None, 'no_seed_provided': False, 'save_dir': 'checkpoints', 'restore_file': 'checkpoint_last.pt', 'continue_once': None, 'finetune_from_model': None, 'reset_dataloader': False, 'reset_lr_scheduler': False, 'reset_meters': False, 'reset_optimizer': False, 'optimizer_overrides': '{}', 'save_interval': 1, 'save_interval_updates': 0, 'keep_interval_updates': -1, 'keep_interval_updates_pattern': -1, 'keep_last_epochs': -1, 'keep_best_checkpoints': -1, 'no_save': False, 'no_epoch_checkpoints': False, 'no_last_checkpoints': False, 'no_save_optimizer_state': False, 'best_checkpoint_metric': 'loss', 'maximize_best_checkpoint_metric': False, 'patience': -1, 'checkpoint_suffix': '', 'checkpoint_shard_count': 1, 'load_checkpoint_on_all_dp_ranks': False, 'write_checkpoints_asynchronously': False, 'buffer_size': 0, 'input': '-', 'data': '', 'config_yaml': 'config.yaml', 'max_source_positions': 6000, 'max_target_positions': 1024, 'force_anneal': None, 'lr_shrink': 0.1, 'warmup_updates': 0, 'pad': 1, 'eos': 2, 'unk': 3, '_name': 'speech_to_text'}

    task = Namespace(**dic_task)
    task = setup_task(task)
    
    overrides = {}
    
    models, _model_args = load_model_ensemble(
        split_paths("model/checkpoint_best.pt"),
        arg_overrides=overrides,
        task=task,
        suffix="",
        strict=(1 == 1),
        num_shards=1,
    )
    
    
    # Set dictionaries
    src_dict = task.source_dictionary
    tgt_dict = task.target_dictionary

    generator = task.build_generator(models,{'_name': None, 'beam': 5, 'nbest': 1, 'max_len_a': 0.0, 'max_len_b': 200, 'min_len': 1, 'match_source_len': False, 'unnormalized': False, 'no_early_stop': False, 'no_beamable_mm': False, 'lenpen': 1.0, 'unkpen': 0.0, 'replace_unk': None, 'sacrebleu': False, 'score_reference': False, 'prefix_size': 0, 'no_repeat_ngram_size': 0, 'sampling': False, 'sampling_topk': -1, 'sampling_topp': -1.0, 'constraints': None, 'temperature': 1.0, 'diverse_beam_groups': -1, 'diverse_beam_strength': 0.5, 'diversity_rate': -1.0, 'print_alignment': None, 'print_step': False, 'lm_path': None, 'lm_weight': 0.0, 'iter_decode_eos_penalty': 0.0, 'iter_decode_max_iter': 10, 'iter_decode_force_max_iter': False, 'iter_decode_with_beam': 1, 'iter_decode_with_external_reranker': False, 'retain_iter_history': False, 'retain_dropout': False, 'retain_dropout_modules': None, 'decoding_format': None, 'no_seed_provided': False})

    # Handle tokenization and BPE
    
    
    tokenizer = task.build_tokenizer(None)
    bpe = task.build_bpe(None)

    def encode_fn(x):
        if tokenizer is not None:
            x = tokenizer.encode(x)
        if bpe is not None:
            x = bpe.encode(x)
        return x

    def decode_fn(x):
        if bpe is not None:
            x = bpe.decode(x)
        if tokenizer is not None:
            x = tokenizer.decode(x)
        return x

    # Load alignment dictionary for unknown word replacement
    # (None if no unknown word replacement, empty if no path to align dictionary)
#     print()
    align_dict = load_align_dict(None)

    max_positions = resolve_max_positions(
        task.max_positions(), *[model.max_positions() for model in models]
    )


    start_id = 0

    results = []


    # inputs = ['ser_output5.wav']
    # inputs = ['input.wav']
    inputs = ['input_16k.wav']

    for batch in make_batches(inputs,  task, max_positions, encode_fn):

        bsz = batch.src_tokens.size(0)

        src_tokens = batch.src_tokens
        src_lengths = batch.src_lengths
        constraints = batch.constraints

        if use_cuda:
            src_tokens = src_tokens.cuda()
            src_lengths = src_lengths.cuda()
            if constraints is not None:
                constraints = constraints.cuda()

        sample = {
            "net_input": {
                "src_tokens": src_tokens,
                "src_lengths": src_lengths,
            },
        }
        
        translate_start_time = time.time()
        translations = task.inference_step(
            generator, models, sample, constraints=constraints
        )
        translate_time = time.time() - translate_start_time
        total_translate_time += translate_time
        list_constraints = [[] for _ in range(bsz)]
        if None:
            list_constraints = [unpack_constraints(c) for c in constraints]
        for i, (id, hypos) in enumerate(zip(batch.ids.tolist(), translations)):
            src_tokens_i = strip_pad(src_tokens[i], tgt_dict.pad())
            constraints = list_constraints[i]
            results.append(
                (
                    start_id + id,
                    src_tokens_i,
                    hypos,
                    {
                        "constraints": constraints,
                        "time": translate_time / len(translations),
                    },
                )
            )
#     print('132')
#     print(cfg.common_eval.post_process)
    # sort output to match input order
    for id_, src_tokens, hypos, info in sorted(results, key=lambda x: x[0]):
        src_str = ""
        if src_dict is not None:
            src_str = src_dict.string(src_tokens, None)
            print("S-{}\t{}".format(id_, src_str))
            print("W-{}\t{:.3f}\tseconds".format(id_, info["time"]))
            for constraint in info["constraints"]:
                print(
                    "C-{}\t{}".format(
                        id_,
                        tgt_dict.string(constraint, None),
                    )
                )

        # Process top predictions
        for hypo in hypos[: min(len(hypos), 1)]:
            hypo_tokens, hypo_str, alignment = post_process_prediction(
                hypo_tokens=hypo["tokens"].int().cpu(),
                src_str=src_str,
                alignment=hypo["alignment"],
                align_dict=align_dict,
                tgt_dict=tgt_dict,
                remove_bpe=None,
                extra_symbols_to_ignore=get_symbols_to_strip_from_output(generator),
            )
            detok_hypo_str = decode_fn(hypo_str)
            score = hypo["score"] / math.log(2)  # convert to base 2
            # original hypothesis (after tokenization and BPE)

            # detokenized hypothesis
            print("D-{}\t{}\t{}".format(id_, score, detok_hypo_str))


    # update running id_ counter
    start_id += len(inputs)
    print("Total time: {:.3f} seconds; translation time: {:.3f}".format(
           time.time() - start_time, total_translate_time
        ))
    
    return detok_hypo_str




if __name__ == "__main__":
    s2t()
#     cli_main()
#     main()