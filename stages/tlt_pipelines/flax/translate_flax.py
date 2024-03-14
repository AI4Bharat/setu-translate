import os
import sys
import subprocess
import torch
from torch.utils.data import DataLoader
import jax
jax.distributed.initialize()
import jax.numpy as jnp
import numpy as np

import pyarrow as pa
import pyarrow.parquet as pq
import pandas as pd

import re

from flax.jax_utils import replicate
from flax.training.common_utils import shard

# from transformers import AutoModelForSeq2SeqLM
from modeling_flax_indictrans import FlaxIndicTransForConditionalGeneration

from datasets.distributed import split_dataset_by_node

from datasets import (
    Dataset as HFDataset, 
    load_from_disk,
    load_dataset,
    concatenate_datasets,
    disable_caching
)
disable_caching()

from datasets.distributed import split_dataset_by_node

import glob
from functools import partial

import tqdm
import argparse

import gcsfs

from hashlib import sha256

from jax_smi import initialise_tracking
initialise_tracking()

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def parse_args():

    parser = argparse.ArgumentParser(description="Perform distributed inference")

    parser.add_argument(
        "--data_files",
        type=str,
    )

    parser.add_argument(
        "--base_save_dir",
        type=str,
    )

    parser.add_argument(
        "--batch_size",
        type=int,
    )

    parser.add_argument(
        "--save_batch_size",
        type=int,
        default=100_000,
        required=False,
    )

    parser.add_argument(
        "--total_procs",
        type=int
    )

    parser.add_argument(
        "--accelerator_type",
        type=str,
        choices=["gpu", "tpu"],
        default="gpu",
        required=False
    )

    parser.add_argument(
        "--devices",
        type=str,
        default=None,
        required=False,
    )

    parser.add_argument(
        "--setu_translate_root",
        type=str,
        required=True,
    )

    parser.add_argument(
        "--direction",
        type=str,
        required=False,
        choices=["en-indic", "indic-en", "indic-indic"],
        default="en-indic",
    )

    parser.add_argument(
        "--format",
        type=str,
        choices=["parquet", "arrow"],
        default="arrow",
        required=False,       
    )

    parser.add_argument(
        "--keep_in_memory",
        type=str2bool,
        default=None,
        required=False,
    )

    parser.add_argument(
        "--cache_dir",
        type=str,
        required=False,
        default=None
    )

    parser.add_argument(
        "--split",
        type=str,
        required=False,
        default="train",
    )

    parser.add_argument(
        "--num_loader_workers",
        type=int,
        required=False,
        default=32,
    )

    parser.add_argument(
        "--gcp_project",
        type=str,
        default=None,
        required=False
    )

    parser.add_argument(
        "--streaming",
        type=str2bool,
        default=False,
        required=False
    )

    args = parser.parse_args()

    return args


def padding_collator(
    batch,
    keys_to_pad=[
            ("input_ids", 1),
            ("attention_mask", 0),
        ]
    ):

    batch_out = {key: [] for key in batch[0].keys()}
    
    for sample in batch:
        for key in batch_out.keys():
            batch_out[key] += [sample[key]]
    
    for key, value_to_pad_with in keys_to_pad:

        len_list = list(map(lambda x: len(x), batch_out[key]))

        padding_length = max(len_list)
        array_list = []
        for i, x in enumerate(batch_out[key]):

            if len(x) < padding_length:
                padded_array = np.concatenate([np.full((padding_length - len(x)), value_to_pad_with), np.array(x)])
                array_list.append(padded_array)
            else:
                array_list.append(np.array(x))

        batch_out[key] = np.stack(array_list)

    return batch_out

def get_dataset_split(ds, split):
    split_set_info_pattern = r"\[(.*?)\]"
    split_set_info = re.search(split_set_info_pattern, split).group(1)
    if split_set_info:
        
        print(f"Split info found: split-info={split_set_info} ......")
        ds_size = len(ds)
        print(f"Dataset Size: {ds_size}")
        use_percentage = False
        start, end = split_set_info.split(":")

        if not len(start):
            start = None
        if not len(end):
            end = None
        if not start and not end:
            print(f"Using full dataset as split={split} ......")
            return ds

        if (not start and "%" in end) or (not end and "%" in start) or ("%" in start and "%" in end):
            use_percentage = True
            print(f"Using `percentage` based splitting....[{start}:{end}]")
            if start:
                start = float(start.rstrip("%"))
            else:
                start = 0
            if end:
                end = float(end.rstrip("%"))
            else:
                end = 100
        else:
            use_percentage = False
            print(f"Using `count` based splitting....[{start}:{end}]")
            if start:
                start = int(start)
            else:
                start = 0
            if end:
                end = int(end)
            else:
                end = ds_size
    
        if use_percentage:
            start = int(ds_size * (start / 100))
            end = int(ds_size * (end / 100))
            print(f"Calculated `start`:`end` indices from percentage: `[{start}:{end}]` ")

        if (start <= end <= ds_size):
            ds = ds.select(range(start, end))
        return ds
    else:
        print(f"Using full dataset as split={split} ......")
        return ds

def save_parquets(
    batch, idx, save_dir,
    process_rank, schema,
):
    
    print(f"Batch size inside map: {len(idx)}")

    batch_fname = sha256(f"{idx}".encode('utf-8')).hexdigest()

    batch_table = pa.Table.from_pydict(dict(batch), schema=schema)

    if save_dir.startswith("gs://"):
        os.makedirs(os.path.join("/tmp", "translation_job"), exist_ok=True)
        tmp_save_path = os.path.join("/tmp", "translation_job", f"{batch_fname}.parquet")
        pq.write_table(batch_table, tmp_save_path, compression='snappy')
        subprocess.run([
                "gsutil",
                "cp",
                tmp_save_path,
                save_dir if save_dir.endswith("/") else f"{save_dir}/"
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.STDOUT
        )
        subprocess.run([
                "rm",
                tmp_save_path
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.STDOUT
        )
    else:
        os.makedirs(save_dir, exist_ok=True)
        pq.write_table(
            batch_table, 
            os.path.join(save_dir, f"{batch_fname}.parquet"), 
            compression='snappy'
        )

if __name__ == "__main__":

    args = parse_args()

    process_rank = jax.process_index()
    total_process_count = jax.process_count()

    if process_rank == 0:
        print(f"Translate Job called using following command-line command: {sys.argv[0]}")
        print("Parsed arguments are: ")
        print(args)

    if args.accelerator_type == "gpu":
        os.environ['CUDA_VISIBLE_DEVICES'] = args.devices
        device_ids = [ int(idx.strip()) for idx in args.devices.split(",") if idx and len(idx.strip()) ]
        local_device_count = len(device_ids)
    else:
        local_device_count = jax.local_device_count()

    total_device_count = jax.device_count()

    if args.data_files.startswith("gs://"):
        use_gcs = True
        fs = gcsfs.GCSFileSystem(**storage_options)
        data_files = [f"gs://{data_file}" for data_file in fs.glob(args.data_files)]
    else:
        use_gcs = False
        data_files = glob.glob(args.data_files) if "*" in args.data_files else args.data_files

    if args.gcp_project:
        storage_options = {
            "project": args.gcp_project,
        }
    else:
        storage_options = None
        
    if process_rank == 0:
        print("All Devices: \n", jax.devices())
        print("Total Device Count: ", total_device_count)
        print("Local Device Count: ", local_device_count)
        print("Storage Options: ", storage_options)
        print("Total DataFiles: ", len(data_files) if isinstance(data_files, list) else 1)
        if isinstance(data_files, str):
            print("DataFolder to use: ", data_files)
        
    if isinstance(data_files, list):
        ds = load_dataset(
            args.format,
            data_files=data_files if isinstance(data_files, list) else None,
            num_proc=args.total_procs if not args.streaming else None,
            cache_dir=args.cache_dir if args.cache_dir not in ["None", "none", None] else None,
            split=args.split,
            keep_in_memory=args.keep_in_memory,
            storage_options=storage_options,
            streaming=args.streaming
        )
    elif isinstance(data_files, str) and not args.streaming:
        ds = load_from_disk(
            data_files,
            keep_in_memory=args.keep_in_memory,
            storage_options=storage_options,
        )
        if args.split:
            ds = get_dataset_split(ds=ds, split=args.split)
    else:
        raise Exception(
            """Invalid `data_files` and `args.streaming` combination. `args.streaming=True` can only be used when `args.data_files` is a glob pattern i.e `data_files` is a `list`.""")

    if process_rank == 0:
        print(
            "Dataset Size: ", 
            len(ds) if not args.streaming else "<INVALID AS USING STREAMING DATASET>"
        )
        print("Dataset Info: ",ds)

    process_ds = split_dataset_by_node(ds, rank=process_rank, world_size=total_process_count)
    print(
        f"Dataset Size for process-{process_rank}: ", 
        len(process_ds) if not args.streaming else "<INVALID AS USING STREAMING DATASET>"
    )

    data_loader = torch.utils.data.DataLoader(
        process_ds,
        batch_size=args.batch_size,
        drop_last=True,
        num_workers=32,
        collate_fn=padding_collator,
    )

    model = FlaxIndicTransForConditionalGeneration.from_pretrained(
        os.path.join(args.setu_translate_root, f"stages/tpu/flax_weights/{args.direction}/200m"),
        local_files_only=True,
        dtype=jnp.bfloat16,
    )

    params = replicate(model.params)

    def generate(
        batch,
        params,
    ):
        model.params = params
        return model.generate(
            **batch,
            num_beams=1,
            num_return_sequences=1,
            max_length=256,
            do_sample=False,
        ).sequences

    p_generate = jax.pmap(generate)   

    def run_inference_step(batch, params, run_ds):

        input_batch = {
            "input_ids": shard(jnp.array(batch["input_ids"])),
            "attention_mask": shard(jnp.array(batch["attention_mask"]))
        }

        outputs = p_generate(input_batch, params)

        outputs = outputs.block_until_ready()

        if local_device_count != 1:
            outputs = outputs.reshape(-1, *outputs.shape[2:])
        else:
            outputs = outputs[0]

        run_ds = concatenate_datasets(
            [
                run_ds,
                HFDataset.from_dict(
                    {
                        "input_ids": batch["input_ids"], 
                        "sid": batch["sids"], 
                        "sub_str": batch["sub_strs"], 
                        "tlt_idx": batch["tlt_idx"], 
                        "placeholder_entity_map": batch["placeholder_entity_map"],
                        "translated_input_ids": outputs,
                        "tlt_file_loc": batch["tlt_file_loc"],
                    }
                ),
            ],
        )

        return run_ds

    run_ds = HFDataset.from_dict(
        {   
            key: [] for key in [
                "doc_id", 
                "sids" 
                "sub_strs", 
                "tlt_idx", 
                "placeholder_entity_map", 
                "translation_ids"
            ]
        },
    ) 

    if process_rank == 0:
        for idx, batch in tqdm.tqdm(
            enumerate(data_loader, 0), 
            unit=f"ba: {args.batch_size} samples/ba", 
            total=len(data_loader) if not args.streaming else None,
        ):
            run_ds = run_inference_step(batch, params, run_ds)
    else:
        for idx, batch in enumerate(data_loader, 0):
            run_ds = run_inference_step(batch, params, run_ds)

    save_dir = os.path.join(
        args.base_save_dir, f"devices_hw:{args.accelerator_type}_{process_rank}"
    )

    if not save_dir.startswith("gs://"):
        os.makedirs(save_dir, exist_ok=True)

    parquet_schema = pa.schema(
        [
            pa.field('input_ids', pa.list_(pa.int32())),
            pa.field('sid', pa.string()),
            pa.field('sub_str', pa.string()),
            pa.field('tlt_idx', pa.int64()),
            pa.field('placeholder_entity_map', pa.string()),
            pa.field('translated_input_ids', pa.list_(pa.int32())),
            pa.field('tlt_file_loc', pa.string()),
        ]
    )
    
    print(f"Outputs Dataset for process-rank:{process_rank} is {len(run_ds)} .......")

    print(f"Using batch-size: {args.save_batch_size} for saving to parquets....")

    _ = run_ds.map(
        partial(
            save_parquets,
            save_dir=save_dir,
            process_rank=process_rank,
            schema=parquet_schema,
        ),
        batched=True,
        batch_size=args.save_batch_size,
        num_proc=args.total_procs,
        with_indices=True,
        load_from_cache_file=False,
        keep_in_memory=True,
    )
