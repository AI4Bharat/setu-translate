import os
import torch
from torch.utils.data import DataLoader
import jax
import jax.numpy as jnp
import numpy as np

from flax.jax_utils import replicate
from flax.training.common_utils import shard

# from transformers import AutoModelForSeq2SeqLM
from modeling_flax_indictrans import FlaxIndicTransForConditionalGeneration


from datasets import (
    Dataset as HFDataset, 
    load_from_disk,
    load_dataset,
    concatenate_datasets
)

from datasets.distributed import split_dataset_by_node

import glob
from functools import partial

import tqdm
import argparse

from jax_smi import initialise_tracking
initialise_tracking()

def parse_args():

    parser = argparse.ArgumentParser(description="Perform distributed inference")

    parser.add_argument(
        "--root_dir",
        type=str,
    )

    parser.add_argument(
        "--data_files",
        type=str,
    )

    parser.add_argument(
        "--cache_dir",
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
        "--total_procs",
        type=int
    )

    parser.add_argument(
        "--devices",
        type=str,
        required=True,
    )

    args = parser.parse_args()

    return args


def padding_collator(
    batch,
    keys_to_pad=[
            ("input_ids", 1), 
            ("attention_masks", 0),
        ]
    ):

    batch_out = {key: [] for key in batch[0].keys()}
    
    for sample in batch:
        for key in batch_out.keys():
            batch_out[key] += [sample[key]]
    
    for key, value_to_pad_with in keys_to_pad:

        len_list = list(map(lambda x: len(x), batch_out[key]))

        # padding_length = max(len_list)
        padding_length = 256
        array_list = []
        for i, x in enumerate(batch_out[key]):

            if len(x) < padding_length:
                # Use np.array for creating arrays and np.concatenate for combining them
                padded_array = np.concatenate([np.full((padding_length - len(x)), value_to_pad_with), np.array(x)])
                array_list.append(padded_array)
            else:
                array_list.append(np.array(x)[:padding_length])

        # Use np.stack to stack arrays along a new axis
        batch_out[key] = np.stack(array_list)

    return batch_out

if __name__ == "__main__":

    args = parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.devices
    device_ids = [ int(idx.strip()) for idx in args.devices.split(",") if idx and len(idx.strip()) ]
    total_device_count = len(device_ids)

    ds = load_dataset(
        "arrow",
        data_files=glob.glob(args.data_files),
        num_proc=args.total_procs,
        cache_dir=args.cache_dir,
        split="train",
    )

    data_loader = torch.utils.data.DataLoader(
        ds,
        batch_size=args.batch_size,
        drop_last=True,
        num_workers=8,
        collate_fn=padding_collator,
    )

    model = FlaxIndicTransForConditionalGeneration.from_pretrained(
        "/data/priyam/translation/setu-translate/stages/tpu/flax_weights/200m",
        local_files_only=True,
        dtype=jnp.bfloat16,
    )

    params = replicate(model.params)

    def generate(
        input_ids,
        params,
    ):
        model.params = params
        return model.generate(
            input_ids=input_ids,
            num_beams=1,
            num_return_sequences=1,
            max_length=256,
            do_sample=False,
        )

    p_generate = jax.pmap(generate)

    run_ds = HFDataset.from_dict(
        { key: [] for key in ["doc_id", "sids" "sub_strs", "tlt_idx", "placeholder_entity_map", "translation_ids"] },
    )    

    for idx, batch in tqdm.tqdm(enumerate(data_loader, 0), unit=f"ba: {args.batch_size} samples/ba", total=len(data_loader)):

        input_ids = shard(jnp.array(batch["input_ids"]))

        p_outputs = p_generate(input_ids, params)

        if total_device_count != 1:
            outputs = p_outputs[0].block_until_ready()
            outputs = outputs.reshape(-1, *outputs.shape[2:])
        else:
            outputs = p_outputs[0][0]

        run_ds = concatenate_datasets(
            [
                run_ds,
                HFDataset.from_dict(
                    {
                        "doc_id": batch["input_ids"], 
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

    save_dir = os.path.join(args.base_save_dir, f"devices_{args.devices.replace(',', '_')}")
    os.makedirs(save_dir, exist_ok=True)
    run_ds.save_to_disk(
        save_dir,
        num_proc=args.total_procs,
    )