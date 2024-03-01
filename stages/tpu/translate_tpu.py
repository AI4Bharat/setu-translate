import os
import jax.numpy as jnp
from joblib import Parallel, delayed
from joblib.externals.loky.backend.context import get_context

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
        "--joblib_temp_folder",
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
        type=lambda x: [ int(idx.strip()) for idx in x.split(",") if idx and len(idx.strip()) ],
        required=True,
    )

    args = parser.parse_args()

    return args

def padding_collator_jax(
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

        padding_length = max(len_list)
        array_list = []
        for i, x in enumerate(batch_out[key]):

            if len(x) < padding_length:
                # Use jnp.array for creating arrays and jnp.concatenate for combining them
                padded_array = jnp.concatenate([jnp.full((padding_length - len(x)), value_to_pad_with), jnp.array(x)])
                array_list.append(padded_array)
            else:
                array_list.append(jnp.array(x))

        # Use jnp.stack to stack arrays along a new axis
        batch_out[key] = jnp.stack(array_list)

    return batch_out


def _mp_fn(
    ds,
    base_save_dir,
    batch_size,
    rank,
    device,
    world_size,
    procs_to_write,
):

    device = f"cuda:{device}"
    world_size = world_size
    rank = rank

    rank_ds = split_dataset_by_node(
        ds, 
        rank=rank,
        world_size=world_size,
    )

    model = FlaxIndicTransForConditionalGeneration.from_pretrained(
        "/data/priyam/translation/setu-translate/stages/tpu/flax/200m",
        local_files_only=True,
        dtype=jnp.bfloat16,
    )

    run_ds = HFDataset.from_dict(
        { key: [] for key in ["doc_id", "sids" "sub_strs", "tlt_idx", "placeholder_entity_map", "translation_ids"] },
    )

    with torch.no_grad():

        for idx, batch in tqdm.tqdm(enumerate(data_loader, 0), unit=f"ba: {batch_size} samples/ba", total=len(data_loader)):

            input_ids = batch["input_ids"].to(device)

            outputs = model.generate(
                input_ids=input_ids,
                num_beams=1,
                num_return_sequences=1,
                max_length=256,
                do_sample=False,
            )

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
                            "translated_input_ids": outputs.to("cpu"),
                            "tlt_file_loc": batch["tlt_file_loc"],
                        }
                    ),
                ], 
            )

    save_dir = os.path.join(base_save_dir, f"rank_{rank}-device_{device}")
    os.makedirs(save_dir, exist_ok=True)
    run_ds.save_to_disk(
        save_dir,
        num_proc=procs_to_write,
    )
    return True

if __name__ == "__main__":

    args = parse_args()

    ds = load_dataset(
        "arrow",
        data_files=glob.glob(args.data_files),
        num_proc=args.total_procs,
        cache_dir=args.cache_dir,
        split="train",
    )

    batch_status = Parallel(
        n_jobs=len(args.devices),
        verbose=0, 
        backend="loky",
        batch_size="auto",
        pre_dispatch='n_jobs',
        temp_folder=args.joblib_temp_folder,
    )(
        delayed(_mp_fn)(
            ds=ds,
            base_save_dir=args.base_save_dir,
            batch_size=args.batch_size,
            rank=idx,
            device=device,
            world_size=len(args.devices),
            procs_to_write=args.total_procs//len(args.devices)
        )  for idx, device in enumerate(args.devices)
    )

    
