import os

import torch
from torch.utils.data import DataLoader

from joblib import Parallel, delayed
from joblib.externals.loky.backend.context import get_context

from transformers import AutoModelForSeq2SeqLM

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

        padding_length = max(len_list)
        tensor_list = []
        for i, x in enumerate(batch_out[key]):

            if len(x) < padding_length:
                tensor_list += [torch.tensor([value_to_pad_with]*(padding_length - len_list[i]) + x)]
            else:
                tensor_list += [torch.tensor(x)]

        batch_out[key] = torch.stack(tensor_list)

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

    data_loader = torch.utils.data.DataLoader(
        rank_ds,
        batch_size=batch_size,
        drop_last=False,
        num_workers=8,
        collate_fn=padding_collator,
        multiprocessing_context=get_context('loky')
    )

    model = AutoModelForSeq2SeqLM.from_pretrained(
        "ai4bharat/indictrans2-en-indic-dist-200M", 
        trust_remote_code=True
    )

    model = model.eval().to(device)

    run_ds = HFDataset.from_dict(
        { key: [] for key in ["doc_id", "sids" "sub_strs", "tlt_idx", "translation_ids"] },
    )

    with torch.no_grad():

        for idx, batch in tqdm.tqdm(enumerate(data_loader, 0), unit=f"ba: {batch_size} samples/ba", total=len(data_loader)):

            input_ids = batch["input_ids"].to(device)

            outputs = model.generate(
                input_ids=input_ids,
                num_beams=1,
                num_return_sequences=1,
                max_length=256
            )

            run_ds = concatenate_datasets(
                [
                    run_ds, 
                    HFDataset.from_dict(
                        {
                            "doc_id": batch["input_ids"], 
                            "sids": batch["sids"], 
                            "sub_strs": batch["sub_strs"], 
                            "tlt_idx": batch["tlt_idx"], 
                            "translation_ids": outputs.to("cpu"),
                        }
                    ),
                ], 
            )

    save_dir = os.path.join(base_save_dir, f"rank_{rank}-device_{device}")
    os.makedirs(save_dir, exist_ok=True)
    output_ds.save_to_disk(
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

    
