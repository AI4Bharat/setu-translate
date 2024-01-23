import os
import torch
from IndicTransTokenizer import IndicProcessor, IndicTransTokenizer
from datasets import load_dataset, load_from_disk
from datasets.distributed import split_dataset_by_node

from joblib import Parallel, delayed
import glob
from functools import partial

import json

import argparse

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def parse_args():

    parser = argparse.ArgumentParser(description="Perform binarization")

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
        "--binarized_dir",
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
        "--run_joblib",
        type=str2bool
    )

    parser.add_argument(
        "--src_lang",
        type=str,
    )

    parser.add_argument(
        "--tgt_lang",
        type=str,
    )

    args = parser.parse_args()

    return args

def binarize(
    batch,
    tokenizer, 
    src_lang="eng_Latn",
    tgt_lang="hin_Deva"
):
    p_batch = dict()

    ip = IndicProcessor(inference=True)

    sentences = ip.preprocess_batch(
        batch["sub_strs"], 
        src_lang=src_lang,
        tgt_lang=tgt_lang
    )

    placeholder_entity_maps = list(map(lambda ple_map: json.dumps(ple_map), ip.get_placeholder_entity_maps(clear_ple_maps=True)))

    p_batch["input_ids"], p_batch["attention_masks"] = tokenizer(
        sentences, 
        src=True, 
        return_tensors="pt",
    ).values()

    return {
        "tlt_idx": torch.tensor(batch["tlt_idx"]), 
    } | p_batch | {
        "placeholder_entity_map": placeholder_entity_maps,
    }

def _mp_fn(
    index,
    total_procs,
    ds,
    binarize_procs,
    binarized_dir,
    batch_size,
    src_lang,
    tgt_lang,
):

    tokenizer = IndicTransTokenizer(direction="en-indic")

    rank_ds = split_dataset_by_node(
        ds, 
        rank=index,
        world_size=total_procs,
    )

    binarized_ds = rank_ds.map(
        partial(
            binarize,
            tokenizer=tokenizer,
            src_lang=src_lang,
            tgt_lang=tgt_lang,
        ),
        batched=True,
        batch_size=batch_size,
        num_proc=binarize_procs,
    )

    save_dir = os.path.join(binarized_dir, f"{index}")
    os.makedirs(save_dir, ok_exist=True)
    binarized_ds.save_to_disk(
        save_dir,
        num_proc=binarize_procs,
    )

    return True


if __name__ == "__main__":

    args = parse_args()

    ds = load_dataset(
        "arrow",
        data_files=glob.glob(args.data_files),
        num_proc=args.total_procs,
        cache_dir=args.cache_dir,  
        split="train"
    )

    print("Loaded Dataset....")

    if args.run_joblib:
        
        batch_status = Parallel(
            n_jobs=args.total_procs,
            verbose=0, 
            prefer="processes",
            batch_size="auto",
            pre_dispatch='n_jobs',
            temp_folder=args.joblib_temp_folder,
        )(
            delayed(_mp_fn)(
                index=i,
                total_procs=args.total_procs,
                ds=ds,
                binarize_procs=1,
                binarized_dir=args.binarized_dir,
                batch_size=args.batch_size,
                src_lang=args.src_lang,
                tgt_lang=args.tgt_lang,
            ) for i in range(args.total_procs)
        )

    else:

        tokenizer = IndicTransTokenizer(direction="en-indic")

        print("Loaded Tokenizer and IP ....")

        binarized_ds = ds.map(
            partial(
                binarize,
                tokenizer=tokenizer,
                src_lang=args.src_lang,
                tgt_lang=args.tgt_lang,
            ),
            batched=True,
            batch_size=args.batch_size,
            num_proc=args.total_procs,
        )

        os.makedirs(args.binarized_dir, exist_ok=True)

        binarized_ds.save_to_disk(
            args.binarized_dir,
            num_proc=args.total_procs,
        )

    