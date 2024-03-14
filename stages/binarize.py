import os
import torch
from IndicTransTokenizer import IndicProcessor, IndicTransTokenizer
from datasets import load_dataset

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
        "--format",
        type=str,
        default="arrow",
        required=False,
    )

    parser.add_argument(
        "--binarized_dir",
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
        "--padding",
        choices=["longest", "max_length"],
        type=str,
        default="longest",
        required=False
    )

    parser.add_argument(
        "--src_lang",
        type=str,
    )

    parser.add_argument(
        "--tgt_lang",
        type=str,
    )

    parser.add_argument(
        "--return_format",
        choices=["np", "pt"],
        type=str,
        default="np",
        required=False
    )

    args = parser.parse_args()

    return args

def binarize(
    batch,
    padding="longest",
    src_lang="eng_Latn",
    tgt_lang="hin_Deva",
    return_format="np",
):
    p_batch = dict()

    ip = IndicProcessor(inference=True)
    tokenizer = IndicTransTokenizer(direction="en-indic")

    sentences = ip.preprocess_batch(
        batch["sub_strs"], 
        src_lang=src_lang,
        tgt_lang=tgt_lang
    )
    
    placeholder_entity_maps = list(map(lambda ple_map: json.dumps(ple_map), ip.get_placeholder_entity_maps(clear_ple_maps=True)))

    p_batch["input_ids"], p_batch["attention_mask"] = tokenizer(
        sentences, 
        src=True, 
        padding=padding,
        truncation=True if padding == "max_length" else False,
        return_tensors=return_format,
    ).values()

    return {
        "tlt_idx": torch.tensor(batch["tlt_idx"]), 
    } | p_batch | {
        "placeholder_entity_map": placeholder_entity_maps,
    }


if __name__ == "__main__":

    args = parse_args()

    ds = load_dataset(
        args.format,
        data_files=glob.glob(args.data_files),
        num_proc=args.total_procs,
        cache_dir=args.cache_dir,  
        split="train"
    )

    print("Loaded Dataset....")

    binarized_ds = ds.map(
        partial(
            binarize,
            padding=args.padding,
            src_lang=args.src_lang,
            tgt_lang=args.tgt_lang,
            return_format=args.return_format,
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

    