import numpy as np
from datasets import load_dataset, Dataset
from functools import partial
import glob
import os
import re
import pandas as pd
import csv
from document import Document
import pickle
import json
import argparse

def parse_args():

    parser = argparse.ArgumentParser(description="Perform reverse on samples for urdu")

    parser.add_argument(
        "--paths_data",
        type=str,
        required=True        
    )

    parser.add_argument(
        "--cache_dir",
        type=str,
        required=True,
    )

    parser.add_argument(
        "--num_procs",
        type=int,
        required=False,
        default=64,
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        required=False,
        default=512,
    )

    parser.add_argument(
        "--reversed_save_path",
        type=str,
        required=True,
    )

    args = parser.parse_args()

    return args

def split_with_delimiter(
    text,
    delimiter_pattern=r'(?<!\d)\.(?!\d)|(?<!\w)\.(?!\w)|[?!।|॥؟۔\n](?:\n+)?', 
):
    lines = re.split(f'({delimiter_pattern})', text)
    if len(lines) % 2 == 0:
        iter_range = range(0, len(lines), 2)
        out = [lines[i]+lines[i+1] for i in iter_range]
    else:
        iter_range = range(0, len(lines) - 1, 2)
        out = [lines[i]+lines[i+1] for i in iter_range] + [lines[-1]]
    return out

def reverse_chunks(samples):

    reversed_docs = []
    doc_texts = samples.pop("text")
    for i, _doc_text in enumerate(doc_texts):

        chunks = [ _chunk for _chunk in _doc_text.split("\n") if _chunk and len(_chunk) ]
        reversed_chunks = [ "".join(split_with_delimiter(_chunk)[::-1]) for _chunk in chunks ]
        _reversed_doc = "\n".join(reversed_chunks)
        reversed_docs += [_reversed_doc]

    return samples | {
        "text": reversed_docs,
        "original_text": doc_texts,
    }


if __name__ == "__main__":

    args = parse_args()

    paths_ds = load_dataset(
        "arrow",
        data_files=glob.glob(args.paths_data),
        cache_dir=args.cache_dir,
        num_proc=args.num_procs,
        split="train",
    )

    reverse_ds = paths_ds.map(
        reverse_chunks,
        batched=True,
        batch_size=args.batch_size,
        num_proc=args.num_procs,
        load_from_cache_file=False,
    )

    os.makedirs(args.reversed_save_path, exist_ok=True)
    reverse_ds.save_to_disk(
        args.reversed_save_path,
        num_proc=args.num_procs,
    )