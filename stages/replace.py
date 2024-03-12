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

    parser = argparse.ArgumentParser(description="Perform replace on samples")

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
        "--batch_size",
        type=int,
        required=False,
        default=64,
    )

    parser.add_argument(
        "--num_procs",
        type=int,
        required=False,
        default=512,
    )

    parser.add_argument(
        "--decode_base_path",
        type=str,
        required=False,
        default=None,
    )

    parser.add_argument(
        "--translated_save_path",
        type=str,
        required=True,
    )

    args = parser.parse_args()

    return args

# Define a function for each match
def replace_match(match, replacements):
    return replacements.get(match.group(0), match.group(0))

def replace_translated(samples, decode_base_path=None):

    translated = []
    sub_strs_tlt = []

    for i in range(len(samples["doc_id"])):
        if not samples["text"][i] or not len(samples["text"][i]):
            translated += [str(None)]
            sub_strs_tlt += [[str(None)]]
            continue

        replacements = dict()

        sids = json.loads(samples["sids"][i])

        sub_tlts = []

        for s_idx, sid in enumerate(sids):

            sid_tlt_file_path = os.path.join(samples["tlt_folder"][i], sid)

            sub_strs = json.loads(samples["sub_strs"][i])

            if decode_base_path:
                sid_tlt_file_path = os.path.join(decode_base_path, *sid_tlt_file_path.split("/")[-2:])

            if not os.path.exists(sid_tlt_file_path):
                continue

            with open(sid_tlt_file_path, "r") as tlt_f:
                tlt_sub = tlt_f.read()

            replacements[sub_strs[s_idx]] = tlt_sub

            sub_tlts += [tlt_sub]

        pattern = re.compile('|'.join(re.escape(key) for key in replacements.keys()))

        # Perform the replacements
        translated += [
            pattern.sub(
                partial(replace_match, replacements=replacements),
                samples["text"][i],
            )
        ]

        sub_strs_tlt += [sub_tlts if len(sub_tlts) else [str(None)]]

    return samples | {
        "translated": translated,
        "substr_tlt": sub_strs_tlt,
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

    replace_ds = paths_ds.map(
        partial(replace_translated, decode_base_path=args.decode_base_path),
        batched=True,
        batch_size=args.batch_size,
        num_proc=args.num_procs,
        load_from_cache_file=False,
    )

    os.makedirs(args.translated_save_path, exist_ok=True)
    replace_ds.save_to_disk(
        args.translated_save_path,
        num_proc=args.num_procs,
    )