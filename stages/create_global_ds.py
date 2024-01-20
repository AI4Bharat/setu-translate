from datasets import load_dataset
import glob
import pandas as pd
import argparse
import json

def parse_args():

    parser = argparse.ArgumentParser(description="Creating a global sentence dataset")

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
        "--global_sent_ds_path",
        type=str,
        required=True,
    )

    args = parser.parse_args()

    return args

def flatten_substr(
    samples,
    keys=["doc_id", "sids", "sub_strs"],
):
    out = {key: [] for key in keys}
    for i in range(len(samples["doc_id"])):

        if not samples["sub_strs"][i] or not len(samples["sub_strs"][i]):
            continue

        sids = json.loads(samples["sids"][i])
        sub_strs = json.loads(samples["sub_strs"][i])

        substr_idx = []

        for idx, substr in enumerate(sub_strs):
            if not isinstance(substr, str) or not len(substr.strip()):
                continue
            substr_idx += [idx]
        
        out["doc_id"] += [samples["doc_id"][i]] * len(substr_idx)

        for key, value in [("sids", sids), ("sub_strs", sub_strs)]:
            for idx in substr_idx:
                out[key] += [value[idx]]

    return out
    
if __name__ == "__main__":

    args = parse_args()

    paths_ds = load_dataset(
        "arrow",
        data_files=[args.paths_data],
        cache_dir=args.cache_dir,
        num_proc=64,
        split="train"
    )

    sentence_ds = paths_ds.map(
        flatten_substr,
        batched=True,
        batch_size=256,
        num_proc=64,
        remove_columns=paths_ds.features,
    )

    sentence_ds = sentence_ds.map(
        lambda samples, idx: samples | { "tlt_idx": idx },
        batched=True,
        batch_size=256,
        num_proc=64,
        with_indices=True
    )

    sentence_ds.save_to_disk(
        args.global_sent_ds_path,
        num_proc=64,
    )

