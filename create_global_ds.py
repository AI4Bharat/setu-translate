import numpy as np
from datasets import load_dataset, Dataset
from functools import partial
import glob
import os
import re
import pandas as pd
import csv
from document import Document
import dask.dataframe as dd
import pickle
import json
import argparse

def parse_args():

    parser = argparse.ArgumentParser(description="Creating a global sentence dataset")

    parser.add_argument(
        "--templated_csv_path",
        type=str,
        required=True
    )

    parser.add_argument(
        "--cache_dir",
        type=str,
        required=True,
    )

    parser.add_argument(
        "--hf_out_csv",
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

def flatten_sentence(samples):
    substrs = []
    sids = []
    for i in range(len(samples["doc_id"])):
        if samples["sub_strs"][i] and samples["sids"][i]:
            substrs += json.loads(samples["sub_strs"][i])
            sids += json.loads(samples["sids"][i])
    return {
        "sid": sids,
        "substr": substrs,
    }
    
if __name__ == "__main__":

    args = parse_args()

    templated_rw = load_dataset(
        "csv",
        data_files=[args.templated_csv_path],
        cache_dir=args.cache_dir,
        num_proc=96,
        split="train")

    sentence_rw = templated_rw.map(
        flatten_sentence,
        batched=True,
        batch_size=256,
        num_proc=96,
        remove_columns=templated_rw.features
    )

    # "/data-3/priyam/translation/output/translation/global_sentences.csv"
    sentence_rw.to_csv(args.hf_out_csv)

    df = dd.read_csv(args.hf_out_csv)
    print("Total Sentences in the dataset: ", len(df))
    df = df.drop_duplicates(subset=["sid"])
    print("Total Sentences in the dataset after deduplication: ", len(df))
    # "/data-3/priyam/translation/output/translation/deduped_global_sentences.csv"
    df.to_csv(args.global_sent_ds_path, index=False)





    