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

sample_size = 100

# Define a function for each match
def replace_match(match, replacements):
    return replacements[match.group(0)]

def replace_en_to_hi(samples):

    global_str_df = dd.read_csv(f"/data-3/priyam/translation/output/translation/sent_translated_{sample_size}.csv")
    global_str_df = global_str_df.set_index("sid")

    translated = []

    for i in range(len(samples["doc_id"])):
        if not samples["text"][i] or not len(samples["text"][i]):
            translated += [str(None)]
            continue

        replacements = dict()

        sids = json.loads(samples["sids"][i])

        for sid in sids:
            # replacements[f"{{sid::{sid}}}"] = global_str_df.loc[sid, "translated"].compute().iloc[0]
            # print(sid)
            # print(global_str_df.loc[sid].compute())
            replacement_row = global_str_df.loc[sid].compute()
            if len(replacement_row):
                replacements[replacement_row["substr"].iloc[0]] = replacement_row["translated"].iloc[0]
            # print(global_str_df.loc[sid, "translated"].compute().iloc[0])
                # replacements[global_str_df.loc[sid, "substr"].compute().iloc[0]] = global_str_df.loc[sid, "translated"].compute().iloc[0]

        pattern = re.compile('|'.join(re.escape(key) for key in replacements.keys()))

        # Perform the replacements
        translated += [
            pattern.sub(
                partial(replace_match, replacements=replacements),
                # samples["templated_text"][i],
                samples["text"][i],
            )
        ]

    return samples | {
        "translated": translated
    }

if __name__ == "__main__":

    templated_rw = load_dataset(
        "csv",
        data_files=[f"/data-3/priyam/translation/output/translation/template_{sample_size}.csv"],
        cache_dir="/data-3/priyam/translation/refinedweb-mini/cache",
        num_proc=96,
        split="train"
    )

    translated_rw = templated_rw.map(
        replace_en_to_hi,
        batched=True,
        batch_size=1,
        num_proc=96,
    )

    translated_rw.to_csv(f"/data-3/priyam/translation/output/translation/translated_{sample_size}.csv")