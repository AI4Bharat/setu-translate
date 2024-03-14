from datasets import load_dataset, Dataset
from functools import partial
import glob
import os
import argparse

def parse_args():

    parser = argparse.ArgumentParser(description="Save substr-level files")

    parser.add_argument(
        "--data_files",
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
        "--format",
        type=str,
        default="arrow",
        required=False
    )
    parser.add_argument(
        "--base_save_dir",
        type=str,
        required=True
    )

    args = parser.parse_args()

    return args

def save_to_str_lvl(batch, base_save_dir=None):
    written_file = []
    for i in range(len(batch["sid"])):
        if base_save_dir:
            file_path = os.path.join(base_save_dir, *batch["tlt_file_loc"][i].split("/")[-2:])
        else:
            file_path = batch["tlt_file_loc"][i]
        
        try:
            file_dir = os.path.dirname(file_path)
            os.makedirs(file_dir, exist_ok=True)
            with open(file_path, "w") as str_f:
                str_f.write(batch["translated"][i])
            written_file += [True]
        except Exception as e:
            written_file += [False]
    return batch | {
        "written": written_file,
    }

if __name__ == "__main__":

    args = parse_args()

    ds = load_dataset(
        args.format,
        data_files=glob.glob(args.data_files),
        cache_dir=args.cache_dir,
        num_proc=args.num_procs,
        split="train",
    )

    updated_ds = ds.map(
        partial(save_to_str_lvl, base_save_dir=args.base_save_dir),
        batched=True,
        batch_size=args.batch_size,
        num_proc=args.num_procs,
    )


