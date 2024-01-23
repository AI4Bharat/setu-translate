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
        "--decode_dir",
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

def decode(
    batch,
    tokenizer, 
    src_lang="eng_Latn",
    tgt_lang="hin_Deva"
):
    
    ip = IndicProcessor(inference=True)
    
    p_batch = dict()
    input_ids = batch.pop("translated_input_ids")
    placeholder_entity_maps = list(map(lambda ple_map: json.loads(ple_map), batch["placeholder_entity_map"]))
    outputs = tokenizer.batch_decode(input_ids, src=False)
    p_batch["translated"] = ip.postprocess_batch(outputs, lang=tgt_lang, placeholder_entity_maps=placeholder_entity_maps)
    return p_batch

def save_to_str_lvl(batch):
    written_file = []
    for i in range(len(batch["sid"])):
        try:
            file_dir = os.path.dirname(batch["tlt_file_loc"][i])
            os.makedirs(file_dir, exist_ok=True)
            with open(batch["tlt_file_loc"][i], "w") as str_f:
                str_f.write(batch["translated"][i])
            written_file += [True]
        except Exception as e:
            written_file += [False]
    return batch | {
        "written": written_file
    }

def _mp_fn(
    index,
    total_procs,
    ds,
    decode_procs,
    decode_dir,
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

    decoded_ds = rank_ds.map(
        partial(
            decode,
            tokenizer=tokenizer,
            src_lang=src_lang,
            tgt_lang=tgt_lang,
        ),
        batched=True,
        batch_size=batch_size,
        num_proc=decode_procs,
        remove_columns=["translated_input_ids", "placeholder_entity_map"]
    )

    decoded_ds = decoded_ds.map(
        save_to_str_lvl,
        batched=True,
        batch_size=batch_size,
        num_proc=decode_procs,
    )

    save_dir = os.path.join(decode_dir, f"{index}")
    os.makedirs(save_dir, ok_exist=True)
    decoded_ds.save_to_disk(
        save_dir,
        num_proc=decode_procs,
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

        decoded_ds = ds.map(
            partial(
                decode,
                tokenizer=tokenizer,
                src_lang=args.src_lang,
                tgt_lang=args.tgt_lang,
            ),
            batched=True,
            batch_size=args.batch_size,
            num_proc=args.total_procs,
            remove_columns=["translated_input_ids", "placeholder_entity_map"]
        )

        decoded_ds = decoded_ds.map(
            save_to_str_lvl,
            batched=True,
            batch_size=args.batch_size,
            num_proc=args.total_procs,
        )

        os.makedirs(args.decode_dir, exist_ok=True)

        decoded_ds.save_to_disk(
            args.decode_dir,
            num_proc=args.total_procs,
        )

    