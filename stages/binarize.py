import os
import torch
from IndicTransTokenizer import IndicProcessor, IndicTransTokenizer
from datasets import load_dataset, load_from_disk
from datasets.distributed import split_dataset_by_node

from joblib import Parallel, delayed
import glob
from functools import partial

def binarize(
    batch,
    tokenizer, 
    ip
):
    p_batch = dict()

    sentences = ip.preprocess_batch(
        batch["sub_strs"], 
        src_lang="eng_Latn",
        tgt_lang="hin_Deva"
    )

    p_batch["input_ids"], p_batch["attention_masks"] = tokenizer(
        sentences, 
        src=True, 
        return_tensors="pt",
    ).values()
    
    return {
        "ttl_idx": torch.tensor(batch["ttl_idx"]), 
    } | p_batch

def _mp_fn(
    index,
    total_procs,
    ds,
    binarize_procs,
    binarized_dir,
    batch_size
):

    tokenizer = IndicTransTokenizer(direction="en-indic")
    ip = IndicProcessor(inference=True)

    rank_ds = split_dataset_by_node(
        ds, 
        rank=index,
        world_size=total_procs,
    )

    binarized_ds = rank_ds.map(
        partial(
            binarize,
            tokenizer=tokenizer,
            ip=ip,
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


if __name__ == "__main__":

    root_dir = "/mnt/data/translation"
    data_files=f"{root_dir}/output/wiki_en/sentences"
    cache_dir=f"{root_dir}/data/wiki_en/cache"
    binarized_dir=f"{root_dir}/output/wiki_en/binarized_sentences"
    joblib_temp_folder=f"{root_dir}/tmp"
    batch_size=2048
    total_procs=64
    run_joblib=False

    ds = load_dataset(
        "arrow",
        data_files=glob.glob(f"{data_files}/*.arrow"),
        num_proc=total_procs,
        cache_dir=cache_dir,  
        split="train"
    )

    print("Loaded Tokenizers, IPs and Dataset....")

    if run_joblib:
        
        batch_status = Parallel(
            n_jobs=total_procs,
            verbose=0, 
            prefer="processes",
            batch_size="auto",
            pre_dispatch='n_jobs',
            temp_folder=joblib_temp_folder,
        )(
            delayed(_mp_fn)(
                index=i,
                total_procs=total_procs,
                ds=ds,
                binarize_procs=1,
                binarized_dir=binarized_dir,
                batch_size=batch_size
            ) for i in range(total_procs)
        )

    else:

        tokenizer = IndicTransTokenizer(direction="en-indic")
        ip = IndicProcessor(inference=True)

        binarized_ds = ds.map(
            partial(
                binarize,
                tokenizer=tokenizer,
                ip=ip,
            ),
            batched=True,
            batch_size=batch_size,
            num_proc=total_procs,
        )

        binarized_ds.save_to_disk(
            binarized_dir,
            num_proc=total_procs,
        )

    