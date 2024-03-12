import os
os.environ["XLA_USE_BF16"]='1'

import torch
from torch.utils.data import DataLoader
import torch.distributed as dist
import torch_xla
import torch_xla.core.xla_model as xm
import torch_xla.distributed.parallel_loader as pl
import torch_xla.distributed.xla_multiprocessing as xmp
import torch_xla.distributed.xla_backend

from transformers import AutoModelForSeq2SeqLM
from datasets import (
    Dataset as HFDataset, 
    load_from_disk,
    load_dataset
)
from datasets.distributed import split_dataset_by_node

import glob
from functools import partial


def padding_collator(
    batch, 
    keys_to_pad=[
            ("input_ids", 1), 
            ("attention_masks", 0),
        ]
    ):

    batch_out = {key: [] for key in batch[0].keys()}
    
    for sample in batch:
        for key in batch_out.keys():
            batch_out[key] += [sample[key]]
    
    for key, value_to_pad_with in keys_to_pad:

        len_list = list(map(lambda x: len(x), batch_out[key]))

        padding_length = max(len_list)
        # padding_length = 256
        tensor_list = []
        for i, x in enumerate(batch_out[key]):

            if len(x) < padding_length:
                tensor_list += [torch.tensor([value_to_pad_with]*(padding_length - len_list[i]) + x)]
            else:
                tensor_list += [torch.tensor(x)]

        batch_out[key] = torch.stack(tensor_list)

    return batch_out


def _mp_fn(
    index,  
    ds,
    root_dir,
    data_files,
    cache_dir,
    base_save_dir,
    batch_size,
    on_gpu,
):

    device = xm.xla_device()
    world_size = xm.xrt_world_size()
    rank = xm.get_ordinal()

    dist.init_process_group("xla", init_method='xla://')

    rank_ds = split_dataset_by_node(
        ds, 
        rank=rank,
        world_size=world_size,
    )

    data_loader = torch.utils.data.DataLoader(
        rank_ds,
        batch_size=batch_size,
        drop_last=False,
        num_workers=8,
        collate_fn=padding_collator
    )

    model = AutoModelForSeq2SeqLM.from_pretrained(
        "ai4bharat/indictrans2-en-indic-dist-200M", 
        trust_remote_code=True
    )

    model = model.eval().to(device)

    dist.barrier()

    if rank == 0:
        print("Loaded Model on the TPUs and prepared Data... Starting inference")

    # output_ds = HFDataset.from_dict(
    #     { key: [] for key in [""] },
    # )

    with torch.no_grad():

        for i, batch in enumerate(data_loader):

            if rank == 0:
                print(batch)

            input_ids = batch["input_ids"].to(device)

            if rank == 0:
                print(f"Data sent to {device}")

            outputs = model.generate(
                input_ids=input_ids,
                num_beams=1,
                num_return_sequences=1,
                max_length=256
            )

            if rank == 0:
                print(outputs.to("cpu"))
            # print(outputs)
            # outputs = tokenizer.batch_decode(outputs, src=False)
            # outputs = ip.postprocess_batch(outputs, lang="hin_Deva")

            break


if __name__ == "__main__":

    root_dir = "/data-3/priyam/translation"
    data_files=f"{root_dir}/output/wiki_en/binarized_sentences"
    cache_dir=f"{root_dir}/data/wiki_en/cache"
    base_save_dir=f"{root_dir}/data/wiki_en/model_out"
    batch_size=8
    on_gpu=False

    ds = load_dataset(
        "arrow",
        data_files=glob.glob(f"{data_files}/*.arrow"),
        num_proc=64,
        cache_dir=cache_dir,
        split="train",
    )

    args = (
        ds,
        root_dir,
        data_files,
        cache_dir,
        base_save_dir,
        batch_size,
        on_gpu,
    )

    xmp.spawn(
        _mp_fn, 
        args=args,
        nprocs=1, 
        start_method='fork'
    )
