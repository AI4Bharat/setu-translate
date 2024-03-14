import os
import torch
from torch.utils.data import DataLoader
import jax
import jax.numpy as jnp
import numpy as np

from IndicTransTokenizer import IndicProcessor, IndicTransTokenizer

from flax.jax_utils import replicate
from flax.training.common_utils import shard

# from transformers import AutoModelForSeq2SeqLM
from modeling_flax_indictrans import FlaxIndicTransForConditionalGeneration


from datasets import (
    Dataset as HFDataset, 
    load_from_disk,
    load_dataset,
    concatenate_datasets
)

from datasets.distributed import split_dataset_by_node

import glob
from functools import partial

import tqdm
import argparse

import json

from jax_smi import initialise_tracking
initialise_tracking()

def parse_args():

    parser = argparse.ArgumentParser(description="Evaluate Metrics")

    parser.add_argument(
        "--base_save_dir",
        type=str,
    )

    parser.add_argument(
        "--setu_translate_root",
        type=str,
    )

    parser.add_argument(
        "--direction",
        type=str,
        choices=["en-indic", "indic-en", "indic-indic"],
        default="en-indic",
        required=False    
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
        "--devices",
        type=str,
        required=True,
    )

    args = parser.parse_args()

    return args

def binarize(
    batch,
    text_col="eng_latn",
    padding="longest",
    src_lang="eng_Latn",
    tgt_lang="hin_Deva",
    return_format="np",
    direction="en-indic",
):
    p_batch = dict()

    ip = IndicProcessor(inference=True)
    tokenizer = IndicTransTokenizer(direction=direction)

    sentences = ip.preprocess_batch(
        batch[text_col], 
        src_lang=src_lang,
        tgt_lang=tgt_lang
    )

    placeholder_entity_maps = list(map(lambda ple_map: json.dumps(ple_map), ip.get_placeholder_entity_maps(clear_ple_maps=True)))

    p_batch[f"input_ids-{src_lang}__{tgt_lang}"], p_batch[f"attention_mask-{src_lang}__{tgt_lang}"] = tokenizer(
        sentences, 
        src=True, 
        padding=padding,
        truncation=True if padding == "max_length" else False,
        return_tensors=return_format,
    ).values()

    return p_batch | {
        f"pem-{src_lang}__{tgt_lang}": placeholder_entity_maps,
    }

def decode(
    batch,
    src_lang="eng_Latn",
    tgt_lang="hin_Deva",
    direction="en-indic"
):
    
    ip = IndicProcessor(inference=True)
    tokenizer = IndicTransTokenizer(direction=direction)

    p_batch = dict()
    input_ids = batch.pop(f"tlt_input_ids-{src_lang}__{tgt_lang}")
    placeholder_entity_maps = list(map(lambda ple_map: json.loads(ple_map), batch[f"pem-{src_lang}__{tgt_lang}"]))
    outputs = tokenizer.batch_decode(input_ids, src=False)
    p_batch[f"tlt-{src_lang}__{tgt_lang}"] = ip.postprocess_batch(outputs, lang=tgt_lang, placeholder_entity_maps=placeholder_entity_maps)
    return p_batch | {
        f"tlt_input_ids-{src_lang}__{tgt_lang}": input_ids,
    }


if __name__ == "__main__":

    args = parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.devices
    device_ids = [ int(idx.strip()) for idx in args.devices.split(",") if idx and len(idx.strip()) ]
    total_device_count = len(device_ids)

    indic_languages = [
        "asm_Beng",
        "ben_Beng",
        "brx_Deva",
        "doi_Deva",
        "gom_Deva",
        "guj_Gujr",
        "hin_Deva",
        "kan_Knda",
        "kas_Arab",
        "mai_Deva",
        "mal_Mlym",
        "mar_Deva",
        "mni_Mtei",
        "npi_Deva",
        "ory_Orya",
        "pan_Guru",
        "san_Deva",
        "sat_Olck",
        "snd_Deva",
        "tam_Taml",
        "tel_Telu",
        "urd_Arab"
    ]

    eng_script = "eng_Latn"

    def prepare_dataloader(ds):
        return torch.utils.data.DataLoader(
            ds,
            batch_size=args.batch_size,
            drop_last=False if total_device_count == 1 else True,
            num_workers=8,
        )

    datasets_for_evaluation = {
        "gen": prepare_dataloader(load_dataset("ai4bharat/IN22-Gen", "all", split="gen")),
        "conv": prepare_dataloader(load_dataset("ai4bharat/IN22-Conv", "all", split="conv"))
    }

    print("Loaded the datasets....")

    model = FlaxIndicTransForConditionalGeneration.from_pretrained(
        os.path.join(args.setu_translate_root, "stages/tpu/flax_weights/200m"),
        local_files_only=True,
        dtype=jnp.bfloat16,
    )

    params = replicate(model.params)
    print("Loaded the flax port of IndicTrans2....")

    def generate(
        batch,
        params,
    ):
        model.params = params
        return model.generate(
            **batch,
            num_beams=1,
            num_return_sequences=1,
            max_length=256,
            do_sample=False,
        ).sequences

    p_generate = jax.pmap(generate)

    def evaluate(
        data_loader,
        src_lang,
        tgt_lang,
        batch_size,
        params,
        total_device_count,
        base_save_dir,
        direction,
        total_procs,
        ds_type,
    ):
        run_ds = HFDataset.from_dict(
            { key: [] for key in ["id", f"sentence_{src_lang}", f"sentence_{tgt_lang}", f"tlt-{src_lang}__{tgt_lang}"] },
        )
        for idx, batch in tqdm.tqdm(enumerate(data_loader, 0), unit=f"ba: {batch_size} samples/ba", total=len(data_loader)):

            binarized_batch = binarize(
                batch,
                text_col=f"sentence_{src_lang}",
                padding="max_length",
                src_lang=src_lang,
                tgt_lang=tgt_lang,
                return_format="np",
                direction=direction
            )

            input_batch = {
                "input_ids": shard(jnp.array(binarized_batch[f"input_ids-{src_lang}__{tgt_lang}"])),
                "attention_mask": shard(jnp.array(binarized_batch[f"attention_mask-{src_lang}__{tgt_lang}"]))
            }

            outputs = p_generate(input_batch, params)

            outputs = outputs.block_until_ready()

            if total_device_count != 1:
                outputs = outputs.reshape(-1, *outputs.shape[2:])
            else:
                outputs = outputs[0]

            decode_batch = {
                f"tlt_input_ids-{src_lang}__{tgt_lang}": np.asarray(outputs),
                f"pem-{src_lang}__{tgt_lang}": binarized_batch[f"pem-{src_lang}__{tgt_lang}"]
            }

            decode_outputs = decode(decode_batch, src_lang, tgt_lang, direction=direction)

            run_ds = concatenate_datasets(
                [
                    run_ds,
                    HFDataset.from_dict(
                        {
                            "id": batch["id"],
                            f"sentence_{src_lang}": batch[f"sentence_{src_lang}"], 
                            f"sentence_{tgt_lang}": batch[f"sentence_{tgt_lang}"],
                            f"tlt-{src_lang}__{tgt_lang}": decode_outputs[f"tlt-{src_lang}__{tgt_lang}"],
                            f"tlt_input_ids-{src_lang}__{tgt_lang}": decode_outputs[f"tlt_input_ids-{src_lang}__{tgt_lang}"]
                        }
                    ),
                ],
            )

        save_dir = os.path.join(base_save_dir, ds_type, direction, f"{src_lang}__{tgt_lang}")
        os.makedirs(save_dir, exist_ok=True)
        run_ds.save_to_disk(
            save_dir,
            num_proc=total_procs,
        )

    if args.direction == "en-indic":
        src_lang = eng_script
        print(f"Evaluation Direction: {args.direction}")
        for i, tgt_lang in enumerate(indic_languages):
            print(f"Performing evaluation FOR: {src_lang} -> {tgt_lang}")
            for ds_type, data_loader in datasets_for_evaluation.items():
                print(f"Performing evaluation ON: {ds_type}")
                evaluate(
                    data_loader=data_loader,
                    src_lang=src_lang,
                    tgt_lang=tgt_lang,
                    batch_size=args.batch_size,
                    params=params,
                    total_device_count=total_device_count,
                    base_save_dir=args.base_save_dir,
                    direction=args.direction,
                    total_procs=args.total_procs,
                    ds_type=ds_type,
                )

    elif args.direction == "indic-en":
        tgt_lang = eng_script
        print(f"Evaluation Direction: {args.direction}")
        for i, src_lang in enumerate(indic_languages):
            print(f"Performing evaluation FOR: {src_lang} -> {tgt_lang}")
            for ds_type, data_loader in datasets_for_evaluation.items():
                print(f"Performing evaluation ON: {ds_type}")
                evaluate(
                    data_loader=data_loader,
                    src_lang=src_lang,
                    tgt_lang=tgt_lang,
                    batch_size=args.batch_size,
                    params=params,
                    total_device_count=total_device_count,
                    base_save_dir=args.base_save_dir,
                    direction=args.direction,
                    total_procs=args.total_procs,
                    ds_type=ds_type,
                )

            
    elif args.direction == "indic-indic":
        print(f"Evaluation Direction: {args.direction}")
        for i, src_lang in enumerate(indic_languages):
            print(f"Performing evaluation FOR: {src_lang} -> {tgt_lang}")
            for j, tgt_lang in enumerate(indic_languages):
                print(f"Performing evaluation ON: {ds_type}")
                if src_lang == tgt_lang:
                    print("Same `src_lang` and `tgt_lang`, so, skipping......")
                    continue
                for ds_type, data_loader in datasets_for_evaluation.items():
                    evaluate(
                        data_loader=data_loader,
                        src_lang=src_lang,
                        tgt_lang=tgt_lang,
                        batch_size=args.batch_size,
                        params=params,
                        total_device_count=total_device_count,
                        base_save_dir=args.base_save_dir,
                        direction=args.direction,
                        total_procs=args.total_procs,
                        ds_type=ds_type,
                    )
                