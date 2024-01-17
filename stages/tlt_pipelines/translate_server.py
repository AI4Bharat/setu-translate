'''
Usage:  
- Do `pip install tritonclient[all] gevent` first.
- Then use this sample script with the appropriate Triton-server endpoint_url
'''

import tritonclient.http as http_client
from tritonclient.utils import *
import numpy as np
from datasets import load_dataset, Dataset
from functools import partial
import glob
import os
import re
import pandas as pd
import csv
from .document import Document
import argparse
import json
import os
from datasets.distributed import split_dataset_by_node
import sys

def cleanup():
    print("Performing cleanup operations...")
    # Add your cleanup code here

class TritonTranslator:

    def __init__(self, endpoint_url='localhost:8000', enable_ssl=False, api_key="__PASTE_KEY_HERE__") -> None:
        self.endpoint_url = endpoint_url
        self.enable_ssl = enable_ssl
        self.http_headers = {"Authorization": f"Bearer {api_key}"}

        # Connect to the server
        if self.enable_ssl:
            import gevent.ssl
            self.triton_http_client = http_client.InferenceServerClient(
                url=self.endpoint_url, verbose=False,
                ssl=True, ssl_context_factory=gevent.ssl._create_default_https_context,
            )
        else:
            self.triton_http_client = http_client.InferenceServerClient(
                url=self.endpoint_url, verbose=False,
            )
    
    def get_string_tensor(self, string_values, tensor_name):
        string_obj = np.array(string_values, dtype="object")
        input_obj = http_client.InferInput(tensor_name, string_obj.shape, np_to_triton_dtype(string_obj.dtype))
        input_obj.set_data_from_numpy(string_obj)
        return input_obj

    def get_translation_input_for_triton(self, texts: list, src_lang: str, tgt_lang: str):
        return [
            self.get_string_tensor([[text] for text in texts], "INPUT_TEXT"),
            self.get_string_tensor([[src_lang]] * len(texts), "INPUT_LANGUAGE_ID"),
            self.get_string_tensor([[tgt_lang]] * len(texts), "OUTPUT_LANGUAGE_ID"),
        ]

    def translate(self, input_sentences, src_lang, tgt_lang):
        inputs = self.get_translation_input_for_triton(input_sentences, src_lang, tgt_lang)
        output0 = http_client.InferRequestedOutput("OUTPUT_TEXT")
        response = self.triton_http_client.infer(
            "nmt",
            model_version='1',
            inputs=inputs,
            outputs=[output0],
            headers=self.http_headers,
        )
        output_batch = response.as_numpy('OUTPUT_TEXT').tolist()
        return [translation[0].decode("utf-8") for translation in output_batch]

PROC_COUNT = 8
TltTranslatorsSet = [TritonTranslator('0.0.0.0:8000') for i in range(PROC_COUNT)]

def translate(
    samples,
    rank,
    src_lang,
    tgt_lang,
):

    translator = TltTranslatorsSet[rank]
    strs_to_translate = []
    idx_to_translate = []
    translated = []
    
    for i in range(len(samples["substr"])):
        translated += [str(None)]
        if samples["substr"][i] and len(samples["substr"][i].strip()):
            strs_to_translate += [samples["substr"][i]]
            idx_to_translate += [i]
        
    if not len(strs_to_translate):
        return samples | {
            "translated": translated
        }

    translated_out = translator.translate(strs_to_translate, src_lang, tgt_lang)

    for i, idx in enumerate(idx_to_translate):
        translated[idx] = translated_out[i]

    return samples | {
        "translated": translated
    }

# def translate(
#     samples,
#     rank,
#     src_lang,
#     tgt_lang,
# ):

#     translator = TltTranslatorsSet[rank]
#     translated = []
    
#     for i in range(len(samples["substr"])):
#         if not samples["substr"][i] or not len(samples["substr"][i].strip()):
#             translated += [str(None)]
#             continue

#         translated += [translator.translate(samples["substr"][i], src_lang, tgt_lang)]

#     return samples | {
#         "translated": translated
#     }

def read_csvs(
    samples,
):
    df = pd.DataFrame(columns=["doc_id", "sid", "substr"])
    for i in range(len(samples["csv_path"])):
        mini_df = pd.read_csv(samples["csv_path"][i])
        df = pd.concat([df, mini_df], ignore_index=True)
    return df.to_dict("list")

def flatten_sentence(samples):
    substrs = []
    sids = []
    doc_ids = []
    for i in range(len(samples["doc_id"])):
        if samples["sub_strs"][i] and samples["sids"][i]:
            substrs += json.loads(samples["sub_strs"][i])
            sids += json.loads(samples["sids"][i])
            doc_ids += [samples["doc_id"][i]]*len(json.loads(samples["sids"][i]))
    out = {
        "document_id": doc_ids,
        "sid": sids,
        "substr": substrs,
    }
    return out

if __name__ == "__main__":

    try:
        # Your main code here
        print("Program is running. Press Ctrl+C to interrupt.")
        
        ds = load_dataset(
            "csv",
            data_files=glob.glob("/home/llm/translate/output/wiki_en/template_wiki_en.csv"),
            cache_dir="/home/llm/translate/data/wiki_en/cache",
            num_proc=96,
            split="train"
        )

        sent_ds = ds.map(
            flatten_sentence,
            batched=True,
            batch_size=256,
            num_proc=96,
            remove_columns=ds.features,
        )

        # sent_ds = split_dataset_by_node(sent_ds, rank=int(os.environ["RANK"]), world_size=int(os.environ["WORLD_SIZE"]))

        sent_ds_filtered = sent_ds.filter(
            lambda samples: [ False if not samples["substr"][i] or not len(samples["substr"][i].strip()) else True for i in range(len(samples["sids"])) ],
            batched=True,
            batch_size=1024,
            num_proc=96,
        )

        sent_ds_translated = sent_ds_filtered.map(
            partial(
                translate,
                src_lang="en",
                tgt_lang="hi",
            ),
            batched=True,
            batch_size=32,
            num_proc=PROC_COUNT,
            with_rank=True
        )

        sent_ds_translated_filtered = sent_ds_translated.filter(
            lambda samples: [ True if samples["translated"][i] != str(None) else False for i in range(len(samples["translated"])) ],
            batched=True,
            batch_size=256,
            num_proc=96,
        )

        sent_ds_translated_filtered.to_csv(
            f"/home/llm/translate/output/wiki_en/sent_translated",
            num_proc=96
        )

    except KeyboardInterrupt:
        print("Interrupted by user")
        cleanup()
        sys.exit(0)
    