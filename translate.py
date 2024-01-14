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
from document import Document

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
        # print("Is server ready - {}".format(self.triton_http_client.is_server_ready(headers=self.http_headers)))
    
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

def translate(
    samples,
    src_lang,
    tgt_lang,
):
    
    translator = TritonTranslator('0.0.0.0:8000')
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
    
if __name__ == "__main__":

    sample_size = 100

    sent_ds = load_dataset(
        "csv",
        data_files=glob.glob("/data-3/priyam/translation/output/translation/deduped_global_sentences.csv/*"),
        cache_dir="/data-3/priyam/translation/refinedweb-mini/cache",
        num_proc=96,
        split="train"
    )

    sent_ds_translated = sent_ds.map(
        partial(
            translate,
            src_lang="en",
            tgt_lang="hi",
        ),
        batched=True,
        batch_size=1,
        num_proc=1
    )

    sent_ds_translated_filtered = sent_ds_translated.filter(
        lambda samples: [ True if samples["translated"][i] != str(None) else False for i in range(len(samples["translated"])) ],
        batched=True,
        batch_size=256,
        num_proc=96,
    )

    sent_ds_translated_filtered.to_csv(f"/data-3/priyam/translation/output/translation/sent_translated_{sample_size}.csv")

    