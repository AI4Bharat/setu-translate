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
    errors = []
    translated_txts = dict()

    docs = []
    sentences = []
    sent_per_doc = []
    docs_to_translate = []

    for i in range(len(samples["content"])):
        for key in Document.get_doc_schema():
            translated_txts[key] = translated_txts.get(key, []) + [str(None)]
        if not samples["content"][i] or not len(samples["content"][i]) or samples["content"][i] == str(None):
            continue
        else:
            docs_to_translate += [i]
        doc = Document(
            text=samples["content"][i],
            doc_id=None, 
            url=samples["url"][i], 
            timestamp=samples["timestamp"][i],
            source_type="refinedweb_cc",
            translation_type="sentence",
        )
        doc_sentences = doc.sentences["text"].tolist()
        docs += [doc]
        sentences += doc_sentences       
        sent_per_doc += [len(doc_sentences)]

    assert len(docs_to_translate) == len(docs)

    if not len(sentences):
        return translated_txts

    batch_size = 128
    start = 0
    end = start + batch_size if len(sentences) > start + batch_size else len(sentences)
    translated_out = []
    while(start < end):
        translated_out += translator.translate(sentences[start:end], src_lang, tgt_lang)
        start = end
        end = start + batch_size if len(sentences) > start + batch_size else len(sentences)

    curr = 0
    for i in range(len(docs)):
        translated_sents_per_doc = translated_out[curr:curr+sent_per_doc[i]]
        _ = docs[i].translate(translated_sents_per_doc)
        doc_dict = docs[i].get_document_attrs()
        for key in doc_dict:
            # translated_txts[key][docs_to_translate[i]] = translated_txts.get(key, []) + [doc_dict[key]]
            translated_txts[key][docs_to_translate[i]] = doc_dict[key]
        curr += sent_per_doc[i]

    return translated_txts
    
if __name__ == "__main__":

    # rw = load_dataset(
    #     "parquet",
    #     data_files=glob.glob("/data-3/priyam/translation/refinedweb/*.parquet"),
    #     cache_dir="/data-3/priyam/translation/refinedweb-cached",
    #     num_proc=96,
    # )["train"]

    # rw_trimmed = rw.shuffle(seed=42).select(range(100_000))

    # for sample in rw.select(range(3)):
    #     print(sample)

    # rw_trimmed.to_parquet("/data-3/priyam/translation/refinedweb-mini/100_000.parquet")

    sample_size = 100

    rw = load_dataset(
        "parquet",
        data_files=glob.glob("/data-3/priyam/translation/refinedweb-mini/100_000.parquet"),
        cache_dir="/data-3/priyam/translation/refinedweb-mini/cache",
        num_proc=96,
    )["train"].select(range(sample_size))

    rw_cleaned = rw.map(
        remove_non_terminated_chunks,
        batched=True,
        batch_size=8,
        num_proc=96,
    )

    rw_translated = rw_cleaned.map(
        partial(
            translate,
            src_lang="en",
            tgt_lang="hi",
        ),
        batched=True,
        batch_size=1,
        num_proc=1,
        remove_columns=rw_cleaned.features
    )

    rw_translated.to_csv(f"/data-3/priyam/translation/output/translation/{sample_size}.csv")

    