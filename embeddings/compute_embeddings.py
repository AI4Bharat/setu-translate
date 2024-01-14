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
import pickle

class TritonEmbeddingExtractor:

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

    def get_embedding_input_for_triton(self, texts: list):
        return [
            self.get_string_tensor([[text] for text in texts], "INPUT_TEXT"),
        ]

    def extract_embeddings(self, input_sentences):
        inputs = self.get_embedding_input_for_triton(input_sentences)
        output0 = http_client.InferRequestedOutput("EMBEDDING_VECTOR")
        response = self.triton_http_client.infer(
            "opt_125m",
            model_version='1',
            inputs=inputs,
            outputs=[output0],
            headers=self.http_headers,
        )
        output_batch = response.as_numpy('EMBEDDING_VECTOR').tolist()
        return list(map(lambda byte_string: pickle.loads(byte_string), output_batch))

def get_embedding_vectors(
    samples,
    col_for_embedding_extraction,
    output_col_name,
):
    extractor = TritonEmbeddingExtractor('0.0.0.0:8000')    
    embedding_vector_out = []
    to_send_data = []
    send_indices = []
    for i in range(len(samples[col_for_embedding_extraction])):
        to_send = samples[col_for_embedding_extraction][i]
        if to_send and len(to_send):
            to_send_data += [samples[col_for_embedding_extraction][i]]
            send_indices += [i]
            embedding_vector_out += ['']
        else:
            embedding_vector_out += [None]

    embeddings = extractor.extract_embeddings(to_send_data)

    for i, indice in enumerate(send_indices):
        embedding_vector_out[indice] = embeddings[i]

    return samples | {
        output_col_name: embedding_vector_out,
    }

def remove_non_terminated_chunks(samples):
    TERMINAL_PUNCTUATIONS = (
            ".", "!", "?", ":", ",", ";", ")", "\"", "\'",
    )
    # chunks ending with these patterns should be completely removed.
    TERMINAL_PUNCTUATIONS_EXCEPTION = (
        "...",
        "####",
    )
    
    def is_terminal_valid(text):
        if text.endswith(TERMINAL_PUNCTUATIONS_EXCEPTION):
            return False
        return text.endswith(TERMINAL_PUNCTUATIONS)
        
    texts = samples.pop("content")
    cleaned_text = []
    for text in texts:
        chunks = [chunk for chunk in text.split("\n") if is_terminal_valid(chunk) ] 
        cleaned_text += ["\n".join(chunks)]
    
    return samples | {
        "content": cleaned_text
    }

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

    rw = load_dataset(
        "parquet",
        data_files=glob.glob("/data-3/priyam/translation/refinedweb-mini/100_000.parquet"),
        cache_dir="/data-3/priyam/translation/refinedweb-mini/cache",
        num_proc=96,
    )["train"]

    rw_cleaned = rw.map(
        remove_non_terminated_chunks,
        batched=True,
        batch_size=8,
        num_proc=96,
    )

    rw_extracted = rw_cleaned.map(
        partial(
            get_embedding_vectors,
            col_for_embedding_extraction="content",
            output_col_name="embedding_vector",
        ),
        batched=True,
        batch_size=32,
        num_proc=3,
    )

    rw_extracted.to_csv("/data-3/priyam/translation/output/embedding/100_000.csv")

