'''
Usage:  
- Do `pip install tritonclient[all] gevent` first.
- Then use this sample script with the appropriate Triton-server endpoint_url
'''

import tritonclient.http as http_client
from tritonclient.utils import *
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from functools import partial
import glob
import os
import re
import pandas as pd
import csv
from document import Document
import argparse
import json
import os
import multiprocessing as mp
from joblib import Parallel, delayed
import sys
import tqdm
import requests
import time
from datetime import datetime
import traceback

def cleanup(
    idx_logging,
    resume_log_file_path,
    status_df,
    status_dir
):
    with open(resume_log_file_path, "a+") as resume_f:
        resume_f.write(f'Saving latest resume-index: <{idx_logging}> at - {datetime.now().strftime("%m/%d/%Y, %H:%M:%S")}\n')
    print(f"Saved `resume_idx` to {resume_log_file_path}....")

    status_df_save_path = os.path.join(status_dir, datetime.now().strftime("%m_%d_%Y-%H:%M:%S"))
    status_df.to_csv(status_df_save_path)
    print(f"Saved latest `status_df` to {status_df_save_path}....")
    
    print("Performed clean-up!")

def extract_resume_idx(text):
    # Regular expression to find text between < and >
    pattern = re.compile(r'<(.*?)>')
    # Find all occurrences of the pattern
    matches = pattern.findall(text)
    return matches

def send_request(texts, src_lang, tgt_lang, server_url="http://0.0.0.0:8000/v2/models/nmt/infer", auth_key=None):

    # The inference server URL
    TRITON_SERVER_URL = server_url

    # Authentication header
    headers = {
        "Authorization": f"Bearer {auth_key}",
        "Content-Type": "application/json"
    }

    body = json.dumps({
        "inputs": [
            {
                "name": "INPUT_TEXT",
                "shape": [len(texts), 1],
                "datatype": "BYTES",
                "data": np.array([[text] for text in texts], dtype=object).tolist()
            },
            {
                "name": "INPUT_LANGUAGE_ID",
                "shape": [len(texts), 1],
                "datatype": "BYTES",
                "data": [[src_lang]] * len(texts),
            },
            {
                "name": "OUTPUT_LANGUAGE_ID",
                "shape": [len(texts), 1],
                "datatype": "BYTES",
                "data": [[tgt_lang]] * len(texts)
            }
        ],
        "outputs": [
            {
                "name": "OUTPUT_TEXT"
            }
        ]
    })

    # Make the request
    response = requests.post(TRITON_SERVER_URL, headers=headers, data=body)

    # Check if the request was successful
    if response.status_code != 200:
        error = f"Error during inference request: {response.text}"
        return None, error, "triton"

    # Extract results from the response
    try:
        output_data = json.loads(response.text)
    except Exception as e:
        exception_info = str(e) # Capture the exception information
        traceback_info = traceback.format_exc() # Capture the traceback
        error = f"Exception: {exception_info}\nTraceback:\n{traceback_info}" # Combine exception information and traceback
        return None, error, "response_json"

    translated_txt = output_data['outputs'][0]['data'][0]
    return translated_txt, None, None

def translate(path, src_lang, tgt_lang, server_url="http://0.0.0.0:8000/v2/models/nmt/infer", auth_key=None):
    try:
        df = pd.read_csv(path)
        df = df[df.apply(lambda row: False if not row["substr"] or not len(str(row["substr"]).strip()) else True, axis=1)]
        strings = df["substr"].tolist()
        translated, error, stage = send_request(strings, src_lang, tgt_lang, server_url, auth_key)
        if error:
            with open(f"{path}.error.{stage}", "w") as error_f:
                error_f.write(error)
            return False, stage
        else:
            df["translated"] = translated
            df.to_csv(f"{path}.translated")
            return True, None
    except Exception as e:
        exception_info = str(e) # Capture the exception information
        traceback_info = traceback.format_exc() # Capture the traceback
        error = f"Exception: {exception_info}\nTraceback:\n{traceback_info}" # Combine exception information and traceback
        with open(f"{path}.error.main", "w") as error_f:
            error_f.write(error)
        return False, "main"

class TranslateData(Dataset):

    def __init__(self, paths_data, filetype, resume_idx=None):
        self.paths = self.get_read_api(filetype)(paths_data)
        self.resume_idx = resume_idx
        self.exhausted = False
        if resume_idx and resume_idx >= len(paths):
            self.exhausted = True

    def __getitem__(self, idx):
        return {"path": self.paths.iloc[idx]["csv_path"]}

    def __len__(self):
        return len(self.paths)

    def apply_resume(self):
        self.paths = self.paths.iloc[self.resume_idx:]

    @staticmethod
    def get_read_api(filetype):
        read_apis = {
            "csv": pd.read_csv,
            "parquet": pd.read_parquet,
        }
        return read_apis[filetype]


if __name__ == "__main__":

    try:
        print("Program is running. Press Ctrl+C to interrupt.")

        resume_idx=None
        resume_log_file_path="/home/llm/translate/output/wiki_en/resume_idx_log"
        resume_status_dir="/home/llm/translate/output/wiki_en/resume_status"
        paths_data="/home/llm/translate/output/wiki_en/doc_csvs/paths.csv"

        if os.path.isfile(resume_log_file_path):
            with open(resume_log_file_path, "r") as resume_f:
                latest_idx = extract_resume_idx(list(map(lambda x: x.strip(), resume_f.readlines()))[-1])[0]
                resume_idx = int(latest_idx) if latest_idx and len(latest_idx) else None

        ds = TranslateData(
            paths_data=paths_data,
            filetype="csv",
            resume_idx=resume_idx
        )

        if not ds.exhausted:
            ds.apply_resume()

        if not ds:
            print(f"Dataset loading failed as the `resume-idx`: {resume_idx} >= no.of paths. It seems the dataset is already exhausted.")
            sys.exit(0)
        else:
            print(f"Loaded Dataset from {paths_data} and resume-idx `{resume_idx}`")

        data_loader = DataLoader(
            ds,
            num_workers=1, 
            batch_size=256,
            collate_fn=lambda batch: { "path": [sample["path"] for sample in batch] },
            prefetch_factor=1,
            shuffle=False
        )
        print(f"Created DataLoader for the dataset")

        tlt_func = partial(
            translate,
            src_lang="en",
            tgt_lang="hi", 
            server_url="http://0.0.0.0:8000/v2/models/nmt/infer", 
            auth_key=None,
        )

        idx_logging = resume_idx if resume_idx else 0
        status_df = pd.DataFrame(columns=["path", "completed", "reason"])
        for idx, batch in tqdm.tqdm(enumerate(data_loader, 0), unit="batch", total=len(data_loader)):
            batch_status = Parallel(
                n_jobs=128,
                backend="threading",
                verbose=0, 
                batch_size="auto",
                pre_dispatch='2*n_jobs',
                temp_folder="/home/llm/translate/tmp"
            )(
                delayed(tlt_func)(path) for path in batch["path"]
            )
            idx_logging += len(batch["path"])
            batch_info_mapping = {
                "path": batch["path"],
                "completed": [completed for completed, _ in batch_status],
                "reason": [reason for _, reason in batch_status],
            }
            status_df = pd.concat(
                [
                    status_df, 
                    pd.DataFrame(batch_info_mapping)
                ], 
                ignore_index=True
            )
        

    except KeyboardInterrupt:

        print("Interrupted by user")

        cleanup(
            idx_logging=idx_logging,
            resume_log_file_path=resume_log_file_path,
            status_df=status_df,
            status_dir=resume_status_dir,
        )

        sys.exit(0)
    