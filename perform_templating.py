import numpy as np
from datasets import load_dataset, Dataset
from functools import partial
import glob
import os
import re
import pandas as pd
import csv
from document import Document
import spacy
import argparse
import json
import csv
from hashlib import sha256

def parse_args():

    parser = argparse.ArgumentParser(description="Performs templating on the entire dataset and extract strings")

    parser.add_argument(
        "--glob_path",
        type=str,
        required=True,
    )

    parser.add_argument(
        "--cache_dir_for_original_data",
        type=str,
        required=True,
    )

    parser.add_argument(
        "--save_path",
        type=str,
        required=True,
    )

    parser.add_argument(
        "--base_save_path",
        type=str,
        required=True,
    )

    parser.add_argument(
        "--id_col",
        type=str,
        required=False,
        default=None
    )

    parser.add_argument(
        "--text_col",
        type=str,
        required=False,
        default="content"
    )

    parser.add_argument(
        "--url_col",
        type=str,
        required=False,
        default=None
    )

    parser.add_argument(
        "--timestamp_col",
        type=str,
        required=False,
        default=None
    )

    parser.add_argument(
        "--source_type",
        type=str,
        required=True,
    )   

    parser.add_argument(
        "--translation_type",
        type=str,
        required=False,
        default="sentence",
        choices=["sentence", "chunk"]
    )

    parser.add_argument(
        "--use_cache",
        action="store_true"
    )

    parser.add_argument(
        "--filter_invalid_terminal",
        action="store_true"
    )

    parser.add_argument(
        "--use_spacy",
        action="store_true"
    )

    parser.add_argument(
        "--add_placeholders",
        action="store_true"
    )

    parser.add_argument(
        "--sample_size",
        type=int,
        required=False,
        default=None
    )

    parser.add_argument(
        "--write_to_mini_csvs",
        action="store_true"
    )

    parser.add_argument(
        "--write_style",
        type=str,
        required=False,
        default="doc",
        choices=["doc", "batch"]
    )

    args = parser.parse_args()

    return args

def remove_non_terminated_chunks(
    samples,
    text_col="content",
):
    
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
        
    texts = samples.pop(text_col)
    cleaned_text = []
    for text in texts:
        chunks = [chunk for chunk in text.split("\n") if is_terminal_valid(chunk) ] 
        cleaned_text += ["\n".join(chunks)]
    
    return samples | {
        text_col: cleaned_text
    }

def perform_templating(
    samples,
    add_placeholders=False, 
    sent_tokenization_model=None,
    id_col=None,
    text_col="content",
    url_col="url",
    timestamp_col="timestamp",
    source_type="refinedweb_cc",
    translation_type="sentence",
):

    templated_txts = dict()
    docs_to_template = []

    for i in range(len(samples[text_col])):

        doc_schema = Document.get_template_doc_schema()
        for key in doc_schema.keys():
            templated_txts[key] = templated_txts.get(key, []) + doc_schema[key]

        if not samples[text_col][i] or not len(samples[text_col][i]) or samples[text_col][i] == str(None):
            continue
        else:
            docs_to_template += [i]

        doc = Document(
            text=samples[text_col][i],
            doc_id=samples[id_col][i] if id_col else None,
            url=samples[url_col][i] if url_col else None,
            timestamp=samples[timestamp_col][i] if timestamp_col else None,
            source_type=source_type,
            translation_type=translation_type,
            sent_tokenization_model=sent_tokenization_model,
        )

        if add_placeholders:
            doc.add_placeholders()

        doc_dict = doc.get_templated_document_attrs()

        for key in doc_schema.keys():
            templated_txts[key][docs_to_template[i]] = doc_dict[key]
        
    return templated_txts

def write_to_batch_lvl_csvs(
    samples,
    idx,
    base_save_path=None,
):
    batch_filename = sha256(f"{idx}".encode('utf-8')).hexdigest()
    df_save_path = os.path.join(base_save_path, batch_filename)

    csv_paths = []
    start_pos_s = []
    end_pos_s = []

    with open(df_save_path, 'w') as csvfile:

        csvwriter = csv.writer(csvfile)

        fields = ["doc_id", "sid", "substr"]
        csvwriter.writerow(fields) # writing the fields  

        start_pos = 1

        for i in range(len(samples["doc_id"])):

            sids = json.loads(samples["sids"][i])
            substrs = json.loads(samples["sub_strs"][i])
            doc_ids = samples["doc_id"][i]*len(samples["sids"][i])
            rows = list(zip(doc_ids, sids, substrs))

            csvwriter.writerows(rows)  # writing the data rows  

            end_pos = start_pos + len(rows)
        
            csv_paths += [df_save_path]
            start_pos_s += [start_pos]
            end_pos_s += [end_pos]

            start_pos = end_pos

    return samples | {
        "csv_path": csv_paths,
        "start_pos": start_pos_s,
        "end_pos": end_pos_s,
    }

def write_to_doc_lvl_csvs(
    samples,
    idx,
    base_save_path=None,
):
    batch_folder_name = sha256(f"{idx}".encode('utf-8')).hexdigest()
    batch_folder_path = os.path.join(base_save_path, batch_folder_name)
    os.makedirs(batch_folder_path, exist_ok=True)

    csv_paths = []

    for i in range(len(samples["doc_id"])):

        sids = json.loads(samples["sids"][i])
        substrs = json.loads(samples["sub_strs"][i])
        doc_ids = [samples["doc_id"][i]]*len(samples["sids"][i])
        rows = list(zip(doc_ids, sids, substrs))

        fields = ["doc_id", "sid", "substr"]

        df_save_path = os.path.join(batch_folder_path, samples["doc_id"][i])

        with open(df_save_path, 'w') as csvfile:
            csvwriter = csv.writer(csvfile)
            csvwriter.writerow(fields) # writing the fields 
            csvwriter.writerows(rows) # writing the data rows  

        csv_paths += [df_save_path]

    return samples | {
        "csv_path": csv_paths,
    }
    
if __name__ == "__main__":

    args = parse_args()

    sample_size = args.sample_size
    perform_invalid_terminal_chunk_removal = args.filter_invalid_terminal
    use_spacy = args.use_spacy

    rw = load_dataset(
        "parquet",
        data_files=glob.glob(args.glob_path),
        cache_dir=args.cache_dir_for_original_data,
        num_proc=96,
        split="train")

    print(f"Loaded Dataset from path - {args.glob_path}")
    
    if args.sample_size:
        rw = rw.select(range(sample_size))
        print(f"Sampled dataset of size - {args.sample_size}")

    if perform_invalid_terminal_chunk_removal:
        rw = rw.map(
            partial(
                remove_non_terminated_chunks,
                text_col=args.text_col
            ),
            batched=True,
            batch_size=256,
            num_proc=96,
            load_from_cache_file=args.use_cache,
        )
        print(f"Performed `terminal punctuation check`")

    sent_tokenization_model = spacy.load("en_core_web_md") if use_spacy else None
    os.makedirs(args.base_save_path, exist_ok=True)
    rw_templated = rw.map(
        partial(
            perform_templating,
            add_placeholders=args.add_placeholders,
            sent_tokenization_model=sent_tokenization_model,
            id_col=args.id_col,
            text_col=args.text_col,
            url_col=args.url_col,
            timestamp_col=args.timestamp_col,
            source_type=args.source_type,
            translation_type=args.translation_type,
        ),
        batched=True,
        batch_size=256,
        num_proc=96,
        remove_columns=rw.features,
        load_from_cache_file=args.use_cache,
    )
    print(f"Performed `templating`")

    rw_templated_filtered = rw_templated.filter(
        lambda samples: [ True if samples["doc_id"][i] != str(None) else False for i in range(len(samples["doc_id"])) ],
        batched=True,
        batch_size=256,
        num_proc=96,
        load_from_cache_file=args.use_cache,
    )
    print(f"Filtered `null` text docs")

    if args.write_to_mini_csvs:

        write_style = {
            "batch": write_to_batch_lvl_csvs,
            "doc": write_to_doc_lvl_csvs,
        }

        rw_templated_filtered = rw_templated_filtered.map(
            partial(
                write_style[args.write_style],
                base_save_path=args.base_save_path,
            ),
            batched=True,
            batch_size=256,
            num_proc=96,
            load_from_cache_file=args.use_cache,
            with_indices=True
        )
        print(f"Written mini sentence csvs using {args.write_style} style")

        csv_paths = rw_templated_filtered.remove_columns([col for col in rw_templated_filtered.features if col != "csv_path"])
        csv_paths_file = os.path.join(args.base_save_path, "paths.csv")
        csv_paths.to_csv(csv_paths_file, num_proc=96)
        print(f"Saved `paths.csv` to {csv_paths_file}")

    rw_templated_filtered.to_csv(args.save_path, num_proc=96)
    print(f"Saved `templated` dataset to {args.save_path}")
    

    