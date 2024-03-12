from joblib import Parallel, delayed
import pandas as pd
import json
import glob
import os

ROW_COUNT = 2000

def process_files(file_paths):
    data = []
    for file_path in file_paths:
        try:
            with open(file_path, "r") as jf:
                content = json.load(jf)
            record = [
                file_path.split("/")[-1],
                content["url"],
                content["source"],
                content["timestamp"],
                content["html"]
            ]
            data.append(record)
        except Exception as e:
            print(f"Error processing {file_path} - {e}")
    return data

def save_to_parquet(data, output_path, lang, batch_index):
    df = pd.DataFrame(data, columns=["doc_id", "url", "source", "timestamp", "html"])
    file_path = os.path.join(output_path, f"{lang}_{batch_index}.parquet")
    df.to_parquet(file_path)

def _to_parquet_parallel(files, output_path, lang, n_jobs=-1):
    os.makedirs(output_path, exist_ok=True)

    # Splitting the file list into batches
    file_batches = [files[i:i + ROW_COUNT] for i in range(0, len(files), ROW_COUNT)]

    # Processing each batch in parallel and saving to separate parquet files
    Parallel(n_jobs=n_jobs)(
        delayed(lambda batch, idx: save_to_parquet(process_files(batch), output_path, lang, idx))(batch, idx)
        for idx, batch in enumerate(file_batches)
    )

if __name__ == "__main__":
    json_paths = glob.glob("/data-3/priyam/translation/data/wikimedia/*/clean_data/en/*")
    lang = "english"
    base_dir = "/home/safi/sangraha++/parquets/yourstory/"

    print("Starting conversion....")
    _to_parquet_parallel(json_paths, base_dir, lang)
    print(f"ed {lang} files to parquet in batches of {ROW_COUNT} rows each.")