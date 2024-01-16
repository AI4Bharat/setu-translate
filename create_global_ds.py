from datasets import load_dataset
import glob
import pandas as pd
import argparse

def parse_args():

    parser = argparse.ArgumentParser(description="Creating a global sentence dataset")

    parser.add_argument(
        "--paths_data",
        type=str,
        required=True
    )

    parser.add_argument(
        "--cache_dir",
        type=str,
        required=True,
    )

    parser.add_argument(
        "--global_sent_ds_path",
        type=str,
        required=True,
    )

    args = parser.parse_args()

    return args

def read_csvs(
    samples,
    keys=["doc_id", "sid", "substr"],
):
    out = {key: [] for key in keys}
    for i in range(len(samples["csv_path"])):
        df = pd.read_csv(samples["csv_path"][i])
        df = df.dropna(subset=["substr"])
        df = df[df.apply(lambda row: False if not isinstance(row["substr"], str) or not len(row["substr"].strip()) else True, axis=1)]
        df_dict = df.to_dict('list')
        try:
            for key in out.keys():
                out[key] += df_dict[key]
        except Exception as e:
            print(f"Failed for path: {samples['csv_path'][i]}")
            print(str(e))
    return out
    
if __name__ == "__main__":

    args = parse_args()

    paths_ds = load_dataset(
        "csv",
        data_files=[args.paths_data],
        cache_dir=args.cache_dir,
        num_proc=128,
        split="train"
    )

    sentence_ds = paths_ds.map(
        read_csvs,
        batched=True,
        batch_size=256,
        num_proc=128,
        remove_columns=paths_ds.features
    )

    sentence_ds.to_csv(args.global_sent_ds_path)

