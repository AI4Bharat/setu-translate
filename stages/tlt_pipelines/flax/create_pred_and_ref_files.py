from datasets import load_dataset
import argparse
import glob
import os

def parse_args():

    parser = argparse.ArgumentParser(description="create prediction and reference files for metric computation")

    parser.add_argument(
        "--ds_glob_path",
        type=str,
    )

    args = parser.parse_args()

    return args

if __name__ == "__main__":

    args = parse_args()

    ds_paths = glob.glob(args.ds_glob_path)

    for ds_path in ds_paths:
        src_lang, tgt_lang = os.path.split(ds_path)[1].split("__")
        print(f"Creating `preds` and `refs` files from: {ds_path}......")
        ds = load_dataset(
            "arrow", 
            data_files=os.path.join(ds_path, "*.arrow"),
            split="train",
        )
        with open(os.path.join(ds_path, "ref.txt"), "w") as ref_f, open(os.path.join(ds_path, "pred.txt"), "w") as pred_f:
            refs, preds = [], []
            for sample in ds:
                refs += [sample[f"tlt-{src_lang}__{tgt_lang}"]]
                preds += [sample[f"sentence_{tgt_lang}"]]
            ref_f.write("\n".join(refs))
            pred_f.write("\n".join(preds))
    
