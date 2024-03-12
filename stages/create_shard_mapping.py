import os
import glob
import argparse

def parse_args():

    parser = argparse.ArgumentParser(description="Create shard mappings")

    parser.add_argument(
        "--shards_root_dir",
        type=str,
        required=True,
        help="Root directory of shards",
    )

    parser.add_argument(
        "--devices_for_translation",
        type=lambda x: [ int(idx.strip()) for idx in x.split(",") if idx and len(idx.strip()) ],
        required=True, 
        help="List of devices for translation, separated by commas"
    )

    parser.add_argument(
        "--mapping_save_folder",
        type=str,
        required=True,
        help="Folder where mapping .txts will be saved."
    )

    args = parser.parse_args()

    return args


if __name__ == "__main__":

    args = parse_args()

    shards_paths = glob.glob(os.path.join(args.shards_root_dir, "*"))

    mapping_dict = { i: {"paths": [], "device": idx} for i, idx in enumerate(args.devices_for_translation) }

    for i, shard_path in enumerate(shards_paths):

        bin_id = i % len(args.devices_for_translation)

        mapping_dict[bin_id]["paths"] += [shard_path]

    os.makedirs(args.mapping_save_folder, exist_ok=True)

    for i in mapping_dict.keys():

        paths = mapping_dict[i]["paths"]
        idx = mapping_dict[i]["device"]

        with open(os.path.join(args.mapping_save_folder, f"device_map_{idx}"), "w") as mapping:

            mapping.write("\n".join(paths) + "\n") # Added extra newline for ease of reading via bash