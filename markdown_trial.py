# import re

# def strip_dates_from_ends(text):
#     # Regex pattern to match dates at the beginning and end of the text
#     # Matches YYYY-MM-DD, MM/DD/YYYY, or DD-MM-YYYY
#     pattern_start = r'^\s*(?:\d{4}-\d{2}-\d{2}|\d{2}/\d{2}/\d{4}|\d{2}-\d{2}-\d{4})\s*'
#     pattern_end = r'\s*(?:\d{4}-\d{2}-\d{2}|\d{2}/\d{2}/\d{4}|\d{2}-\d{2}-\d{4})\s*$'
    
#     # Remove date from the start
#     text = re.sub(pattern_start, '', text)
#     # Remove date from the end
#     text = re.sub(pattern_end, '', text)

#     return text

# # Example text with dates
# example_text = """
# 2021-05-15 This event occurred on a specific date and was followed by another event. 12/31/2021
# A third event occurred on a non-specific date. 15-06-2021
# """

# # Strip dates from the beginning and end
# result = strip_dates_from_ends(example_text)
# print(result)

from datasets import load_dataset
import glob
import pandas as pd
import json

data_files = glob.glob("/data-3/priyam/priyam/ananth-data/wiki_data/en/*")
print("Glob Complete...")
paths_df = pd.DataFrame({"paths": data_files})
paths_df.to_csv("/data-3/priyam/translation/wiki/paths.csv", index=False)

paths_ds = load_dataset(
    "csv", 
    data_files=["/data-3/priyam/translation/wiki/paths.csv"], 
    cache_dir="/data-3/priyam/translation/wiki",
    num_proc=96,
    split="train"
)

def read_files(samples):
    out = dict()
    for i in range(len(samples["paths"])):
        with open(samples["paths"][i], 'r') as f:
            data = json.load(f) 
        for key in data.keys():
            out[key] = out.get(key, []) + [data[key]]
    return out

wiki_ds = paths_ds.map(
    read_files,
    batched=True,
    batch_size=256,
    num_proc=96,
    remove_columns=paths_ds.features,
)

wiki_ds.to_parquet(
    "/data-3/priyam/translation/wiki/wiki_en_data.parquet",
)

# import glob
# import json
# import pandas as pd
# import threading
# from math import ceil

# def process_files(file_list, thread_index):
#     # Create a DataFrame for each thread
#     df = pd.DataFrame()
#     for file in file_list:
#         with open(file, 'r') as f:
#             data = json.load(f)
#             df = df.append(data, ignore_index=True)
#     # Save the DataFrame to a CSV file
#     save_path = f'/data-3/priyam/translation/wiki/csvs/thread_{thread_index}.csv'
#     df.to_csv(save_path, index=False)
#     print(f"Finished thread-{thread_index}. Written DF to {save_path}.....")

# def partition_files(file_paths, num_threads):
#     # Determine the size of each partition
#     total_files = len(file_paths)
#     partition_size = ceil(total_files / num_threads)
#     return [file_paths[i:i + partition_size] for i in range(0, total_files, partition_size)]

# def main():
#     # Path to JSON files (modify this path as needed)
#     path_to_json = "/data-3/priyam/priyam/ananth-data/wiki_data/en/*"

#     # Number of threads
#     num_threads = 96

#     # Get all JSON files matching the path
#     json_files = glob.glob(path_to_json)
#     print(f"Glob Complete...Found {len(json_files)} JSON files")

#     # Partition the files among the threads
#     partitions = partition_files(json_files, num_threads)
#     print(f"Created Partitions...Got {len(partitions)} partitions")

#     # Create and start threads
#     threads = []
#     for i, partition in enumerate(partitions):
#         thread = threading.Thread(target=process_files, args=(partition, i))
#         threads.append(thread)
#         thread.start()

#     # Wait for all threads to complete
#     for thread in threads:
#         thread.join()

# if __name__ == "__main__":
#     main()
