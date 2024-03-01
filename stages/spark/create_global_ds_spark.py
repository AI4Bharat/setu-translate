import glob
import argparse
import os
from pyspark.sql import SparkSession 

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
    
if __name__ == "__main__":

    args = parse_args()

    spark = SparkSession \
                .builder \
                .appName("create global ds") \
                .getOrCreate()

    ds = spark.read.format("parquet").load(args.glob_path)

    df = df.withColumn("tlt_data", F.arrays_zip("sids", "sub_strs"))
    df = df.select("*", F.posexplode("tlt_data")).drop(*["text", "tlt_data"])
    sentence_df = df.select("*", F.col("col.sids").alias("sids"), F.col("col.sub_strs").alias("sub_strs")).drop("col")

    os.makedirs(args.global_sent_ds_path, exist_ok=True)

    sentence_df \
        .write \
        .mode("overwrite") \
        .parquet(args.global_sent_ds_path)
    print(f"Saved `Sentence-Level` dataset to {args.global_sent_ds_path}")


