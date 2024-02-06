import argparse
import os
import subprocess
from joblib import Parallel, delayed
from joblib.externals.loky.backend.context import get_context
import glob
import pandas as pd

def parse_args():
    parser = argparse.ArgumentParser(description="Run binarization and translation")

    # Adding arguments
    parser.add_argument(
        "--shards_root_dir", 
        type=str, 
        required=True, 
        help="Root directory for shards"
    )
    parser.add_argument(
        "--data_cache_dir", 
        type=str, 
        required=True, 
        help="Directory for data caching"
    )
    parser.add_argument(
        "--joblib_temp_folder", 
        type=str, 
        required=True, 
        help="Temporary folder for joblib"
    )
    parser.add_argument(
        "--src_lang", 
        type=str, 
        required=True, 
        help="Source language"
    )
    parser.add_argument(
        "--tgt_lang", 
        type=str, 
        required=True, 
        help="Target language"
    )
    parser.add_argument(
        "--num_procs_for_data_ops", 
        type=int, 
        required=True, 
        help="Number of processes for data operations"
    )
    parser.add_argument(
        "--batch_size", 
        type=int, 
        required=True, 
        help="Batch size"
    )
    parser.add_argument(
        "--devices_for_translation",
        type=lambda x: [ int(idx.strip()) for idx in x.split(",") if idx and len(idx.strip()) ],
        required=True, 
        help="List of devices for translation, separated by commas"
    )
    parser.add_argument(
        "--error_log_dir", 
        type=str, 
        required=True, 
        help="Save error logfiles in this folder"
    )
    parser.add_argument(
        "--save_status_path", 
        type=str, 
        required=True, 
        help="Save status csv at this path"
    )
    args = parser.parse_args()
    return args

def run_subprocess(command, shard_dir, stage):
    try:
        with subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE) as p:
            buffer = ''
            while True:
                # Check if child process has terminated
                exit_code = p.poll()
                if exit_code is not None:
                    # If the exit code indicates an error, handle it accordingly
                    if exit_code != 0:
                        # Read any error message from stderr
                        error_message = p.stderr.read().decode("utf-8")
                        raise subprocess.CalledProcessError(
                            cmd=command,
                            stderr=f"`{stage}` process ended with error code {exit_code}:\n {error_message}",
                            returncode=exit_code
                        )
                    break

                character = p.stdout.read1().decode("utf-8")
                if character == '\r':
                    print(buffer, end='\r', flush=True)
                    buffer = ''
                elif character == '\n':
                    print(buffer, flush=True)
                    buffer = ''
                else:
                    buffer += character

            # Make sure to read any remaining output after child process ends
            buffer = p.stdout.read().decode("utf-8")
            print(buffer, flush=True)

            print(f"Completed `{stage}` for {shard_dir}")

            return (shard_dir, None)

    except subprocess.CalledProcessError as e:

        error_file_path = os.path.join(log_dir, f"{os.path.split(shard_dir)[1]}.{stage}.error")
        with open(error_file_path, "w") as error_f:
            error_f.write(e.stderr)

        return (shard_dir, error_file_path)

def run_binarize_and_translate(
    shard_dir,
    setu_translate_root,
    data_cache_dir,
    joblib_temp_folder,
    src_lang,
    tgt_lang,
    num_procs_for_data_ops,
    batch_size,
    device_idx,
    log_dir,
):
    print(f"Starting processing of: {shard_dir}")
    binarize_command = [
        "python",
        os.path.join(setu_translate_root, "setu-translate/stages/binarize.py"),
        "--root_dir",
        os.environ["PWD"],
        "--data_files",
        f"{shard_dir}/sentences/*.arrow",
        "--cache_dir",
        data_cache_dir,
        "--binarized_dir",
        f"{shard_dir}/{tgt_lang}/binarized_sentences",
        "--joblib_temp_folder",
        joblib_temp_folder,
        "--batch_size",
        f"{batch_size}",
        "--total_procs",
        f"{num_procs_for_data_ops}",
        "--run_joblib",
        f"{False}",
        "--src_lang",
        src_lang,
        "--tgt_lang",
        tgt_lang
    ]
    
    output = run_subprocess(binarize_command, shard_dir, "binarize")

    if output[1]:
        return output

    translate_command = [
        "python",
        os.path.join(setu_translate_root, "setu-translate/stages/tlt_pipelines/translate_joblib.py"),
        "--root_dir",
        os.environ["PWD"],
        "--data_files",
        f"{shard_dir}/{tgt_lang}/binarized_sentences/*.arrow",
        "--cache_dir",
        data_cache_dir,
        "--base_save_dir",
        f"{shard_dir}/{tgt_lang}/model_out",
        "--joblib_temp_folder",
        joblib_temp_folder,
        "--batch_size",
        f"{batch_size}",
        "--total_procs",
        f"{num_procs_for_data_ops}",
        "--devices",
        f"{device_idx}",
    ]

    output = run_subprocess(translate_command, shard_dir, "translate")
    
    return output

if __name__ == "__main__":

    args = parse_args()

    shards_status = Parallel(
        n_jobs=len(args.devices_for_translation),
        verbose=0, 
        backend="loky",
        batch_size="auto",
        pre_dispatch='1.5*n_jobs',
    )(
        delayed(run_binarize_and_translate)(
            shard_dir=shard_dir,
            setu_translate_root=os.environ["SETU_TRANSLATE_ROOT"],
            data_cache_dir=args.data_cache_dir,
            joblib_temp_folder=args.joblib_temp_folder,
            src_lang=args.src_lang,
            tgt_lang=args.tgt_lang,
            num_procs_for_data_ops=args.num_procs_for_data_ops // len(args.devices_for_translation),
            batch_size=args.batch_size,
            device_idx=args.devices_for_translation[idx % len(args.devices_for_translation)],
            log_dir=args.error_log_dir,
        )  for idx, shard_dir in enumerate(glob.glob(os.path.join(args.shards_root_dir, "*")))
    )

    status_dict = {
        "shard_id": [shard_dir for shard_dir, _ in shards_status],
        "status_file": [status_file for _, status_file in shards_status],
    }

    df = pd.DataFrame.from_dict(status_dict)

    df.to_csv(args.save_status_path)