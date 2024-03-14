# Creating a TPU Pod:
1. Upload `setu-translate` repo to GCS.
2. Run the below command. Make sure to pass the paths and names.
```
gcloud compute tpus tpu-vm create tlt-flax-build \
    --zone=us-central2-b \
    --accelerator-type=v4-64 \
    --version=tpu-ubuntu2204-base \
    --metadata=startup-script-url=<GCS-path-of-"setup_environment.sh"> \
    --data-disk source=projects/<project-id>/zones/us-central2-b/disks/<disk-name-of-disk-containing-data>,mode=read-only
```

# To calculate the metrics:
1. use `evaluate_metrics.py` to get flax predictions on `IN22-Conv` and `IN22-Gen`.
2. use `create_pred_and_ref_files.py` to create `.txt` files of the predictions and references.
3. run `compute_metrics.sh`. This file internally calls [IndicTrans2 repo's](https://github.com/AI4Bharat/IndicTrans2) `compute_metrics.sh`.

### Some Jax/Flax help:
Refer to: https://www.googlecloudcommunity.com/gc/Developer-Tools/TPU-POD-no-initiation/m-p/597179

#### Handle Multi-Host process:
https://jax.readthedocs.io/en/latest/multi_process.html
https://github.com/google/jax/discussions/16789#discussioncomment-6615502
https://jax.readthedocs.io/en/latest/_autosummary/jax.make_array_from_single_device_arrays.html
https://jax.readthedocs.io/en/latest/_autosummary/jax.experimental.multihost_utils.process_allgather.html#jax.experimental.multihost_utils.process_allgather
