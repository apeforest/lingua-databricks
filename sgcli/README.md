# Lingua Example

This example demonstrates how to run [Meta Lingua](https://github.com/facebookresearch/lingua) on Serverless GPU Compute using SGCLI.

Meta Lingua is an open source library from Meta for training language models with efficient data loading and distributed training.

## Prerequisites

1. A Hugging Face account
2. Your HF token stored in the secrets vault at the path specified in `workload.yaml`
3. Access to a Databricks workspace with the training data

## Files

- `workload.yaml` - SGCLI workload configuration
- `requirements.yaml` - Python dependencies (xformers, mlflow)

## Usage

This example requires access to the training data in the df1 workspace.

```bash
# Set the Databricks profile
export DATABRICKS_CONFIG_PROFILE=df1

# Run the workload
sgcli run -f workload.yaml
```

Or watch the logs in real-time:

```bash
sgcli run -f workload.yaml --watch
```

## Configuration Details

This example uses:

- **Code Snapshot** - Syncs the local `lingua` repository to the remote environment
- **torchrun** - For distributed training across 8 GPUs
- **MLflow** - For experiment tracking

## Customization

### Training Configuration

```yaml
command: |-
  torchrun ... -m apps.main.train config=apps/main/configs/your_config.yaml
```

### Multi-Node Training

To scale to multiple nodes:

```yaml
compute:
  gpus: 16  # 2 nodes × 8 GPUs
  gpu_type: h100

command: |-
  torchrun \
    --nnodes=2 \
    --nproc_per_node=8 \
    ...
```

### Data Preparation

To download and prepare data before training:

```bash
python setup/download_prepare_hf_data.py fineweb_edu 16 --data_dir /tmp/data --seed 42 --nchunks 32
```

## Resources

- [Meta Lingua GitHub](https://github.com/facebookresearch/lingua)
- [SGCLI User Guide](https://docs.google.com/document/d/1gjwD4YiR1x8L1vZ5VzDomcUeNuMG1wVVrKsmT-M_nUU)