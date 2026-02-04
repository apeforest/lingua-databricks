# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

import logging
from collections import namedtuple
from dataclasses import dataclass, asdict
from typing import Dict, Any, List, Optional, Union
from pathlib import Path
import json
from datetime import datetime, timezone

import torch
import torch.nn as nn

from lingua.distributed import get_is_master
import wandb

try:
    import mlflow
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False

logger = logging.getLogger()


@dataclass
class WandbArgs:
    job_type: Optional[str] = None
    dir: Optional[str] = None
    project: Optional[str] = None
    entity: Optional[str] = None
    tags: Optional[List] = None
    group: Optional[str] = None
    name: Optional[str] = None
    notes: Optional[str] = None
    config_exclude_keys: Optional[List[str]] = None
    config_include_keys: Optional[List[str]] = None
    anonymous: Optional[str] = None
    mode: Optional[str] = None
    allow_val_change: Optional[bool] = None
    resume: Optional[Union[bool, str]] = None
    force: Optional[bool] = None
    tensorboard: Optional[bool] = None
    sync_tensorboard: Optional[bool] = None
    monitor_gym: Optional[bool] = None
    save_code: Optional[bool] = None
    id: Optional[str] = None
    fork_from: Optional[str] = None
    resume_from: Optional[str] = None


@dataclass
class MLflowArgs:
    """Configuration for MLflow experiment tracking."""
    enabled: bool = True  # Enable/disable MLflow logging
    
    # Experiment settings
    experiment_name: Optional[str] = None  # Name of the experiment (will be created if doesn't exist)
    run_name: Optional[str] = None  # Name of this specific run
    run_id: Optional[str] = None  # Resume a specific run by ID
    
    # Tracking server settings
    tracking_uri: Optional[str] = None  # MLflow tracking server URI (e.g., "databricks" or "http://localhost:5000")
    artifact_location: Optional[str] = None  # Custom artifact storage location
    
    # Run metadata
    tags: Optional[Dict[str, str]] = None  # Tags to attach to the run
    description: Optional[str] = None  # Run description
    
    # Logging options
    log_system_metrics: bool = True  # Log system metrics (CPU, memory, etc.)
    log_model_params: bool = True  # Log model parameters count as metric
    nested: bool = False  # Create nested run under active run


@dataclass
class LoggingArgs:
    freq: int = 10  # Log every freq optimizer steps
    acc_freq: Optional[int] = None  # Log every acc_freq gradient accumulation steps

    wandb: Optional[WandbArgs] = None
    mlflow: Optional[MLflowArgs] = None


class MetricLogger:
    def __init__(self, outdir: Path, args: Optional[Any] = None):
        self.outdir = outdir
        self.jsonl_writer = None
        self.args = args
        self._mlflow_run = None

    def _init_mlflow(self):
        """Initialize MLflow tracking if configured."""
        if not MLFLOW_AVAILABLE:
            logger.warning("MLflow is not installed. Install with: pip install mlflow")
            return
        
        mlflow_args = self.args.logging.mlflow
        
        # Set tracking URI if provided
        if mlflow_args.tracking_uri:
            mlflow.set_tracking_uri(mlflow_args.tracking_uri)
        
        # Set or create experiment
        if mlflow_args.experiment_name:
            mlflow.set_experiment(mlflow_args.experiment_name)
        
        # Prepare run kwargs
        run_kwargs = {}
        if mlflow_args.run_name:
            run_kwargs["run_name"] = mlflow_args.run_name
        if mlflow_args.run_id:
            run_kwargs["run_id"] = mlflow_args.run_id
        if mlflow_args.tags:
            run_kwargs["tags"] = mlflow_args.tags
        if mlflow_args.description:
            run_kwargs["description"] = mlflow_args.description
        if mlflow_args.nested:
            run_kwargs["nested"] = mlflow_args.nested
        if mlflow_args.log_system_metrics:
            run_kwargs["log_system_metrics"] = mlflow_args.log_system_metrics
        
        # Start run
        self._mlflow_run = mlflow.start_run(**run_kwargs)
        
        # Log configuration as params
        config_dict = asdict(self.args)
        flat_config = _flatten_config_for_mlflow(config_dict)
        
        # MLflow has a limit on param value length, so we truncate long values
        truncated_config = {}
        for k, v in flat_config.items():
            str_v = str(v)
            if len(str_v) > 500:
                str_v = str_v[:497] + "..."
            truncated_config[k] = str_v
        
        # Log params in batches (MLflow has limits on batch size)
        param_items = list(truncated_config.items())
        batch_size = 100
        for i in range(0, len(param_items), batch_size):
            batch = dict(param_items[i:i + batch_size])
            mlflow.log_params(batch)
        
        logger.info(f"MLflow run started: {self._mlflow_run.info.run_id}")

    def open(self):
        if self.jsonl_writer is None:
            self.jsonl_writer = open(self.outdir, "a")
        
        # Initialize wandb if configured
        if (
            self.args is not None
            and self.args.logging.wandb is not None
            and get_is_master()
        ):
            run = wandb.init(
                config=asdict(self.args),
                **asdict(self.args.logging.wandb),
            )
        
        # Initialize MLflow if configured and enabled
        if (
            self.args is not None
            and self.args.logging.mlflow is not None
            and self.args.logging.mlflow.enabled
            and get_is_master()
        ):
            self._init_mlflow()

    def log(self, metrics: Dict[str, Any]):
        if (
            self.args is not None
            and self.args.logging.wandb is not None
            and (wandb.run is not None)
        ):
            wandb.log(metrics, step=metrics["global_step"])
        
        # Log to MLflow
        if (
            self.args is not None
            and self.args.logging.mlflow is not None
            and self.args.logging.mlflow.enabled
            and MLFLOW_AVAILABLE
            and self._mlflow_run is not None
        ):
            # Filter out non-numeric values and flatten nested metrics
            mlflow_metrics = {}
            for k, v in metrics.items():
                if isinstance(v, (int, float)) and not isinstance(v, bool):
                    # MLflow metric names can't have certain characters
                    safe_key = k.replace("/", ".").replace(" ", "_")
                    mlflow_metrics[safe_key] = v
            
            if mlflow_metrics:
                mlflow.log_metrics(mlflow_metrics, step=metrics["global_step"])

        metrics.update({"created_at": datetime.now(timezone.utc).isoformat()})
        print(json.dumps(metrics), file=self.jsonl_writer, flush=True)

    def close(self):
        if self.jsonl_writer is not None:
            self.jsonl_writer.close()
            self.jsonl_writer = None
        
        # End MLflow run
        if self._mlflow_run is not None and MLFLOW_AVAILABLE:
            mlflow.end_run()
            self._mlflow_run = None

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    def __del__(self):
        self.close()


def _flatten_config_for_mlflow(d: Dict, parent_key: str = "", sep: str = ".") -> Dict[str, Any]:
    """Flatten a nested dictionary for MLflow param logging."""
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(_flatten_config_for_mlflow(v, new_key, sep=sep).items())
        elif isinstance(v, list):
            # Convert lists to string representation
            items.append((new_key, str(v)))
        elif v is not None:
            items.append((new_key, v))
    return dict(items)


GPUMemStats = namedtuple(
    "GPUMemStats",
    [
        "max_active_gib",
        "max_active_pct",
        "max_reserved_gib",
        "max_reserved_pct",
        "num_alloc_retries",
        "num_ooms",
        "power_draw",
    ],
)


class GPUMemoryMonitor:
    """
    Class to monitor GPU memory usage
    """

    def __init__(self, device: str = "cuda:0"):
        self.device = torch.device(device)  # device object
        self.device_name = torch.cuda.get_device_name(self.device)
        self.device_index = torch.cuda.current_device()
        self.device_capacity = torch.cuda.get_device_properties(
            self.device
        ).total_memory
        self.device_capacity_gib = self._to_gib(self.device_capacity)

        # reset stats, clear cache
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()

    def _to_gib(self, memory_in_bytes):
        # NOTE: GiB (gibibyte) is 1024, vs GB is 1000
        _gib_in_bytes = 1024 * 1024 * 1024
        memory_in_gib = memory_in_bytes / _gib_in_bytes
        return memory_in_gib

    def _to_pct(self, memory):
        return 100 * memory / self.device_capacity

    def get_peak_stats(self):
        cuda_info = torch.cuda.memory_stats(self.device)

        max_active = cuda_info["active_bytes.all.peak"]
        max_active_gib = self._to_gib(max_active)
        max_active_pct = self._to_pct(max_active)

        max_reserved = cuda_info["reserved_bytes.all.peak"]
        max_reserved_gib = self._to_gib(max_reserved)
        max_reserved_pct = self._to_pct(max_reserved)

        num_retries = cuda_info["num_alloc_retries"]
        num_ooms = cuda_info["num_ooms"]
        power_draw = torch.cuda.power_draw()

        if num_retries > 0:
            logger.warning(f"{num_retries} CUDA memory allocation retries.")
        if num_ooms > 0:
            logger.warning(f"{num_ooms} CUDA OOM errors thrown.")

        return GPUMemStats(
            max_active_gib,
            max_active_pct,
            max_reserved_gib,
            max_reserved_pct,
            num_retries,
            num_ooms,
            power_draw,
        )

    def reset_peak_stats(self):
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.reset_accumulated_memory_stats()

    def __str__(self):
        mem_stats = self.get_peak_stats()
        display_str = f"{self.device_name} ({self.device_index}): {self.device_capacity_gib} GiB capacity, "
        display_str += (
            f"{mem_stats.max_reserved_gib} GiB peak, {mem_stats.max_reserved_pct}% peak"
        )
        return f"{display_str}"


def upload_train_to_wandb(
    ckpt_dir, project="lingua", entity="codegen-team", train=True, eval=True
):
    import wandb
    from omegaconf import OmegaConf
    import json
    from pathlib import Path

    cfg = OmegaConf.load(Path(ckpt_dir) / "config.yaml")
    cfg = OmegaConf.to_container(cfg)

    if train:
        wandb.init(config=cfg, name=cfg["name"], project=project, entity=entity)

        with open(Path(ckpt_dir) / "metrics.jsonl") as f:
            for l in f:
                m = json.loads(l)
                wandb.log(m, step=m["global_step"])

        wandb.finish()

    if eval:
        wandb.init(config=cfg, name=cfg["name"], project=project, entity=entity)

        with open(Path(ckpt_dir) / "metrics.eval.jsonl") as f:
            for l in f:
                m = json.loads(l)
                wandb.log(
                    {
                        f"evals/{name.replace('/','.')}": value
                        for name, value in m.items()
                        if "/" in name
                    },
                    step=m["global_step"],
                )

        wandb.finish()


def get_num_params(model: nn.Module) -> int:
    """
    Get the total model params
    Args : only_trainable: whether to only count trainable params
    """
    numel = {n: p.numel() for n, p in model.named_parameters()}
    return sum(numel.values())
