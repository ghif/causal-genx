import logging
import os
from typing import Any, Dict, Tuple

import send2trash
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter

from datasets import cmnist, mimic, morphomnist, ukbb
from hps import Hparams
from utils import (
    ensure_dir,
    is_remote_path,
    linear_warmup,
    local_staging_path,
    path_exists,
    remove_path,
    seed_worker,
)


def setup_dataloaders(args: Hparams) -> Dict[str, DataLoader]:
    cpu_count = os.cpu_count() or 1

    if args.num_workers >= 0:
        num_workers = args.num_workers
    elif args.device.type in {"cuda", "xla"}:
        num_workers = min(8, max(2, cpu_count // 2))
    else:
        num_workers = min(4, max(0, cpu_count // 4))

    if args.pin_memory == "auto":
        pin_memory = args.device.type == "cuda"
    else:
        pin_memory = args.pin_memory == "true"

    if args.persistent_workers == "auto":
        persistent_workers = args.device.type in {"cuda", "xla"} and num_workers > 0
    else:
        persistent_workers = args.persistent_workers == "true"

    if args.prefetch_factor > 0:
        prefetch_factor = args.prefetch_factor
    else:
        prefetch_factor = 4 if args.device.type in {"cuda", "xla"} else 2

    if "ukbb" in args.hps:
        datasets = ukbb(args)
    elif "morphomnist" in args.hps:
        datasets = morphomnist(args)
    elif "cmnist" in args.hps:
        datasets = cmnist(args)
    elif "mimic" in args.hps:
        datasets = mimic(args)
    else:
        NotImplementedError

    kwargs = {
        "batch_size": args.bs,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
        "worker_init_fn": seed_worker,
    }
    if num_workers > 0:
        kwargs["persistent_workers"] = persistent_workers
        kwargs["prefetch_factor"] = prefetch_factor
    samplers = {"train": None, "valid": None, "test": None}
    if args.device.type == "xla":
        from xla_runtime import rank, world_size

        for split in samplers:
            samplers[split] = DistributedSampler(
                datasets[split],
                num_replicas=world_size(),
                rank=rank(),
                shuffle=split == "train",
                drop_last=split == "train",
            )
    dataloaders = {}
    for split in ("train", "valid", "test"):
        dataloaders[split] = DataLoader(
            datasets[split],
            shuffle=split == "train" and samplers[split] is None,
            sampler=samplers[split],
            drop_last=split == "train",
            **kwargs,
        )
    return dataloaders


def setup_optimizer(
    args: Hparams, model: nn.Module
) -> Tuple[torch.optim.Optimizer, Any]:
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.wd, betas=args.betas
    )

    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda=linear_warmup(args.lr_warmup_steps)
    )

    return optimizer, scheduler


def setup_directories(args: Hparams, ckpt_dir: str = "../checkpoints") -> str:
    parents_folder = "_".join([k[0] for k in args.parents_x])
    args.parents_folder = parents_folder
    remote_save_dir = os.path.join(ckpt_dir, parents_folder, args.exp_name)
    args.remote_save_dir = remote_save_dir
    save_dir = remote_save_dir
    if is_remote_path(remote_save_dir):
        save_dir = local_staging_path(remote_save_dir)

    if path_exists(remote_save_dir):
        if (
            input(
                f"\nSave directory '{remote_save_dir}' already exists, overwrite? [y/N]: "
            )
            == "y"
        ):
            if is_remote_path(remote_save_dir):
                remove_path(remote_save_dir)
                print("Done.\n")
            else:
                if input(f"Send '{save_dir}', to Trash? [y/N]: ") == "y":
                    send2trash.send2trash(save_dir)
                    print("Done.\n")
                else:
                    exit()
        else:
            if (
                input(
                    f"\nResume training with save directory '{remote_save_dir}'? [y/N]: "
                )
                == "y"
            ):
                pass
            else:
                exit()
    ensure_dir(save_dir)
    if is_remote_path(remote_save_dir):
        ensure_dir(remote_save_dir)
    return save_dir


def derive_save_directories(args: Hparams, ckpt_dir: str) -> str:
    parents_folder = "_".join([k[0] for k in args.parents_x])
    args.parents_folder = parents_folder
    args.remote_save_dir = os.path.join(ckpt_dir, parents_folder, args.exp_name)
    if is_remote_path(args.remote_save_dir):
        return local_staging_path(args.remote_save_dir)
    return args.remote_save_dir


def setup_tensorboard(args: Hparams, model: nn.Module) -> SummaryWriter:
    """Setup metric summary writer."""
    writer = SummaryWriter(args.save_dir)

    hparams = {}
    for k, v in vars(args).items():
        if isinstance(v, list) or isinstance(v, torch.device):
            hparams[k] = str(v)
        elif isinstance(v, torch.Tensor):
            hparams[k] = v.item()
        else:
            hparams[k] = v

    writer.add_hparams(hparams, {"hparams": 0}, run_name=os.path.abspath(args.save_dir))

    if "vae" in type(model).__name__.lower():
        z_str = []
        if hasattr(model.decoder, "blocks"):
            for i, block in enumerate(model.decoder.blocks):
                if block.stochastic:
                    z_str.append(f"z{i}_{block.res}x{block.res}")
        else:
            z_str = ["z0_" + str(args.z_dim)]

        writer.add_custom_scalars(
            {
                "nelbo": {"nelbo": ["Multiline", ["nelbo/train", "nelbo/valid"]]},
                "nll": {"kl": ["Multiline", ["nll/train", "nll/valid"]]},
                "kl": {"kl": ["Multiline", ["kl/train", "kl/valid"]]}
                # "KL": {
                #     "KL_train": ["Multiline", ['KL_train/'+z[:2] for z in z_str]],
                #     "KL_valid": ["Multiline", ['KL_valid/'+z[:2] for z in z_str]]
                # }
            }
        )
    return writer


def setup_logging(args: Hparams) -> logging.Logger:
    # reset root logger
    [logging.root.removeHandler(h) for h in logging.root.handlers[:]]
    # info logger for saving command line outputs during training
    logging.basicConfig(
        handlers=[
            logging.FileHandler(os.path.join(args.save_dir, "trainlog.txt"), mode="a"),
            logging.StreamHandler(),
        ],
        # filemode='a',  # append to file, 'w' for overwrite
        format="%(asctime)s, %(message)s",
        datefmt="%d-%b-%y %H:%M:%S",
        level=logging.INFO,
    )
    logger = logging.getLogger(args.exp_name)  # name the logger
    return logger
