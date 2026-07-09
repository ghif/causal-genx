import argparse
import gc
import logging
import os
import traceback

import send2trash
import torch

from hps import Hparams
from simple_vae import VAE
from train_setup import (
    derive_save_directories,
    setup_dataloaders,
    setup_directories,
    setup_logging,
    setup_optimizer,
    setup_tensorboard,
)
from trainer import trainer
from utils import EMA, open_file, path_exists, seed_all, select_device, sync_tree
from vae import HVAE
from xla_runtime import NullWriter, is_master, is_xla_device, launch, rank, rendezvous


def main(args: Hparams):
    seed_all(args.seed, args.deterministic)
    # update hyperparams if resuming from a checkpoint
    ckpt = None
    if args.resume:
        if path_exists(args.resume):
            print(f"\nLoading checkpoint: {args.resume}")
            with open_file(args.resume, "rb") as f:
                ckpt = torch.load(f, map_location="cpu")
            ckpt_args = {
                k: v
                for k, v in ckpt["hparams"].items()
                if k not in {"resume", "accelerator", "device"}
            }
            if args.data_dir is not None:
                ckpt_args["data_dir"] = args.data_dir
            if args.lr < ckpt_args["lr"]:
                ckpt_args["lr"] = args.lr
            vars(args).update(ckpt_args)
        else:
            print(f"Checkpoint not found at: {args.resume}")

    args.device = select_device(args.accelerator)

    # load data
    dataloaders = setup_dataloaders(args)

    # init model
    if args.vae == "hierarchical":
        model = HVAE(args)
    elif args.vae == "simple":
        model = VAE(args)
    else:
        NotImplementedError

    def init_bias(m):
        if type(m) == torch.nn.Conv2d:
            torch.nn.init.zeros_(m.bias)

    model.apply(init_bias)
    ema = EMA(model, beta=args.ema_rate)
    ema.ema_model.eval()

    # setup model save directory, logging and tensorboard summaries
    assert args.exp_name != "", "No experiment name given."
    master = not is_xla_device(args.device) or is_master()
    if master:
        args.save_dir = setup_directories(args, ckpt_dir=args.ckpt_dir)
    else:
        args.save_dir = derive_save_directories(args, args.ckpt_dir)
    if is_xla_device(args.device):
        rendezvous("experiment-directories-ready")
    writer = setup_tensorboard(args, model) if master else NullWriter()
    logger = setup_logging(args) if master else logging.getLogger("tpu-worker")

    # setup optimizer
    optimizer, scheduler = setup_optimizer(args, model)

    if args.device.type == "cuda":
        torch.cuda.set_device(args.device)
    model.to(args.device)
    ema.to(args.device)
    if is_xla_device(args.device):
        seed_all(args.seed + rank(), args.deterministic)

    # load checkpoint state dicts
    if ckpt is not None:
        model.load_state_dict(ckpt["model_state_dict"])
        ema.ema_model.load_state_dict(ckpt["ema_model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        for state in optimizer.state.values():
            for k, v in state.items():
                if torch.is_tensor(v):
                    state[k] = v.to(args.device)
        # scheduler.load_state_dict(ckpt['scheduler_state_dict'])
        # update lr of the loaded optimizer
        for p_group in optimizer.param_groups:
            p_group["lr"] = args.lr
            p_group["initial_lr"] = args.lr  # needed to init the scheduler lr
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, lr_lambda=lambda x: x * 0 + 1
        )
        args.start_epoch, args.iter = ckpt["epoch"], ckpt["step"]
        args.best_loss = ckpt["best_loss"]
        del ckpt  # remove reference to checkpoint
    else:
        args.start_epoch, args.iter, args.best_loss = 0, 0, float("inf")

    # train
    try:
        gc.collect()
        if args.device.type == "cuda":
            torch.cuda.empty_cache()
        trainer(args, model, ema, dataloaders, optimizer, scheduler, writer, logger)
    except KeyboardInterrupt:
        print(traceback.format_exc())
        if master and input("Training interrupted, keep logs? [Y/n]: ") == "n":
            if input(f"Send '{args.save_dir}' to Trash? [y/N]: ") == "y":
                send2trash.send2trash(args.save_dir)
                print("Done.")
    finally:
        writer.flush()
        writer.close()
        logging.shutdown()
        if master and hasattr(args, "remote_save_dir"):
            sync_tree(args.save_dir, args.remote_save_dir)


def _tpu_worker(_ordinal: int, args: Hparams):
    main(args)


if __name__ == "__main__":
    from hps import add_arguments, setup_hparams

    parser = argparse.ArgumentParser()
    parser = add_arguments(parser)
    args = setup_hparams(parser)
    if os.environ.get("CAUSAL_GEN_XLA_WORKER") != "1" and (
        args.accelerator == "tpu"
        or (
            args.accelerator == "auto"
            and os.environ.get("PJRT_DEVICE", "").upper() == "TPU"
        )
    ):
        launch(_tpu_worker, args=(args,))
    else:
        main(args)
