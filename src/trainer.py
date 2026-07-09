import copy
import logging
import os
import time
from collections import deque
from typing import Any, Dict

import torch

from utils import ensure_parent_dir, open_file, sync_file, sync_tree
from torch import Tensor, nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from hps import Hparams
from utils import linear_warmup, write_images
from xla_runtime import (
    autocast,
    is_master,
    is_xla_device,
    mark_step,
    optimizer_step,
    reduce_stats,
    save,
    wrap_loader,
)


class ThroughputMeter:
    def __init__(self, window_size: int = 50):
        self.window_size = window_size
        self.total_time = 0.0
        self.total_steps = 0
        self.total_samples = 0
        self._recent = deque(maxlen=window_size)

    def update(self, step_time: float, batch_size: int):
        self.total_time += step_time
        self.total_steps += 1
        self.total_samples += batch_size
        self._recent.append((step_time, batch_size))

    def _recent_totals(self):
        total_time = sum(step_time for step_time, _ in self._recent)
        total_samples = sum(batch_size for _, batch_size in self._recent)
        return total_time, total_samples

    @property
    def avg_steps_per_sec(self):
        return self.total_steps / self.total_time if self.total_time > 0 else 0.0

    @property
    def avg_samples_per_sec(self):
        return self.total_samples / self.total_time if self.total_time > 0 else 0.0

    @property
    def avg_step_time(self):
        return self.total_time / self.total_steps if self.total_steps > 0 else 0.0

    @property
    def window_steps_per_sec(self):
        total_time, _ = self._recent_totals()
        return len(self._recent) / total_time if total_time > 0 else 0.0

    @property
    def window_samples_per_sec(self):
        total_time, total_samples = self._recent_totals()
        return total_samples / total_time if total_time > 0 else 0.0


def preprocess_batch(args: Hparams, batch: Dict[str, Tensor], expand_pa: bool = False):
    non_blocking = args.device.type == "cuda"
    batch["x"] = (
        batch["x"].float().to(args.device, non_blocking=non_blocking) - 127.5
    ) / 127.5  # [-1, 1]
    batch["pa"] = batch["pa"].float().to(args.device, non_blocking=non_blocking)
    if expand_pa:  # used for HVAE parent concatenation
        batch["pa"] = batch["pa"][..., None, None].repeat(1, 1, *(args.input_res,) * 2)
    return batch


def trainer(
    args: Hparams,
    model: nn.Module,
    ema: nn.Module,
    dataloaders: Dict[str, DataLoader],
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    writer: SummaryWriter,
    logger: logging.Logger,
):
    for k in sorted(vars(args)):
        logger.info(f"--{k}={vars(args)[k]}")
    logger.info(f"total params: {sum(p.numel() for p in model.parameters()):,}")

    def run_epoch(dataloader: DataLoader, training: bool = True):
        model.train(training)
        model.zero_grad(set_to_none=True)
        stats = {k: 0.0 for k in ["elbo", "nll", "kl", "beta", "grad_norm"]}
        stats["n"] = 0
        stats["beta_n"] = 0
        stats["grad_norm_n"] = 0
        updates_skipped = 0
        meter = ThroughputMeter(window_size=max(1, getattr(args, "speed_log_freq", 50)))
        last_logged_step = 0

        mininterval = 300 if "SLURM_JOB_ID" in os.environ else 0.1
        if training and hasattr(dataloader.sampler, "set_epoch"):
            dataloader.sampler.set_epoch(args.epoch)
        device_loader = wrap_loader(dataloader, args.device)
        loader = tqdm(
            enumerate(device_loader),
            total=len(dataloader),
            mininterval=mininterval,
            disable=is_xla_device(args.device) and not is_master(),
        )

        for i, batch in loader:
            step_start = time.perf_counter()
            batch = preprocess_batch(args, batch, expand_pa=args.expand_pa)
            bs = batch["x"].shape[0]
            update_stats = True
            beta_value = args.beta
            grad_norm = None

            if training:
                args.iter = i + 1 + (args.epoch - 1) * len(dataloader)
                if args.beta_warmup_steps > 0:
                    beta_value = args.beta_target * linear_warmup(
                        args.beta_warmup_steps
                    )(args.iter)
                    args.beta = beta_value

                with autocast(args.device, args.precision):
                    out = model(batch["x"], batch["pa"], beta=beta_value)
                out["elbo"] = out["elbo"] / args.accu_steps
                out["elbo"].backward()

                if i % args.accu_steps == 0:  # gradient accumulation update
                    grad_norm = nn.utils.clip_grad_norm_(
                        model.parameters(), args.grad_clip
                    )
                    writer.add_scalar("train/grad_norm", grad_norm, args.iter)
                    nll_nan = torch.isnan(out["nll"]).sum()
                    kl_nan = torch.isnan(out["kl"]).sum()

                    if grad_norm < args.grad_skip and nll_nan == 0 and kl_nan == 0:
                        optimizer_step(optimizer, args.device)
                        scheduler.step()
                        ema.update()
                    else:
                        updates_skipped += 1
                        update_stats = False
                        logger.info(
                            f"Updates skipped: {updates_skipped}"
                            + f" - grad_norm: {grad_norm:.3f}"
                            + f" - nll_nan: {nll_nan.item()} - kl_nan: {kl_nan.item()}"
                        )

                    model.zero_grad(set_to_none=True)
            else:
                with torch.no_grad():
                    with autocast(args.device, args.precision):
                        out = ema.ema_model(batch["x"], batch["pa"], beta=args.beta)

            mark_step(args.device)

            step_time = time.perf_counter() - step_start
            meter.update(step_time, bs)
            if update_stats:
                if training:
                    out["elbo"] *= args.accu_steps
                stats["n"] += bs  # samples seen counter
                stats["elbo"] += out["elbo"].detach() * bs
                stats["nll"] += out["nll"].detach() * bs
                stats["kl"] += out["kl"].detach() * bs
                if training:
                    stats["beta"] += float(beta_value)
                    stats["beta_n"] += 1
                    if grad_norm is not None:
                        stats["grad_norm"] += float(grad_norm)
                        stats["grad_norm_n"] += 1

            if training and (
                (i + 1) == len(dataloader)
                or (args.speed_log_freq > 0 and (i + 1) % args.speed_log_freq == 0)
            ):
                loader.set_postfix(
                    {
                        "it/s": f"{meter.window_steps_per_sec:.2f}",
                        "samples/s": f"{meter.window_samples_per_sec:.1f}",
                        "step": f"{meter.total_steps}",
                    },
                    refresh=False,
                )
                last_logged_step = i + 1
            elif not training and (i + 1) == len(dataloader):
                loader.set_postfix(
                    {
                        "it/s": f"{meter.avg_steps_per_sec:.2f}",
                        "samples/s": f"{meter.avg_samples_per_sec:.1f}",
                    },
                    refresh=False,
                )
        stats["updates_skipped"] = updates_skipped
        stats = reduce_stats(stats, args.device)
        result = {}
        for k, v in stats.items():
            if k in {"n", "beta_n", "grad_norm_n", "updates_skipped"}:
                continue
            if k in {"beta", "grad_norm"}:
                denom = stats[f"{k}_n"]
                result[k] = float(v / denom) if denom > 0 else 0.0
            else:
                result[k] = float(v / stats["n"]) if stats["n"] > 0 else 0.0
        result["steps_per_sec"] = meter.avg_steps_per_sec
        result["samples_per_sec"] = meter.avg_samples_per_sec
        result["seconds_per_step"] = meter.avg_step_time
        result["total_time"] = meter.total_time
        result["skipped_updates"] = int(float(stats["updates_skipped"]))
        result["logged_step"] = last_logged_step
        return result

    if args.beta_warmup_steps > 0:
        args.beta_target = copy.deepcopy(args.beta)

    # expand pa to input res, used for HVAE parent concatenation
    args.expand_pa = args.vae == "hierarchical"

    def run_epoch_artifacts(epoch: int, valid_stats: Dict[str, float]):
        n = min(args.context_dim * 5, args.bs)
        if is_xla_device(args.device) and not is_master():
            return
        viz_batch = next(iter(dataloaders["valid"]))
        viz_batch = {k: v[:n] for k, v in viz_batch.items()}
        viz_batch = preprocess_batch(args, viz_batch, expand_pa=args.expand_pa)

        with torch.no_grad():
            write_images(args, ema.ema_model, viz_batch)

        logger.info(
            f'=> valid | nelbo: {valid_stats["elbo"]:.4f}'
            + f' - nll: {valid_stats["nll"]:.4f} - kl: {valid_stats["kl"]:.4f}'
            + f" - steps: {args.iter}"
        )

        if valid_stats["elbo"] < args.best_loss:
            args.best_loss = valid_stats["elbo"]
            save_dict = {
                "epoch": args.epoch,
                "step": args.epoch * len(dataloaders["train"]),
                "best_loss": float(args.best_loss),
                "model_state_dict": model.state_dict(),
                "ema_model_state_dict": ema.ema_model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "hparams": vars(args),
            }
            ckpt_path = os.path.join(args.save_dir, "checkpoint.pt")
            ensure_parent_dir(ckpt_path)
            if is_xla_device(args.device):
                save(save_dict, ckpt_path, args.device)
            else:
                with open_file(ckpt_path, "wb") as f:
                    save(save_dict, f, args.device)
            sync_file(ckpt_path, os.path.join(args.remote_save_dir, "checkpoint.pt"))
            logger.info(f"Model saved: {ckpt_path}")

    # Start training loop
    master = not is_xla_device(args.device) or is_master()
    for epoch in range(args.start_epoch, args.epochs):
        args.epoch = epoch + 1
        logger.info(f"Epoch {args.epoch}:")

        stats = run_epoch(dataloaders["train"], training=True)

        if master:
            writer.add_scalar(f"nelbo/train", stats["elbo"], args.epoch)
            writer.add_scalar(f"nll/train", stats["nll"], args.epoch)
            writer.add_scalar(f"kl/train", stats["kl"], args.epoch)
            writer.add_scalar("train/steps_per_sec", stats["steps_per_sec"], args.epoch)
            writer.add_scalar("train/samples_per_sec", stats["samples_per_sec"], args.epoch)
            writer.add_scalar("train/seconds_per_step", stats["seconds_per_step"], args.epoch)
            writer.add_scalar("train/total_time", stats["total_time"], args.epoch)
            if args.beta_warmup_steps > 0:
                writer.add_scalar("train/beta_kl", stats["beta"], args.epoch)
            writer.add_scalar("train/grad_norm", stats["grad_norm"], args.epoch)
            logger.info(
                f'=> train | nelbo: {stats["elbo"]:.4f}'
                + f' - nll: {stats["nll"]:.4f} - kl: {stats["kl"]:.4f}'
                + f" - steps: {args.iter}"
                + f" - it/s: {stats['steps_per_sec']:.2f}"
                + f" - samples/s: {stats['samples_per_sec']:.1f}"
                + f" - step s: {stats['seconds_per_step']:.4f}"
                + f" - skipped: {stats['skipped_updates']}"
            )

        if (args.epoch - 1) % args.eval_freq == 0:
            valid_stats = run_epoch(dataloaders["valid"], training=False)

            if master:
                writer.add_scalar(f"nelbo/valid", valid_stats["elbo"], args.epoch)
                writer.add_scalar(f"nll/valid", valid_stats["nll"], args.epoch)
                writer.add_scalar(f"kl/valid", valid_stats["kl"], args.epoch)
                writer.add_scalar("valid/steps_per_sec", valid_stats["steps_per_sec"], args.epoch)
                writer.add_scalar("valid/samples_per_sec", valid_stats["samples_per_sec"], args.epoch)
                writer.add_scalar("valid/seconds_per_step", valid_stats["seconds_per_step"], args.epoch)
                writer.add_scalar("valid/total_time", valid_stats["total_time"], args.epoch)
            run_epoch_artifacts(args.epoch, valid_stats)
        if master and hasattr(args, "remote_save_dir"):
            writer.flush()
            sync_tree(args.save_dir, args.remote_save_dir)
    return
