import argparse
import copy
import logging
import os
import sys
from typing import Any, Dict, Optional

import matplotlib.pyplot as plt
import numpy as np
import pyro
import torch
from layers import TraceStorage_ELBO
from sklearn.metrics import roc_auc_score
from torch import Tensor, nn
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm
from utils_pgm import plot_joint, update_stats

sys.path.append("..")
from datasets import cmnist, get_attr_max_min, mimic, morphomnist, ukbb
from hps import Hparams
from train_setup import (
    derive_save_directories,
    setup_directories,
    setup_logging,
    setup_tensorboard,
)
from utils import (
    EMA,
    ensure_parent_dir,
    open_file,
    path_exists,
    seed_all,
    seed_worker,
    select_device,
    sync_file,
    sync_tree,
)
from xla_runtime import (
    NullWriter,
    is_master,
    is_xla_device,
    optimizer_step,
    rank,
    rendezvous,
    save,
    world_size,
)


def _to_device(value: Tensor, device: torch.device) -> Tensor:
    return value.to(device, non_blocking=device.type == "cuda")


def preprocess(
    batch: Dict[str, Tensor],
    dataset: str = "ukbb",
    split: str = "l",
    device: torch.device = torch.device("cpu"),
) -> Dict[str, Tensor]:
    if "x" in batch.keys():
        batch["x"] = (_to_device(batch["x"].float(), device) - 127.5) / 127.5  # [-1,1]
    # for all other variables except x
    not_x = [k for k in batch.keys() if k != "x"]
    for k in not_x:
        if split == "u":  # unlabelled
            batch[k] = None
        elif split == "l":  # labelled
            batch[k] = _to_device(batch[k].float(), device)
            if len(batch[k].shape) < 2:
                batch[k] = batch[k].unsqueeze(-1)
        else:
            NotImplementedError
    if "ukbb" in dataset:
        for k in not_x:
            if k in ["age", "brain_volume", "ventricle_volume"]:
                k_max, k_min = get_attr_max_min(k)
                batch[k] = (batch[k] - k_min) / (k_max - k_min)  # [0,1]
                batch[k] = 2 * batch[k] - 1  # [-1,1]
    return batch


def resolve_loader_settings(args: Hparams) -> Dict[str, Any]:
    cpu_count = os.cpu_count() or 1

    if args.num_workers >= 0:
        num_workers = args.num_workers
    elif args.device.type == "cuda":
        num_workers = min(8, max(2, cpu_count // 2))
    else:
        num_workers = min(4, max(0, cpu_count // 4))

    if args.pin_memory == "auto":
        pin_memory = args.device.type == "cuda"
    else:
        pin_memory = args.pin_memory == "true"

    if args.persistent_workers == "auto":
        persistent_workers = args.device.type == "cuda" and num_workers > 0
    else:
        persistent_workers = args.persistent_workers == "true"

    if args.prefetch_factor > 0:
        prefetch_factor = args.prefetch_factor
    else:
        prefetch_factor = 4 if args.device.type == "cuda" else 2

    loader_kwargs = {
        "batch_size": args.bs,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
        "worker_init_fn": seed_worker,
    }
    if num_workers > 0:
        loader_kwargs["persistent_workers"] = persistent_workers
        loader_kwargs["prefetch_factor"] = prefetch_factor
    return loader_kwargs


def ss_train_epoch(
    args: Hparams,
    model: nn.Module,
    ema: nn.Module,
    dataloaders: Dict[str, DataLoader],
    elbo_fn: TraceStorage_ELBO,
    aux_elbo_fn: TraceStorage_ELBO,
    optimizer: torch.optim.Optimizer,
) -> Dict[str, Any]:
    "semi-supervised training epoch"
    stats = {"loss": 0, "aux_loss": 0, "n": 0}  # sample counter
    alpha = args.alpha * len(dataloaders["l"].dataset)

    # outer loop over largest set, (u) unlabelled or (l) labelled
    if len(dataloaders["u"]) > len(dataloaders["l"]):
        outer, inner = "u", "l"
    else:
        outer, inner = "l", "u"
    iter_outer = iter(dataloaders[outer])
    iter_inner = iter(dataloaders[inner])
    loader = tqdm(range(len(iter_outer)))

    model.train()
    for _ in loader:
        batch = {}
        batch[outer] = next(iter_outer)
        batch[outer] = preprocess(
            batch[outer], args.dataset, split=outer, device=args.device
        )

        try:
            batch[inner] = next(iter_inner)
        except StopIteration:
            iter_inner = iter(dataloaders[inner])  # restart inner iterator
            batch[inner] = next(iter_inner)
        batch[inner] = preprocess(
            batch[inner], args.dataset, split=inner, device=args.device
        )

        # supervised update
        loss = elbo_fn(model.svi_model, model.guide, **batch["l"])
        # unsupervised update
        loss = loss + elbo_fn(model.svi_model, model.guide, **batch["u"])
        # aux supervised update
        aux_loss = aux_elbo_fn(model.model_anticausal, model.guide_pass, **batch["l"])
        loss = loss + alpha * aux_loss
        optimizer.zero_grad()
        loss.backward()
        optimizer_step(optimizer, args.device)
        ema.update()

        stats["loss"] += loss.item()
        stats["aux_loss"] += aux_loss.item()
        stats["n"] += batch[outer]["x"].shape[0]

        loader.set_description(
            f' => train | -elbo: {stats["loss"] / stats["n"]:.4f}'
            + f' - aux_loss: {stats["aux_loss"] / stats["n"] * alpha:.4f}'
        )

    stats = {k: v / stats["n"] for k, v in stats.items() if k != "n"}
    stats["aux_loss"] *= alpha
    return stats


def sup_epoch(
    args: Hparams,
    model: nn.Module,
    ema: Optional[nn.Module],
    dataloader: Dict[str, DataLoader],
    elbo_fn: TraceStorage_ELBO,
    optimizer: Optional[torch.optim.Optimizer] = None,
    is_train: bool = True,
) -> Dict[str, Any]:
    "supervised epoch"
    stats = {"loss": 0, "n": 0}  # sample counter
    loader = tqdm(
        enumerate(dataloader),
        total=len(dataloader),
        miniters=len(dataloader) // 100,
        mininterval=5,
    )

    model.train(is_train)
    for i, batch in loader:
        bs = batch["x"].shape[0]
        batch = preprocess(batch, args.dataset, split="l", device=args.device)

        with torch.set_grad_enabled(is_train):
            if args.setup == "sup_aux":
                loss = (
                    elbo_fn.differentiable_loss(
                        model.model_anticausal, model.guide_pass, **batch
                    )
                    / bs
                )
            elif args.setup == "sup_pgm":
                loss = (
                    elbo_fn.differentiable_loss(
                        model.svi_model, model.guide_pass, **batch
                    )
                    / bs
                )
            else:
                NotImplementedError

        if is_train:
            optimizer.zero_grad()
            loss.backward()
            grad_norm = nn.utils.clip_grad_norm_(model.parameters(), 200)
            optimizer_step(optimizer, args.device)
            ema.update()

        stats["loss"] += loss.item() * bs
        stats["n"] += bs
        stats = update_stats(stats, elbo_fn)
        loader.set_description(
            f' => {("train" if is_train else "eval")} | '
            + f", ".join(
                f'{k}: {v / stats["n"]:.4f}' for k, v in stats.items() if k != "n"
            )
            +
            # ', probs: ' + f', '.join(f'{v:.4f}' for v in F.softmax(model.digit_probs.data, dim=-1).squeeze().tolist()) +
            (f", grad_norm: {grad_norm:.3f}" if is_train else ""),  # refresh=False
        )
    return {k: v / stats["n"] for k, v in stats.items() if k != "n"}


@torch.no_grad()
def eval_epoch(
    args: Hparams, model: nn.Module, dataloader: DataLoader
) -> Dict[str, float]:
    "caution: this can consume lots of memory if dataset is large"
    model.eval()
    preds = {k: [] for k in model.variables.keys()}
    targets = {k: [] for k in model.variables.keys()}

    for batch in tqdm(dataloader):
        for k in targets.keys():
            targets[k].extend(copy.deepcopy(batch[k]))
        # predict
        batch = preprocess(batch, args.dataset, split="l", device=args.device)
        out = model.predict(**batch)

        for k, v in out.items():
            preds[k].extend(v)

    for k, v in preds.items():
        preds[k] = torch.stack(v).squeeze().cpu()
        targets[k] = torch.stack(targets[k]).squeeze()
    stats = {}
    for k in model.variables.keys():
        if "ukbb" in args.dataset:
            if k == "mri_seq" or k == "sex":
                stats[k + "_rocauc"] = roc_auc_score(
                    targets[k].numpy(), preds[k].numpy(), average="macro"
                )
                stats[k + "_acc"] = (
                    targets[k] == torch.round(preds[k])
                ).sum().item() / targets[k].shape[0]
            else:  # continuous variables
                preds_k = (preds[k] + 1) / 2  # [-1,1] -> [0,1]
                _max, _min = get_attr_max_min(k)
                preds_k = preds_k * (_max - _min) + _min
                norm = 1000 if "volume" in k else 1  # for volume in ml
                stats[k + "_mae"] = (targets[k] - preds_k).abs().mean().item() / norm
        elif args.dataset == "morphomnist":
            if k == "digit":
                num_corrects = (targets[k].argmax(-1) == preds[k].argmax(-1)).sum()
                stats[k + "_acc"] = num_corrects.item() / targets[k].shape[0]
            else:  # continuous variables
                # unormalize from [-1,1] back to original range
                min_max = dataloader.dataset.min_max[k]
                _min, _max = min_max[0], min_max[1]
                preds_k = ((preds[k] + 1) / 2) * (_max - _min) + _min
                targets_k = ((targets[k] + 1) / 2) * (_max - _min) + _min
                stats[k + "_mae"] = (targets_k - preds_k).abs().mean().item()
        elif args.dataset == "cmnist":
            num_corrects = (targets[k].argmax(-1) == preds[k].argmax(-1)).sum()
            stats[k + "_acc"] = num_corrects.item() / targets[k].shape[0]
        elif "mimic" in args.dataset:
            if k in ["sex", "finding"]:
                stats[k + "_rocauc"] = roc_auc_score(
                    targets[k].numpy(), preds[k].numpy(), average="macro"
                )
                stats[k + "_acc"] = (
                    targets[k] == torch.round(preds[k])
                ).sum().item() / targets[k].shape[0]
            elif k == "age":
                preds_k = (preds[k] + 1) * 50  # unormalize
                targets_k = (targets[k] + 1) * 50  # unormalize
                stats[k + "_mae"] = (targets_k - preds_k).abs().mean().item()
            elif k == "race":
                num_corrects = (targets[k].argmax(-1) == preds[k].argmax(-1)).sum()
                stats[k + "_acc"] = num_corrects.item() / targets[k].shape[0]
                stats[k + "_rocauc"] = roc_auc_score(
                    targets[k].numpy(),
                    preds[k].numpy(),
                    multi_class="ovr",
                    average="macro",
                )
        else:
            NotImplementedError
    return stats


def setup_dataloaders(args: Hparams) -> Dict[str, DataLoader]:
    if "ukbb" in args.dataset:
        datasets = ukbb(args)
    elif args.dataset == "morphomnist":
        assert args.input_channels == 1
        assert args.input_res == 32
        assert args.pad == 4
        args.parents_x = ["thickness", "intensity", "digit"]
        args.context_norm = "[-1,1]"
        args.concat_pa = False
        datasets = morphomnist(args)
    elif args.dataset == "cmnist":
        assert args.input_channels == 3
        assert args.input_res == 32
        assert args.pad == 4
        args.parents_x = ["digit", "colour"]
        args.concat_pa = False
        datasets = cmnist(args)
    elif args.dataset == "mimic":
        datasets = mimic(args)
    else:
        NotImplementedError

    kwargs = resolve_loader_settings(args)

    def make_loader(dataset, shuffle=False, drop_last=False):
        sampler = None
        if args.device.type == "xla":
            sampler = DistributedSampler(
                dataset,
                num_replicas=world_size(),
                rank=rank(),
                shuffle=shuffle,
                drop_last=drop_last,
            )
        return DataLoader(
            dataset,
            shuffle=shuffle and sampler is None,
            sampler=sampler,
            drop_last=drop_last,
            **kwargs,
        )

    dataloaders = {}
    if args.setup == "sup_pgm":
        dataloaders["train"] = make_loader(
            datasets["train"], shuffle=True, drop_last=True
        )
    else:
        args.n_total = len(datasets["train"])
        args.n_labelled = int(args.sup_frac * args.n_total)
        args.n_unlabelled = args.n_total - args.n_labelled
        idx = np.arange(args.n_total)
        rng = np.random.RandomState(1)
        rng.shuffle(idx)
        train_l = torch.utils.data.Subset(datasets["train"], idx[: args.n_labelled])

        if args.setup == "semi_sup":
            train_u = torch.utils.data.Subset(datasets["train"], idx[args.n_labelled :])
            dataloaders["train_l"] = make_loader(
                train_l, shuffle=True, drop_last=True
            )
            dataloaders["train_u"] = make_loader(
                train_u, shuffle=True, drop_last=True
            )
        elif args.setup == "sup_aux":
            dataloaders["train"] = make_loader(
                train_l, shuffle=True, drop_last=True
            )

    dataloaders["valid"] = make_loader(datasets["valid"])
    dataloaders["test"] = make_loader(datasets["test"])
    return dataloaders


def move_optimizer_state_to_device(
    optimizer: torch.optim.Optimizer, device: torch.device
) -> None:
    for state in optimizer.state.values():
        for key, value in state.items():
            if torch.is_tensor(value):
                state[key] = value.to(device)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--accelerator",
        help="Training accelerator.",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda", "mps", "tpu"],
    )
    parser.add_argument("--exp_name", help="Experiment name.", type=str, default="")
    parser.add_argument("--dataset", help="Dataset name.", type=str, default="ukbb")
    parser.add_argument(
        "--data_dir", help="Data directory to load form.", type=str, default=""
    )
    parser.add_argument(
        "--ckpt_dir",
        help="Directory to store checkpoints.",
        type=str,
        default="gs://medical-airnd/causal-gen/checkpoints",
    )
    parser.add_argument(
        "--load_path", help="Path to load checkpoint.", type=str, default=""
    )
    parser.add_argument(
        "--setup",  # semi_sup/sup_pgm/sup_aux
        help="training setup.",
        type=str,
        default="sup_pgm",
    )
    parser.add_argument("--seed", help="Set random seed.", type=int, default=7)
    parser.add_argument(
        "--deterministic",
        help="Toggle cudNN determinism.",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--testing", help="Test model.", action="store_true", default=False
    )
    # training
    parser.add_argument(
        "--epochs", help="Number of training epochs.", type=int, default=1000
    )
    parser.add_argument("--bs", help="Batch size.", type=int, default=32)
    parser.add_argument(
        "--num_workers",
        help="DataLoader workers; use -1 for an accelerator-aware default.",
        type=int,
        default=-1,
    )
    parser.add_argument(
        "--pin_memory",
        help="Pin host memory for faster device transfers: auto/true/false.",
        type=str,
        default="auto",
        choices=["auto", "true", "false"],
    )
    parser.add_argument(
        "--persistent_workers",
        help="Keep DataLoader workers alive across epochs: auto/true/false.",
        type=str,
        default="auto",
        choices=["auto", "true", "false"],
    )
    parser.add_argument(
        "--prefetch_factor",
        help="DataLoader prefetch factor; use -1 for an accelerator-aware default.",
        type=int,
        default=-1,
    )
    parser.add_argument("--lr", help="Learning rate.", type=float, default=1e-4)
    parser.add_argument(
        "--lr_warmup_steps", help="lr warmup steps.", type=int, default=1
    )
    parser.add_argument("--wd", help="Weight decay penalty.", type=float, default=0.1)
    parser.add_argument(
        "--input_res", help="Input image crop resolution.", type=int, default=192
    )
    parser.add_argument(
        "--input_channels", help="Input image num channels.", type=int, default=1
    )
    parser.add_argument("--pad", help="Input padding.", type=int, default=9)
    parser.add_argument(
        "--hflip", help="Horizontal flip prob.", type=float, default=0.5
    )
    parser.add_argument(
        "--sup_frac", help="Labelled data fraction.", type=float, default=1
    )
    parser.add_argument("--eval_freq", help="Num epochs per eval.", type=int, default=1)
    # model
    parser.add_argument(
        "--widths",
        help="Cond flow fc network width per layer.",
        nargs="+",
        type=int,
        default=[32, 32],
    )
    parser.add_argument(
        "--parents_x", help="Parents of x to load.", nargs="+", default=[]
    )
    parser.add_argument(
        "--alpha",  # for semi_sup learning only
        help="aux loss multiplier.",
        type=float,
        default=1e-3,
    )
    parser.add_argument(
        "--std_fixed", help="Fix aux dist std value (0 is off).", type=float, default=0
    )
    args = parser.parse_known_args()[0]

    seed_all(args.seed, args.deterministic)
    args.device = select_device(args.accelerator)

    # update hparams if loading checkpoint
    if args.load_path:
        if path_exists(args.load_path):
            print(f"\nLoading checkpoint: {args.load_path}")
            with open_file(args.load_path, "rb") as f:
                ckpt = torch.load(f, map_location="cpu")
            ckpt_args = {
                k: v
                for k, v in ckpt["hparams"].items()
                if k not in {"load_path", "accelerator", "device"}
            }
            if args.data_dir is not None:
                ckpt_args["data_dir"] = args.data_dir
            if args.testing:
                ckpt_args["testing"] = args.testing
            vars(args).update(ckpt_args)
        else:
            print(f"Checkpoint not found at: {args.load_path}")

    # Load data
    dataloaders = setup_dataloaders(args)

    # Init model
    pyro.clear_param_store()
    if "ukbb" in args.dataset:
        from flow_pgm import FlowPGM

        model = FlowPGM(args)
    elif args.dataset == "morphomnist":
        from flow_pgm import MorphoMNISTPGM

        model = MorphoMNISTPGM(args)
    elif args.dataset == "cmnist":
        from flow_pgm import ColourMNISTPGM

        model = ColourMNISTPGM(args)
    else:
        NotImplementedError
    ema = EMA(model, beta=0.999)
    model.to(args.device)
    ema.to(args.device)

    # Init loss & optimizer
    elbo_fn = TraceStorage_ELBO(num_particles=2)
    aux_elbo_fn = TraceStorage_ELBO(num_particles=2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)

    if not args.testing:
        # Train model
        master = not is_xla_device(args.device) or is_master()
        if master:
            args.save_dir = setup_directories(args, ckpt_dir=args.ckpt_dir)
        else:
            args.save_dir = derive_save_directories(args, args.ckpt_dir)
        if is_xla_device(args.device):
            rendezvous("pgm-experiment-directories-ready")
        writer = setup_tensorboard(args, model) if master else NullWriter()
        logger = setup_logging(args) if master else logging.getLogger("tpu-pgm-worker")

        for k in sorted(vars(args)):
            logger.info(f"--{k}={vars(args)[k]}")
        if args.setup != "sup_pgm":
            logger.info(
                f"Data splits: #labelled: {args.n_labelled}"
                + f" - #unlabelled: {args.n_unlabelled}"
            )
        args.best_loss = float("inf")

        for epoch in range(args.epochs):
            for dataloader in dataloaders.values():
                if hasattr(dataloader.sampler, "set_epoch"):
                    dataloader.sampler.set_epoch(epoch)
            logger.info(f"Epoch {epoch+1}:")

            # semi supervised training
            if args.setup == "semi_sup":
                stats = ss_train_epoch(
                    args,
                    model,
                    ema,
                    {"l": dataloaders["train_l"], "u": dataloaders["train_u"]},
                    elbo_fn,
                    aux_elbo_fn,
                    optimizer,
                )
                # valid aux loss on labelled data only
                if epoch % args.eval_freq == 0:
                    valid_stats = sup_epoch(
                        args,
                        ema.ema_model,
                        None,
                        dataloaders["valid"],
                        elbo_fn,
                        is_train=False,
                    )
                    steps = (epoch + 1) * max(
                        len(dataloaders["train_l"]), len(dataloaders["train_u"])
                    )
                    stats["aux_loss"] *= args.alpha * len(
                        dataloaders["train_l"].dataset
                    )

                    logger.info(
                        f'loss: {stats["loss"]:.4f}'
                        + f' - aux_loss: {stats["aux_loss"]:.4f}'
                        + f' - valid_aux_loss: {valid_stats["aux_loss"]:.4f} - steps: {steps}'
                    )
            # supervised training of PGM or aux models
            elif args.setup == "sup_pgm" or args.setup == "sup_aux":
                stats = sup_epoch(
                    args,
                    model,
                    ema,
                    dataloaders["train"],
                    elbo_fn,
                    optimizer,
                    is_train=True,
                )
                if epoch % args.eval_freq == 0:
                    valid_stats = sup_epoch(
                        args,
                        ema.ema_model,
                        None,
                        dataloaders["valid"],
                        elbo_fn,
                        is_train=False,
                    )
                    steps = (epoch + 1) * len(dataloaders["train"])
                    if master and args.setup == "sup_pgm":
                        plot_joint(
                            args, ema.ema_model, dataloaders["train"].dataset, steps
                        )

                    logger.info(
                        f'loss | train: {stats["loss"]:.4f}'
                        + f' - valid: {valid_stats["loss"]:.4f} - steps: {steps}'
                    )

                    for k, v in stats.items():
                        writer.add_scalar("train/" + k, v, steps)
                        writer.add_scalar("valid/" + k, valid_stats[k], steps)

                    writer.add_custom_scalars(
                        {"elbo": {"elbo": ["Multiline", ["elbo/train", "elbo/valid"]]}}
                    )
                    writer.add_scalar("elbo/train", stats["loss"], steps)
                    writer.add_scalar("elbo/valid", valid_stats["loss"], steps)
            else:
                NotImplementedError

            if epoch % args.eval_freq == 0:
                if not args.setup == "sup_pgm":  # eval aux classifiers
                    metrics = eval_epoch(args, ema.ema_model, dataloaders["valid"])
                    logger.info(
                        "valid | "
                        + " - ".join(f"{k}: {v:.4f}" for k, v in metrics.items())
                    )

            if master and valid_stats["loss"] < args.best_loss:
                args.best_loss = valid_stats["loss"]
                ckpt_path = os.path.join(args.save_dir, "checkpoint.pt")
                ensure_parent_dir(ckpt_path)
                checkpoint = {
                    "epoch": epoch + 1,
                    "step": steps,
                    "best_loss": args.best_loss,
                    "model_state_dict": model.state_dict(),
                    "ema_model_state_dict": ema.ema_model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "hparams": vars(args),
                }
                if is_xla_device(args.device):
                    save(checkpoint, ckpt_path, args.device)
                else:
                    with open_file(ckpt_path, "wb") as f:
                        save(checkpoint, f, args.device)
                sync_file(ckpt_path, os.path.join(args.remote_save_dir, "checkpoint.pt"))
                logger.info(f"Model saved: {ckpt_path}")
            if master and hasattr(args, "remote_save_dir"):
                writer.flush()
                sync_tree(args.save_dir, args.remote_save_dir)

    else:
        # test model
        model.load_state_dict(ckpt["model_state_dict"])
        ema.ema_model.load_state_dict(ckpt["ema_model_state_dict"])
        print("Evaluating test set:\n")
        stats = sup_epoch(
            args,
            ema.ema_model,
            None,
            dataloaders["test"],
            elbo_fn,
            optimizer=None,
            is_train=False,
        )
        if not args.setup == "sup_pgm":  # eval aux classifiers
            stats = eval_epoch(args, ema.ema_model, dataloaders["test"])
            print("test | " + " - ".join(f"{k}: {v:.4f}" for k, v in stats.items()))
        else:
            plot_joint(args, ema.ema_model, dataloaders["test"].dataset, 0)
            plt.show()
