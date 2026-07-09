import argparse
import time

import torch

from hps import add_arguments, setup_hparams
from trainer import preprocess_batch
from train_setup import setup_dataloaders, setup_optimizer
from utils import EMA, seed_all, select_device
from vae import HVAE
from xla_runtime import (
    autocast,
    is_master,
    is_xla_device,
    master_print,
    optimizer_step,
    runtime_diagnostics,
    synchronize,
    world_size,
    wrap_loader,
)


def run_benchmark():
    parser = argparse.ArgumentParser()
    parser = add_arguments(parser)
    # Set default accelerator and hps
    parser.set_defaults(
        hps="morphomnist",
        exp_name="benchmark_run",
        data_dir="gs://medical-airnd/causal-gen/datasets/morphomnist",
    )
    args = setup_hparams(parser)

    seed_all(args.seed, args.deterministic)
    args.device = select_device(args.accelerator)
    diagnostics = runtime_diagnostics(args.device)
    if not is_xla_device(args.device) or is_master():
        print("Runtime: " + ", ".join(f"{k}={v}" for k, v in diagnostics.items()))

    # Load dataloaders
    dataloaders = setup_dataloaders(args)
    bench_loader = wrap_loader(dataloaders["train"], args.device)

    # Init model
    model = HVAE(args)

    def init_bias(m):
        if type(m) == torch.nn.Conv2d:
            torch.nn.init.zeros_(m.bias)

    model.apply(init_bias)

    ema = EMA(model, beta=args.ema_rate)

    # Optimizer
    optimizer, scheduler = setup_optimizer(args, model)

    # Place on device
    model.to(args.device)
    ema.to(args.device)

    args.expand_pa = args.vae == "hierarchical"

    model.train(True)
    model.zero_grad(set_to_none=True)

    warmup_steps = 2
    timed_steps = 6

    master_print("Starting warmup steps...")
    iterator = iter(bench_loader)
    for i in range(warmup_steps):
        batch = next(iterator)
        batch = preprocess_batch(args, batch, expand_pa=args.expand_pa)
        with autocast(args.device, args.precision):
            out = model(batch["x"], batch["pa"], beta=args.beta)
        out["elbo"] = out["elbo"] / args.accu_steps
        out["elbo"].backward()

        # Optimizer step
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer_step(optimizer, args.device)
        scheduler.step()
        ema.update()
        model.zero_grad(set_to_none=True)
        master_print(f"Warmup step {i+1}/{warmup_steps} completed.")

    synchronize(args.device)
    master_print("Starting timed steps...")
    total_time = 0.0
    elbos = []
    for i in range(timed_steps):
        batch = next(iterator)

        # Synchronize before and after each measured accelerator step.
        synchronize(args.device)

        start_time = time.perf_counter()

        batch = preprocess_batch(args, batch, expand_pa=args.expand_pa)
        with autocast(args.device, args.precision):
            out = model(batch["x"], batch["pa"], beta=args.beta)
        out["elbo"] = out["elbo"] / args.accu_steps
        out["elbo"].backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer_step(optimizer, args.device)
        scheduler.step()
        ema.update()
        model.zero_grad(set_to_none=True)

        synchronize(args.device)

        end_time = time.perf_counter()
        step_time = end_time - start_time
        total_time += step_time
        elbo_val = out["elbo"].item() * args.accu_steps
        elbos.append(elbo_val)
        master_print(
            f"Timed step {i+1}/{timed_steps} completed in {step_time:.4f}s. "
            f"ELBO: {elbo_val:.4f}"
        )
        
    avg_elbo = sum(elbos) / len(elbos)
    seconds_per_step = total_time / timed_steps
    replicas = world_size() if is_xla_device(args.device) else 1
    samples_per_second = (timed_steps * args.bs * replicas) / total_time
    
    master_print("\n--- Benchmark Results ---")
    master_print(f"RESULT_ACCELERATOR={args.accelerator.upper()}")
    master_print(f"RESULT_DEVICE={args.device}")
    master_print(f"RESULT_REPLICAS={replicas}")
    master_print(f"RESULT_GLOBAL_BATCH_SIZE={args.bs * replicas}")
    master_print(f"RESULT_TOTAL_TIME={total_time:.4f}")
    master_print(f"RESULT_SECONDS_PER_STEP={seconds_per_step:.4f}")
    master_print(f"RESULT_SAMPLES_PER_SECOND={samples_per_second:.2f}")
    master_print(f"RESULT_AVG_ELBO={avg_elbo:.4f}")
    
if __name__ == "__main__":
    run_benchmark()
