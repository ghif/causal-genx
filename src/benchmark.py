import argparse
import time
import torch
from torch.utils.data import Subset, DataLoader

from hps import add_arguments, setup_hparams
from trainer import preprocess_batch
from train_setup import setup_dataloaders, setup_optimizer
from utils import EMA, seed_all, select_device
from vae import HVAE

def run_benchmark():
    parser = argparse.ArgumentParser()
    parser = add_arguments(parser)
    # Set default accelerator and hps
    parser.set_defaults(
        hps="morphomnist",
        exp_name="benchmark_run",
        data_dir="gs://causal-gen/datasets/morphomnist",
    )
    args = setup_hparams(parser)

    seed_all(args.seed, args.deterministic)
    args.device = select_device(args.accelerator)
    print(f"Using device: {args.device}")

    # Load dataloaders
    dataloaders = setup_dataloaders(args)

    # Subset train dataset to 256 samples
    train_subset = Subset(dataloaders["train"].dataset, list(range(256)))
    bench_loader = DataLoader(train_subset, batch_size=args.bs, shuffle=False, num_workers=0)

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

    print("Starting warmup steps...")
    iterator = iter(bench_loader)
    for i in range(warmup_steps):
        batch = next(iterator)
        batch = preprocess_batch(args, batch, expand_pa=args.expand_pa)
        out = model(batch["x"], batch["pa"], beta=args.beta)
        out["elbo"] = out["elbo"] / args.accu_steps
        out["elbo"].backward()

        # Optimizer step
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()
        scheduler.step()
        ema.update()
        model.zero_grad(set_to_none=True)
        print(f"Warmup step {i+1}/{warmup_steps} completed.")

    print("Starting timed steps...")
    total_time = 0.0
    elbos = []
    for i in range(timed_steps):
        batch = next(iterator)

        # Synchronization for accurate timing on MPS/CUDA
        if args.device.type == "cuda":
            torch.cuda.synchronize()
        elif args.device.type == "mps":
            if hasattr(torch, "mps") and hasattr(torch.mps, "synchronize"):
                torch.mps.synchronize()

        start_time = time.perf_counter()

        batch = preprocess_batch(args, batch, expand_pa=args.expand_pa)
        out = model(batch["x"], batch["pa"], beta=args.beta)
        out["elbo"] = out["elbo"] / args.accu_steps
        out["elbo"].backward()

        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()
        scheduler.step()
        ema.update()
        model.zero_grad(set_to_none=True)

        if args.device.type == "cuda":
            torch.cuda.synchronize()
        elif args.device.type == "mps":
            if hasattr(torch, "mps") and hasattr(torch.mps, "synchronize"):
                torch.mps.synchronize()

        end_time = time.perf_counter()
        step_time = end_time - start_time
        total_time += step_time
        elbo_val = out["elbo"].item() * args.accu_steps
        elbos.append(elbo_val)
        print(f"Timed step {i+1}/{timed_steps} completed in {step_time:.4f}s. ELBO: {elbo_val:.4f}")
        
    avg_elbo = sum(elbos) / len(elbos)
    seconds_per_step = total_time / timed_steps
    samples_per_second = (timed_steps * args.bs) / total_time
    
    print("\n--- Benchmark Results ---")
    print(f"RESULT_ACCELERATOR={args.accelerator.upper()}")
    print(f"RESULT_DEVICE={args.device}")
    print(f"RESULT_TOTAL_TIME={total_time:.4f}")
    print(f"RESULT_SECONDS_PER_STEP={seconds_per_step:.4f}")
    print(f"RESULT_SAMPLES_PER_SECOND={samples_per_second:.2f}")
    print(f"RESULT_AVG_ELBO={avg_elbo:.4f}")
    
if __name__ == "__main__":
    run_benchmark()
