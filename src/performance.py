import torch
import time
import statistics
import torch.cuda as _cuda
import yaml
import logging
import os

from .profiler import profile_batch_scaling, profile_memory_patterns, profile_model_detailed, print_performance_summary

from .hydra import Hydra
from .hydra_modules import HydraEmbeddings
from .guider import Guider
from .utils import log_setup, load_config
from .train import sample_timestep

LOG_FILE = os.path.join("logs", "performance.log")
LOG_LEVEL = logging.INFO

config_model = load_config("configs/training_config.yaml", pretrained_config=True)

EMBEDDING = HydraEmbeddings(config=config_model)

logger = log_setup("PerformanceLogger", LOG_FILE, LOG_LEVEL)

torch.random.manual_seed(67)

def perf_fwd(runs=30, enable_profiling=False):
    """
    Benchmark forward pass performance with optional detailed profiling.
    
    Args:
        d_model: Model dimension
        vocab: Vocabulary size
        runs: Number of benchmark runs
        enable_profiling: Whether to enable detailed PyTorch profiling
    """
    
    config = load_config("configs/training_config.yaml", dict_config=True)["hydra_config"]
        
    model: Hydra = Hydra(
        d_model=config["hidden_size"],
        d_state=config["d_state"],
        d_conv=config["d_conv"],
        head_dim=config["head_dim"],
        expand=config["expand"],
        activation=config["hidden_act"],
        chunk_size=config["chunk_size"],
        use_eff_compute=False,
    )

    tokens: torch.Tensor = torch.randint(1, config_model.vocab_size, (1, 400), device="cuda")
    attn_mask: torch.Tensor = torch.ones_like(tokens, device="cuda")
    
    current_t, total_t = sample_timestep()

    x: torch.Tensor = EMBEDDING.forward(input_ids=tokens, current_timestep=current_t, total_timestep=total_t)


    # Warmup
    with torch.inference_mode():
        _ = model(x)
    
    # Optional detailed profiling
    if enable_profiling:
        logger.info("Running detailed profiling analysis...")
        def model_forward(x: torch.Tensor):
            return model(x)[0]
        
        tokens: torch.Tensor = torch.randint(0, model.vocab_size, (2, 400), device="cuda")
        attn_mask: torch.Tensor = torch.ones_like(tokens, device="cuda")
        current_t, total_t = sample_timestep()
        
        x: torch.Tensor = EMBEDDING.forward(input_ids=tokens, current_timestep=current_t, total_timestep=total_t)

        profiling_results, _ = profile_model_detailed(
            model_forward, (x,), runs=min(runs, 10), trace_file_prefix="perf_fwd"
        )
        print_performance_summary(profiling_results, prefix="perf_fwd")
    
    # Benchmark
    _cuda.reset_peak_memory_stats()
    times = []
    for _ in range(runs):
        tokens: torch.Tensor = torch.randint(1, config_model.vocab_size, (1, 400), device="cuda")
        attn_mask: torch.Tensor = torch.ones_like(tokens, device="cuda")

        current_t, total_t = sample_timestep()

        x: torch.Tensor = EMBEDDING.forward(input_ids=tokens, current_timestep=current_t, total_timestep=total_t)

        torch.cuda.synchronize()
        start = time.perf_counter()
        
        with torch.inference_mode():
            _ = model(x)[0]
            
        torch.cuda.synchronize()
        times.append(time.perf_counter() - start)

    with torch.inference_mode():
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    peak_alloc = _cuda.max_memory_allocated() / (1024 ** 2)
    logger.info(f"Peak memory allocated: {peak_alloc:.2f} MiB")

    peak_reserved = _cuda.max_memory_reserved() / (1024 ** 2)
    logger.info(f"Peak memory reserved: {peak_reserved:.2f} MiB")
    
    logger.info(f"Model parameters: {total_params:,} total, {trainable_params:,} trainable")

    logger.info(f"perf_fwd: {statistics.mean(times)*1000:.2f} ms ± {statistics.stdev(times)*1000:.2f} ms per run")

def perf_bwd(runs=30, enable_profiling=False):
    """
    Benchmark backward pass performance with optional detailed profiling.
    
    Args:
        d_model: Model dimension
        vocab: Vocabulary size
        runs: Number of benchmark runs
        enable_profiling: Whether to enable detailed PyTorch profiling
    """
    # Create model and input once
    config = load_config("configs/training_config.yaml", dict_config=True)["hydra_config"]
        
    model: Hydra = Hydra(
        d_model=config["hidden_size"],
        d_state=config["d_state"],
        d_conv=config["d_conv"],
        head_dim=config["head_dim"],
        expand=config["expand"],
        activation=config["hidden_act"],
        chunk_size=config["chunk_size"],
        use_eff_compute=False,
    )

    scaler: torch.GradScaler = torch.amp.grad_scaler.GradScaler(device="cuda")
    optimizer: torch.optim.Optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    tokens = torch.randint(0, model.vocab_size, (2, 400), device="cuda")
    attn_mask = torch.ones_like(tokens, device="cuda")
    current_t, total_t = sample_timestep()

    x: torch.Tensor = EMBEDDING.forward(input_ids=tokens, current_timestep=current_t, total_timestep=total_t)
    
    # Warmup
    y: torch.Tensor = model(x)[0]
    loss = y.mean()

    optimizer.zero_grad(set_to_none=True)

    scaler.scale(loss).backward()

    scaler.step(optimizer)

    scaler.update()

    model.zero_grad()
    
    # Optional detailed profiling
    if enable_profiling:
        logger.info("Running detailed profiling analysis...")
        def model_backward(x: torch.Tensor):
            y: torch.Tensor = model(x)[0]
            loss = y.mean()
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            model.zero_grad()
            return y
        
        tokens: torch.Tensor = torch.randint(0, model.vocab_size, (2, 400), device="cuda")
        attn_mask: torch.Tensor = torch.ones_like(tokens, device="cuda")
        current_t, total_t = sample_timestep()

        x: torch.Tensor = EMBEDDING.forward(input_ids=tokens, current_timestep=current_t, total_timestep=total_t)

        profiling_results, prof = profile_model_detailed(
            model_backward, (x,), runs=min(runs, 10), trace_file_prefix="perf_bwd"
        )
        print_performance_summary(profiling_results, prefix="perf_bwd")
    
    # Benchmark
    _cuda.reset_peak_memory_stats()
    times = []
    for _ in range(runs):
        tokens: torch.Tensor = torch.randint(0, model.vocab_size, (2, 400), device="cuda")
        attn_mask: torch.Tensor = torch.ones_like(tokens, device="cuda")
        current_t, total_t = sample_timestep()

        x: torch.Tensor = EMBEDDING.forward(input_ids=tokens, current_timestep=current_t, total_timestep=total_t)

        torch.cuda.synchronize()
        start = time.perf_counter()
        
        y = model(x)[0]
        loss = y.mean()

        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()


        torch.cuda.synchronize()
        times.append(time.perf_counter() - start)
        
        model.zero_grad()
    
    total_params = sum(p.numel() for p in model.parameters())

    peak_alloc = _cuda.max_memory_allocated() / (1024 ** 2)
    logger.info(f"Peak memory allocated: {peak_alloc:.2f} MiB")

    peak_reserved = _cuda.max_memory_reserved() / (1024 ** 2)
    logger.info(f"Peak memory reserved: {peak_reserved:.2f} MiB")

    logger.info(f"Model parameters: {total_params:,} total")
    logger.info(f"perf_bwd: {statistics.mean(times)*1000:.2f} ms ± {statistics.stdev(times)*1000:.2f} ms per run")

def perf_eff_fwd(runs=30, enable_profiling=False):
    """
    Benchmark efficient forward pass performance with optional detailed profiling.
    
    Args:
        d_model: Model dimension
        vocab: Vocabulary size
        runs: Number of benchmark runs
        enable_profiling: Whether to enable detailed PyTorch profiling
    """
    config = load_config("configs/training_config.yaml", dict_config=True)["hydra_config"]
        
    model: Hydra = Hydra(
        d_model=config["hidden_size"],
        d_state=config["d_state"],
        d_conv=config["d_conv"],
        head_dim=config["head_dim"],
        expand=config["expand"],
        activation=config["hidden_act"],
        chunk_size=config["chunk_size"],
        use_eff_compute=True,
    )

    tokens: torch.Tensor = torch.randint(0, model.vocab_size, (2, 512), device="cuda")
    attn_mask: torch.Tensor = torch.ones_like(tokens, device="cuda")
    current_t, total_t = sample_timestep()

    x: torch.Tensor = EMBEDDING.forward(input_ids=tokens, current_timestep=current_t, total_timestep=total_t)

    # Warmup
    with torch.inference_mode():
        _ = model(x)[0]
    
    # Optional detailed profiling
    if enable_profiling:
        logger.info("Running detailed profiling analysis...")
        def model_forward(x: torch.Tensor, t: torch.Tensor):
            with torch.inference_mode():
                return model(x, t)[0]

        tokens: torch.Tensor = torch.randint(0, model.vocab_size, (2, 512), device="cuda")
        attn_mask: torch.Tensor = torch.ones_like(tokens, device="cuda")
        current_t, total_t = sample_timestep()

        x: torch.Tensor = EMBEDDING.forward(input_ids=tokens, current_timestep=current_t, total_timestep=total_t)

        profiling_results, prof = profile_model_detailed(
            model_forward, (x,), runs=min(runs, 10), trace_file_prefix="perf_eff_fwd"
        )
        print_performance_summary(profiling_results, prefix="perf_eff_fwd")
    
    # Benchmark
    _cuda.reset_peak_memory_stats()
    times = []
    for _ in range(runs):
        tokens: torch.Tensor = torch.randint(0, model.vocab_size, (2, 512), device="cuda")
        attn_mask: torch.Tensor = torch.ones_like(tokens, device="cuda")
        current_t, total_t = sample_timestep()

        x: torch.Tensor = EMBEDDING.forward(input_ids=tokens, current_timestep=current_t, total_timestep=total_t)

        torch.cuda.synchronize()
        start = time.perf_counter()
        
        with torch.inference_mode():
            _ = model(x)
            
        torch.cuda.synchronize()
        times.append(time.perf_counter() - start)

    total_params = sum(p.numel() for p in model.parameters())

    logger.info(f"Model parameters: {total_params:,} total")

    logger.info(f"perf_eff_fwd: {statistics.mean(times)*1000:.2f} ms ± {statistics.stdev(times)*1000:.2f} ms per run")

    peak_alloc = _cuda.max_memory_allocated() / (1024 ** 2)
    logger.info(f"Peak memory allocated: {peak_alloc:.2f} MiB")

    peak_reserved = _cuda.max_memory_reserved() / (1024 ** 2)
    logger.info(f"Peak memory reserved: {peak_reserved:.2f} MiB")

def perf_eff_bwd(runs=30, enable_profiling=False):
    """
    Benchmark efficient backward pass performance with optional detailed profiling.
    
    Args:
        runs: Number of benchmark runs
        enable_profiling: Whether to enable detailed PyTorch profiling
    """
    config = load_config("configs/training_config.yaml", dict_config=True)["hydra_config"]
        
    model: Hydra = Hydra(
        d_model=config["hidden_size"],
        d_state=config["d_state"],
        d_conv=config["d_conv"],
        head_dim=config["head_dim"],
        expand=config["expand"],
        activation=config["hidden_act"],
        chunk_size=config["chunk_size"],
        use_eff_compute=True,
    )

    scaler: torch.GradScaler = torch.amp.grad_scaler.GradScaler(device="cuda")
    optimizer: torch.optim.Optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    tokens: torch.Tensor = torch.randint(0, model.vocab_size, (2, 512), device="cuda")
    attn_mask: torch.Tensor = torch.ones_like(tokens, device="cuda")
    current_t, total_t = sample_timestep()

    x: torch.Tensor = EMBEDDING.forward(input_ids=tokens, current_timestep=current_t, total_timestep=total_t)

    # Warmup
    y: torch.Tensor = model(x)[0]
    loss: torch.Tensor = y.mean()
    optimizer.zero_grad(set_to_none=True)
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
    model.zero_grad()
    
    # Optional detailed profiling
    if enable_profiling:
        logger.info("Running detailed profiling analysis...")
        def model_backward(x : torch.Tensor):
            y: torch.Tensor = model(x)[0]
            loss: torch.Tensor = y.mean()
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            model.zero_grad()
            return y
        
        tokens: torch.Tensor = torch.randint(0, model.vocab_size, (2, 512), device="cuda")
        attn_mask: torch.Tensor = torch.ones_like(tokens, device="cuda")
        current_t, total_t = sample_timestep()

        x: torch.Tensor = EMBEDDING.forward(input_ids=tokens, current_timestep=current_t, total_timestep=total_t)
        
        profiling_results, _ = profile_model_detailed(
            model_backward, (x,), runs=min(runs, 10), trace_file_prefix="perf_eff_bwd"
        )
        print_performance_summary(profiling_results, prefix="perf_eff_bwd")
    
    # Benchmark
    _cuda.reset_peak_memory_stats()
    times = []
    for _ in range(runs):
        tokens: torch.Tensor = torch.randint(0, model.vocab_size, (2, 512), device="cuda")
        attn_mask: torch.Tensor = torch.ones_like(tokens, device="cuda")
        current_t, total_t = sample_timestep()

        x: torch.Tensor = EMBEDDING.forward(input_ids=tokens, current_timestep=current_t, total_timestep=total_t)

        torch.cuda.synchronize()
        start = time.perf_counter()
        
        y: torch.Tensor = model(x)[0]
        loss: torch.Tensor = y.mean()
        
        optimizer.zero_grad(set_to_none=True)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        torch.cuda.synchronize()
        times.append(time.perf_counter() - start)
        
        model.zero_grad()

    total_params = sum(p.numel() for p in model.parameters())

    logger.info(f"Model parameters: {total_params:,} total")

    logger.info(f"perf_eff_bwd: {statistics.mean(times)*1000:.2f} ms ± {statistics.stdev(times)*1000:.2f} ms per run")

    peak_alloc = _cuda.max_memory_allocated() / (1024 ** 2)
    logger.info(f"Peak memory allocated: {peak_alloc:.2f} MiB")

    peak_reserved = _cuda.max_memory_reserved() / (1024 ** 2)
    logger.info(f"Peak memory reserved: {peak_reserved:.2f} MiB")

def perf_fwd_guider(runs: int=30, enable_profiling: bool=False):
    """
    Benchmark forward pass performance of the Guider model with optional detailed profiling.

    Args:
        runs: Number of benchmark runs
        enable_profiling: Whether to enable detailed PyTorch profiling
    """

    model: Guider = Guider(**CONFIG["guider"])

    # Warmup
    tokens: torch.Tensor = torch.randint(0, model.vocab_size, (2, 512), device="cuda")
    attn_mask: torch.Tensor = torch.ones_like(tokens, device="cuda")

    x: torch.Tensor = EMBEDDING.forward(input_ids=tokens)

    with torch.inference_mode():
        _: torch.Tensor = model(x)

    if enable_profiling:
        logger.info("Running detailed profiling analysis...")
        def model_forward(x: torch.Tensor):
            with torch.inference_mode():
                return model(x)

        x: torch.Tensor = torch.randint(0, model.vocab_size, (2, 512), device="cuda")
            
        profiling_results, _ = profile_model_detailed(
            model_forward, (x,), runs=min(runs, 10), trace_file_prefix="perf_fwd_guider"
        )

        print_performance_summary(profiling_results, prefix="perf_fwd_guider")

    # Benchmark
    _cuda.reset_peak_memory_stats()

    times: list[float] = list()

    for _ in range(runs):
        x: torch.Tensor = torch.randint(0, model.vocab_size, (2, 512), device="cuda")

        torch.cuda.synchronize()
        start: float = time.perf_counter()

        with torch.inference_mode():
            _: tuple[torch.Tensor, torch.Tensor] = model(x)

        torch.cuda.synchronize()
        times.append(time.perf_counter() - start)

    logger.info(f"perf_fwd_guider: {statistics.mean(times)*1000:.2f} ms ± {statistics.stdev(times)*1000:.2f} ms per run")

    peak_alloc = _cuda.max_memory_allocated() / (1024 ** 2)
    logger.info(f"Peak memory allocated: {peak_alloc:.2f} MiB")

    peak_reserved = _cuda.max_memory_reserved() / (1024 ** 2)
    logger.info(f"Peak memory reserved: {peak_reserved:.2f} MiB")

    total_params = sum(p.numel() for p in model.parameters())

    logger.info(f"Model parameters: {total_params:,} total")

def perf_bwd_guider(runs: int=30, enable_profiling: bool=False):
    """
    Benchmark backward pass performance of the Guider model with optional detailed profiling.

    Args:
        d_model: Model dimension
        vocab: Vocabulary size
        runs: Number of benchmark runs
        enable_profiling: Whether to enable detailed PyTorch profiling
    """

    model: Guider = Guider(**CONFIG["guider"])
    scaler: torch.GradScaler = torch.amp.grad_scaler.GradScaler(device="cuda")
    optimizer: torch.optim.Optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    x: torch.Tensor = torch.randint(0, model.vocab_size, (2, 512), device="cuda")

    # Warmup
    y: tuple[torch.Tensor, torch.Tensor] = model(x)
    loss: torch.Tensor = y[0].mean()

    optimizer.zero_grad(set_to_none=True)

    scaler.scale(loss).backward()

    scaler.step(optimizer)

    scaler.update()

    if enable_profiling:
        logger.info("Running detailed profiling analysis...")
        def model_backward(x: torch.Tensor):
            y: tuple[torch.Tensor, torch.Tensor] = model(x)
            loss: torch.Tensor = y[0].mean()
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            return y
        
        x: torch.Tensor = torch.randint(0, model.vocab_size, (2, 512), device="cuda")

        profiling_results, _ = profile_model_detailed(
            model_backward, (x,), runs=min(runs, 10), trace_file_prefix="perf_bwd_guider"
        )

        print_performance_summary(profiling_results, prefix="perf_bwd_guider")

    # Benchmark
    _cuda.reset_peak_memory_stats()
    times: list[float] = list()

    for _ in range(runs):
        x: torch.Tensor = torch.randint(0, model.vocab_size, (2, 512), device="cuda")

        torch.cuda.synchronize()
        start: float = time.perf_counter()

        y: tuple[torch.Tensor, torch.Tensor] = model(x)
        loss: torch.Tensor = y[0].mean()
        
        optimizer.zero_grad(set_to_none=True)

        scaler.scale(loss).backward()

        scaler.step(optimizer)

        scaler.update()

        torch.cuda.synchronize()
        times.append(time.perf_counter() - start)

        model.zero_grad()

    logger.info(f"perf_bwd_guider: {statistics.mean(times)*1000:.2f} ms ± {statistics.stdev(times)*1000:.2f} ms per run")

    peak_alloc: int = _cuda.max_memory_allocated() / (1024 ** 2)
    logger.info(f"Peak memory allocated: {peak_alloc:.2f} MiB")

    peak_reserved: int = _cuda.max_memory_reserved() / (1024 ** 2)
    logger.info(f"Peak memory reserved: {peak_reserved:.2f} MiB")

    total_params: int = sum(p.numel() for p in model.parameters())

    logger.info(f"Model parameters: {total_params:,} total")


def run_full_profiling_suite(runs=10):
    """
    Run a comprehensive profiling suite with detailed analysis.
    
    Args:
        d_model: Model dimension
        vocab: Vocabulary size
        runs: Number of profiling runs
    """
    print("\n" + "="*80)
    print("COMPREHENSIVE PYTORCH PROFILING SUITE")
    print("="*80)

    logger.info("\n1. Hydra Forward Pass Profiling:")
    perf_fwd(runs, enable_profiling=True)
    
    logger.info("\n2. Hydra Backward Pass Profiling:")
    perf_bwd(runs, enable_profiling=True)

    logger.info("\n3. Hydra Efficient Forward Pass Profiling:")
    perf_eff_fwd(runs, enable_profiling=True)
    
    logger.info("\n4. Hydra Efficient Backward Pass Profiling:")
    perf_eff_bwd(runs, enable_profiling=True)

    logger.info("\n5. Guider Forward Pass Profiling:")
    perf_fwd_guider(runs, enable_profiling=True)

    logger.info("\n6. Guider Backward Pass Profiling:")
    perf_bwd_guider(runs, enable_profiling=True)

if __name__ == "__main__":
    import argparse
    import json
    
    parser = argparse.ArgumentParser(description='PyTorch Performance Profiling Suite')
    parser.add_argument('--mode', choices=['basic', 'profile', 'memory', 'batch', 'all', 'count'], 
                       default='basic', help='Profiling mode to run')
    parser.add_argument('--runs', type=int, default=30, help='Number of benchmark runs')
    parser.add_argument('--params', choices=["3B", "135M", "default"], default="default" ,help='Model parameters to use')

    args: argparse.Namespace = parser.parse_args()

    if args.params == "3B":
        CONFIG_PATH = "src/3B_hydra.yaml"
        CONFIG = yaml.safe_load(open(CONFIG_PATH, "r"))

    elif args.params == "135M":
        CONFIG_PATH = "src/135M_hydra.yaml"
        CONFIG = yaml.safe_load(open(CONFIG_PATH, "r"))
    
    else:
        CONFIG_PATH = "configs/perf_config.yaml"
        CONFIG = yaml.safe_load(open(CONFIG_PATH, "r"))


    if args.mode == 'basic':
        logger.info("Running basic performance benchmarks...")

        perf_fwd(args.runs)
        perf_bwd(args.runs)

        perf_eff_fwd(args.runs)
        perf_eff_bwd(args.runs)

        perf_fwd_guider(args.runs)
        perf_bwd_guider(args.runs)

    elif args.mode == 'profile':
        logger.info("Running detailed profiling suite...")

        hydra: Hydra = Hydra(**CONFIG["hydra"])
        run_full_profiling_suite(min(args.runs, 10))

        guider: Guider = Guider(**CONFIG["guider"])
        run_full_profiling_suite(min(args.runs, 10))

    elif args.mode == 'memory':
        logger.info("Running memory pattern analysis...")

        hydra: Hydra = Hydra(**CONFIG["hydra"])
        profile_memory_patterns(hydra, trace_file_prefix="perf_hydra_memory")

        guider: Guider = Guider(**CONFIG["guider"])
        profile_memory_patterns(guider, trace_file_prefix="perf_guider_memory", guider=True)
    
    
    elif args.mode == 'batch':
        logger.info("Running batch scaling analysis...")

        hydra: Hydra = Hydra(**CONFIG["hydra"])
        profile_batch_scaling(hydra)

        guider: Guider = Guider(**CONFIG["guider"])
        profile_batch_scaling(guider, guider=True)

    elif args.mode == 'all':
        logger.info("Running comprehensive analysis...")

        hydra: Hydra = Hydra(**CONFIG["hydra"])
        guider: Guider = Guider(**CONFIG["guider"])

        run_full_profiling_suite(runs=min(args.runs, 10))

        profile_memory_patterns(hydra)
        profile_batch_scaling(hydra)

        profile_memory_patterns(guider, guider=True)
        profile_batch_scaling(guider, guider=True)

    elif args.mode == "count":
        logger.info("Counting model parameters...")

        hydra: Hydra = Hydra(**CONFIG["hydra"])
        total_params = sum(p.numel() for p in hydra.parameters())
        trainable_params = sum(p.numel() for p in hydra.parameters() if p.requires_grad)
        logger.info(f"Hydra Model Parameters: {total_params:,} total, {trainable_params:,} trainable")

        guider: Guider = Guider(**CONFIG["guider"])
        total_params = sum(p.numel() for p in guider.parameters())
        trainable_params = sum(p.numel() for p in guider.parameters() if p.requires_grad)
        logger.info(f"Guider Model Parameters: {total_params:,} total, {trainable_params:,} trainable")