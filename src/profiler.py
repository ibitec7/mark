import torch
from torch.profiler import profile, record_function, ProfilerActivity
import os
from datetime import datetime
import statistics
import logging
import time

import torch.cuda as _cuda
from .utils import log_setup

PROFILE_DIR = "profiles"

os.makedirs(PROFILE_DIR, exist_ok=True)

LOG_FILE = os.path.join("logs", "profiler.log")
LOG_LEVEL = logging.INFO

logger = log_setup("ProfilerLogger", LOG_FILE, LOG_LEVEL)

# Helper function
def create_profiler(activities: list[torch.profiler.ProfilerActivity]=None,\
                     record_shapes: bool=True, profile_memory: bool=True,\
                          output_dir: str="./profiler_traces") -> torch.profiler.profile:
    
    """
    Create a PyTorch profiler with optimized settings for performance analysis.

    Args:
        activities {list[torch.profiler.ProfilerActivity]}: List of ProfilerActivity to monitor (default: CPU and CUDA if available).
        record_shapes {bool}: Whether to record shapes of tensors (default: True).
        profile_memory {bool}: Whether to profile memory usage (default: True).
        output_dir {str}: Directory to save profiler traces (default: "./profiler_traces").

    Returns:
        torch.profiler.profile: Configured profiler instance.
    """

    if activities is None:
        activities = [ProfilerActivity.CPU]
        if torch.cuda.is_available():
            activities.append(ProfilerActivity.CUDA)
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    return profile(
        activities=activities,
        record_shapes=record_shapes,
        profile_memory=profile_memory,
        with_flops=True,
        with_modules=True,
    )

# Helper function
def analyze_profiler_trace(prof: torch.profiler.profile, top_k: int=10) -> dict:

    """
    Analyze profiler trace and extract key performance insights.

    Args:
        prof {torch.profiler.profile}: The profiler object containing the trace data.
        top_k {int}: Number of top operations to display in the analysis (default: 10).

    Returns:
        dict: Dictionary containing performance analysis results.
    """

    results = {}
    
    # CPU time analysis
    cpu_events = prof.key_averages().table(sort_by="cpu_time_total", row_limit=top_k)
    results['cpu_top_ops'] = cpu_events
    
    # GPU time analysis if CUDA is available
    if torch.cuda.is_available():
        gpu_events = prof.key_averages().table(sort_by="cuda_time_total", row_limit=top_k)
        results['gpu_top_ops'] = gpu_events
    
    # Memory analysis
    if prof.profiler.profile_memory:
        memory_events = prof.key_averages().table(sort_by="cpu_memory_usage", row_limit=top_k)
        results['memory_top_ops'] = memory_events
    
    # Get total times
    key_averages = prof.key_averages()
    total_cpu_time = sum([item.cpu_time_total for item in key_averages])
    results['total_cpu_time_us'] = total_cpu_time
    
    if torch.cuda.is_available():
        total_cuda_time = sum([item.self_device_time_total for item in key_averages])
        results['total_cuda_time_us'] = total_cuda_time
    
    return results

# Helper function
def save_profiler_trace(prof: torch.profiler.profile, filename_prefix: str="trace", \
                        output_dir: str="./profiler_traces") -> str:
    
    """
    Save profiler trace to files for later analysis.

    Args:
        prof {torch.profiler.profile}: The profiler object containing the trace data.
        filename_prefix {str}: Prefix for the saved trace files.
        output_dir {str}: Directory where the trace files will be saved.

    Returns:
        str: Path to the saved Chrome trace file.
    """

    os.makedirs(output_dir, exist_ok=True)
    timestamp: str = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Save Chrome trace format
    chrome_trace_path: str = os.path.join(output_dir, f"{filename_prefix}_{timestamp}.json")
    prof.export_chrome_trace(chrome_trace_path)
    
    return chrome_trace_path

def profile_model_detailed(model_fn: callable, inputs: tuple, runs: int=5,\
     warmup_runs: int=2, trace_file_prefix: str="detailed_profile") -> tuple[dict, torch.profiler.profile]:
    
    """
    Perform detailed profiling of a model function with comprehensive analysis.
    
    Args:
        model_fn {callable}: Function that takes inputs and returns model output
        inputs {tuple}: Input data for the model
        runs {int}: Number of profiling runs
        warmup_runs {int}: Number of warmup runs before profiling
        trace_file_prefix {str}: Prefix for saved trace files

    Returns:
        dict: Dictionary containing profiling results and analysis
    """
    
    # Warmup runs
    print(f"Performing {warmup_runs} warmup runs...")
    for _ in range(warmup_runs):
        model_fn(*inputs)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
    
    # Profiling runs
    print(f"Starting detailed profiling for {runs} runs...")
    
    with create_profiler() as prof:
        for _ in range(runs):
            with record_function("model_forward_pass"):
                model_fn(*inputs)
            
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            prof.step()
    
    # Analyze results
    analysis: dict = analyze_profiler_trace(prof)

    # Save traces
    chrome_path: str = save_profiler_trace(prof, trace_file_prefix)

    results: dict = {
        'analysis': analysis,
        'chrome_trace_path': chrome_path,
        'total_runs': runs,
        'warmup_runs': warmup_runs
    }
    
    return results, prof

# Helper function
def print_performance_summary(results: dict, prefix: str="performance"):

    """
    Print a formatted summary of performance analysis results.

    Args:
        results {dict}: Dictionary containing profiling results and analysis
        prefix {str}: Prefix for the performance summary
    """

    analysis = results['analysis']

    # Save analysis results to a text file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs(PROFILE_DIR, exist_ok=True)
    analysis_file = os.path.join(PROFILE_DIR, f"{prefix}_analysis_{timestamp}.txt")
    
    with open(analysis_file, 'w') as f:
        f.write("PYTORCH PROFILER PERFORMANCE ANALYSIS\n")
        f.write("="*80 + "\n\n")

        f.write(f"Function Prefix: {prefix}\n\n")
        if 'total_cpu_time_us' in analysis:
            f.write(f"Total CPU Time: {analysis['total_cpu_time_us']/1000:.2f} ms\n")
        
        if 'total_cuda_time_us' in analysis:
            f.write(f"Total CUDA Time: {analysis['total_cuda_time_us']/1000:.2f} ms\n")
        
        f.write("\nTop CPU Operations:\n")
        f.write(str(analysis['cpu_top_ops']) + "\n")
        
        if 'gpu_top_ops' in analysis:
            f.write("\nTop GPU Operations:\n")
            f.write(str(analysis['gpu_top_ops']) + "\n")
        
        if 'memory_top_ops' in analysis:
            f.write("\nTop Memory Operations:\n")
            f.write(str(analysis['memory_top_ops']) + "\n")
    
    # Add the file path to the results dictionary
    results['analysis_file_path'] = analysis_file
    
    print("\n" + "="*80)
    print("PYTORCH PROFILER PERFORMANCE ANALYSIS")
    print("="*80)
    
    if 'total_cpu_time_us' in analysis:
        print(f"Total CPU Time: {analysis['total_cpu_time_us']/1000:.2f} ms")
    
    if 'total_cuda_time_us' in analysis:
        print(f"Total CUDA Time: {analysis['total_cuda_time_us']/1000:.2f} ms")
    
    print(f"\nTrace files saved:")
    print(f"- Chrome trace: {results['chrome_trace_path']}")
    
    print("\nTop CPU Operations:")
    print(analysis['cpu_top_ops'])
    
    if 'gpu_top_ops' in analysis:
        print("\nTop GPU Operations:")
        print(analysis['gpu_top_ops'])
    
    if 'memory_top_ops' in analysis:
        print("\nTop Memory Operations:")
        print(analysis['memory_top_ops'])
    
    print("="*80)

def profile_memory_patterns(model: torch.nn.Module, trace_file_prefix: str="memory_seq", \
                            sequence_lengths: list[int]=[128, 256, 512, 1024, 2048], guider: bool=False) -> dict:

    """
    Profile memory usage patterns across different sequence lengths.
    
    Args:
        model {torch.nn.Module}: The model to profile
        sequence_lengths {list[int]}: List of sequence lengths to test

    Returns:
        dict: Dictionary containing memory usage results for each sequence length
    """

    print("\n" + "="*80)
    print("MEMORY USAGE PATTERN ANALYSIS")
    print("="*80)
    
    results = {}
    
    for seq_len in sequence_lengths:
        print(f"\nTesting sequence length: {seq_len}")
        
        x: torch.Tensor = torch.randint(0, model.vocab_size, (2, seq_len), device="cuda")
        t: torch.Tensor = torch.randint(0, 100, (2,), device="cuda")
        
        _cuda.reset_peak_memory_stats()
        
        if not guider:
            def model_forward():
                return model(x, t)[0]
        else:
            def model_forward():
                return model(x)[0]
        
        # Profile with memory focus
        with create_profiler(profile_memory=True) as prof:
            with record_function(f"seq_len_{seq_len}"):
                y: torch.Tensor = model_forward()
            prof.step()

        peak_alloc: float = _cuda.max_memory_allocated() / (1024 ** 2)
        peak_reserved: float = _cuda.max_memory_reserved() / (1024 ** 2)

        results[seq_len] = {
            'peak_allocated_mb': peak_alloc,
            'peak_reserved_mb': peak_reserved,
            'output_shape': y.shape
        }
        
        print(f"  Peak allocated: {peak_alloc:.2f} MiB")
        print(f"  Peak reserved: {peak_reserved:.2f} MiB")
        print(f"  Output shape: {y.shape}")
        
        # Save trace
        trace_path: str = save_profiler_trace(prof, f"{trace_file_prefix}_{seq_len}")
        print(f"  Trace saved: {trace_path}")
    
    # Summary
    print(f"\nMemory Scaling Summary:")
    for seq_len, result in results.items():
        print(f"  {seq_len:4d}: {result['peak_allocated_mb']:6.2f} MiB allocated")
    
    return results

def profile_batch_scaling(model: torch.nn.Module, batch_sizes: list[int]=[1, 2, 4, 8, 16], guider: bool=False) -> dict:
    
    """
    Profile performance scaling across different batch sizes.
    
    Args:
        model {torch.nn.Module}: The model to profile
        batch_sizes {list[int]}: List of batch sizes to test

    Returns:
        dict: Dictionary containing performance results for each batch size
    """
    
    print("\n" + "="*80)
    print("BATCH SIZE SCALING ANALYSIS")
    print("="*80)
    
    results: dict = {}
    seq_len: int = 1024  # Fixed sequence length
    
    for batch_size in batch_sizes:
        logger.info(f"\nTesting batch size: {batch_size}")

        x: torch.Tensor = torch.randint(0, model.vocab_size, (batch_size, seq_len), device="cuda")

        if not guider:
            t: torch.Tensor = torch.randint(0, 100, (batch_size,), device="cuda")

            def model_forward():
                return model(x, t)[0]
        else:
            def model_forward():
                return model(x)[0]

        # Warmup
        model_forward()
        
        # Time measurement
        times: list[float] = []
        _cuda.reset_peak_memory_stats()
        
        for _ in range(5):
            torch.cuda.synchronize()
            start: float = time.perf_counter()
            _ = model_forward()
            torch.cuda.synchronize()
            times.append(time.perf_counter() - start)

        avg_time: float = statistics.mean(times) * 1000  # ms
        peak_alloc: float = _cuda.max_memory_allocated() / (1024 ** 2)
        
        results[batch_size] = {
            'avg_time_ms': avg_time,
            'peak_memory_mb': peak_alloc,
            'throughput_samples_per_sec': batch_size / (avg_time / 1000)
        }
        
        logger.info(f"  Average time: {avg_time:.2f} ms")
        logger.info(f"  Peak memory: {peak_alloc:.2f} MiB")
        logger.info(f"  Throughput: {results[batch_size]['throughput_samples_per_sec']:.2f} samples/sec")
    
    return results