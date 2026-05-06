# PyTorch Profiling Enhancement

This document describes the enhanced PyTorch profiling capabilities added to `src/performance.py`.

## Overview

The performance.py file has been enhanced with comprehensive PyTorch profiling capabilities to collect detailed performance data, identify bottlenecks, and provide better estimates of computational costs.

## New Features

### 1. Detailed Profiling Functions

- `create_profiler()`: Creates a PyTorch profiler with optimized settings
- `analyze_profiler_trace()`: Analyzes profiler traces and extracts key insights
- `save_profiler_trace()`: Saves profiler traces in Chrome format
- `profile_model_detailed()`: Performs detailed profiling with comprehensive analysis
- `print_performance_summary()`: Prints formatted performance analysis results

### 2. Enhanced Existing Functions

All existing performance functions now accept an `enable_profiling` parameter:

- `perf_fwd(enable_profiling=True)`: Forward pass with optional profiling
- `perf_bwd(enable_profiling=True)`: Backward pass with optional profiling  
- `perf_eff_fwd(enable_profiling=True)`: Efficient forward pass with optional profiling
- `perf_eff_bwd(enable_profiling=True)`: Efficient backward pass with optional profiling

### 3. New Analysis Functions

- `run_full_profiling_suite()`: Comprehensive profiling of all model variants
- `profile_memory_patterns()`: Memory usage analysis across sequence lengths
- `profile_batch_scaling()`: Performance scaling analysis across batch sizes

## Usage

### Basic Usage (No Changes Required)

```bash
# Run existing benchmarks (unchanged behavior)
python src/performance.py
```

### Detailed Profiling

```bash
# Run with detailed profiling
python src/performance.py --mode profile

# Memory pattern analysis
python src/performance.py --mode memory

# Batch scaling analysis  
python src/performance.py --mode batch

# Run all analyses
python src/performance.py --mode all
```

### Programmatic Usage

```python
from src.performance import perf_fwd, run_full_profiling_suite

# Run forward pass with profiling
perf_fwd(d_model=768, vocab=32000, runs=30, enable_profiling=True)

# Run comprehensive profiling suite
run_full_profiling_suite(d_model=768, vocab=32000, runs=10)
```

## Output Files

The profiling system generates the following output files:

- `./profiler_traces/trace_TIMESTAMP.json`: Chrome trace format for visualization

## Key Insights Provided

### Performance Metrics
- CPU and GPU operation timing
- Memory allocation patterns
- Operation-level bottleneck identification
- Kernel execution details

### Memory Analysis
- Peak memory allocation and reservation
- Memory usage scaling with sequence length
- Memory efficiency patterns

### Scaling Analysis  
- Performance scaling with batch size
- Throughput measurements
- Computational efficiency metrics

## Profiler Trace Visualization

### Chrome Tracing
1. Open Chrome browser
2. Navigate to `chrome://tracing`
3. Load the generated `.json` trace file
4. Analyze detailed timeline and operation breakdowns

## Advanced Usage

### Custom Profiling
```python
from src.performance import create_profiler, analyze_profiler_trace

with create_profiler() as prof:
    # Your model operations here
    model_output = model(input_data)
    prof.step()

# Analyze results
analysis = analyze_profiler_trace(prof)
print(analysis['cpu_top_ops'])
```

### Memory Pattern Analysis
```python
from src.performance import profile_memory_patterns

# Analyze memory usage across different sequence lengths
results = profile_memory_patterns(
    d_model=768, 
    vocab=32000, 
    sequence_lengths=[128, 256, 512, 1024]
)
```

## Benefits

1. **Bottleneck Identification**: Identify CPU/GPU bottlenecks and optimization opportunities
2. **Memory Optimization**: Understand memory allocation patterns for better optimization
3. **Performance Tracking**: Track performance changes across model iterations
4. **Resource Planning**: Better estimates of computational costs for deployment
5. **Debugging**: Detailed insights for performance debugging and optimization

## Backwards Compatibility

All existing functionality remains unchanged. The profiling features are opt-in and do not affect existing workflows.
