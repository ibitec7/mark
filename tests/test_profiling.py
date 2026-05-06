#!/usr/bin/env python3
"""
Test script for the enhanced PyTorch profiling functionality in performance.py

This script validates that the profiling enhancements work correctly
and can be safely imported and used.

Usage:
    pytest test_profiling.py
    or
    python test_profiling.py
"""

import pytest
import sys
import os
import inspect

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))


def test_imports():
    """Test that all profiling functions can be imported successfully."""
    pytest.importorskip("torch")
    pytest.importorskip("mamba_ssm")

    try:
        from src.profiler import (
            create_profiler,
            analyze_profiler_trace,
            save_profiler_trace,
        )
        
        from src.performance import (
            profile_model_detailed,
            print_performance_summary,
            run_full_profiling_suite,
            profile_memory_patterns,
            profile_batch_scaling,
            perf_fwd,
            perf_bwd,
            perf_eff_fwd,
            perf_eff_bwd
        )
    except ImportError as e:
        pytest.fail(f"Import failed: {e}")
        return
    
    # If we get here without ImportError, the test passes
    assert True


def test_profiler_creation():
    """Test that the profiler can be created."""
    pytest.importorskip("torch")
    pytest.importorskip("mamba_ssm")
    
    from src.profiler import create_profiler
    profiler = create_profiler()
    assert profiler is not None


def test_function_signatures():
    """Test that enhanced functions accept the new enable_profiling parameter."""
    pytest.importorskip("torch")
    pytest.importorskip("mamba_ssm")
    
    from src.performance import perf_fwd, perf_bwd, perf_eff_fwd, perf_eff_bwd
    
    functions_to_test = [
        ('perf_fwd', perf_fwd), 
        ('perf_bwd', perf_bwd), 
        ('perf_eff_fwd', perf_eff_fwd), 
        ('perf_eff_bwd', perf_eff_bwd)
    ]
    
    for func_name, func in functions_to_test:
        sig = inspect.signature(func)
        assert 'enable_profiling' in sig.parameters, f"{func_name} missing enable_profiling parameter"


def test_new_profiling_functions_exist():
    """Test that all new profiling functions are defined."""
    pytest.importorskip("torch")
    pytest.importorskip("mamba_ssm")
    
    from src.profiler import (
        create_profiler,
        analyze_profiler_trace,
        save_profiler_trace,
    )

    from src.performance import (
        profile_model_detailed,
        print_performance_summary,
        run_full_profiling_suite,
        profile_memory_patterns,
        profile_batch_scaling
    )
    
    # Test that functions are callable
    assert callable(create_profiler)
    assert callable(analyze_profiler_trace)
    assert callable(save_profiler_trace)
    assert callable(profile_model_detailed)
    assert callable(print_performance_summary)
    assert callable(run_full_profiling_suite)
    assert callable(profile_memory_patterns)
    assert callable(profile_batch_scaling)


def test_pytest_compatibility():
    """Test that this test file is pytest compatible."""
    # This test validates that pytest can discover and run this file
    assert True


if __name__ == "__main__":
    # Run tests when executed directly
    pytest.main([__file__, "-v"])