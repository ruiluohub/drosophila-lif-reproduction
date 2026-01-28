#!/usr/bin/env python3
"""
Parallel Test Script for Experiment 1: Sugar GRN Activation

Tests the parallel implementation of frequency sweep experiment.
Each worker independently builds network and runs simulations.

Usage:
    python scripts/test_parallel_exp1.py --workers 6 --trials 2
    
Expected runtime: 2-3 minutes on 6-core CPU
Expected speedup: 3-5x vs serial execution
"""

import sys
from pathlib import Path
import argparse
import pickle
import time
import numpy as np
import pandas as pd

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from flylif.core.parameters import DEFAULT_PARAMS
from flylif.core.network import build_network
from flylif.core.simulation import run_simulation
from flylif.core.data_loader import load_all_data
from joblib import Parallel, delayed
from brian2 import Hz


# =============================================================================
# Configuration
# =============================================================================

CONFIG = {
    'data_dir': Path('./data_783'),
    'connections_file': 'proofread_connections_783.feather',
    'root_ids_file': 'proofread_root_ids_783.npy',
    'classification': 'classification.csv',
}

# Sugar GRN neurons (left hemisphere, FlyWire v783)
NEU_SUGAR_LEFT = [
    720575940624963786, 720575940630233916, 720575940637568838, 720575940638202345,
    720575940617000768, 720575940630797113, 720575940632889389, 720575940621754367,
    720575940621502051, 720575940640649691, 720575940639332736, 720575940616885538,
    720575940639259967, 720575940617937543, 720575940632425919, 720575940633143833,
    720575940612670570, 720575940628853239, 720575940629176663, 720575940611875570,
]

# MN9 neurons (target for validation)
NEU_MN9_RIGHT = [720575940660219265]


# =============================================================================
# Worker Function
# =============================================================================

def run_frequency_worker(freq, neu_exc, data_dict, config_dict, params, n_trials):
    """
    Worker function: builds network and runs simulation for one frequency.
    
    Each worker independently:
    1. Builds complete network (~8 seconds)
    2. Runs multiple trials serially
    3. Returns aggregated results
    
    Parameters
    ----------
    freq : int
        Stimulation frequency in Hz
    neu_exc : list
        FlyWire IDs of neurons to activate
    data_dict : dict
        Preloaded connectivity data
    config_dict : dict
        Column names and configuration
    params : dict
        Brian2 simulation parameters
    n_trials : int
        Number of trials to run
    
    Returns
    -------
    tuple : (frequency, result_dict)
    """
    
    # Build network (independent per worker)
    net_components = build_network(
        data=data_dict,
        pre_col=config_dict['pre_col'],
        post_col=config_dict['post_col'],
        weight_col=config_dict['weight_col'],
        nt_prob_cols=config_dict['nt_prob_cols'],
        params=params,
        syn_threshold=5,
        verbose=False  # Suppress worker output
    )
    
    # Run simulation
    result = run_simulation(
        net_components=net_components,
        neu_exc=neu_exc,
        params={'r_poi': freq * Hz},
        n_trials=n_trials,
        verbose=False
    )
    
    # Return only serializable data
    return (freq, {
        'n_spikes': result['n_spikes'],
        'n_active': result['n_active'],
        'df': result['df'].copy() if 'df' in result else pd.DataFrame(),
    })


# =============================================================================
# Main Execution
# =============================================================================

def run_parallel_test(n_workers=6, n_trials=2, freq_list=None):
    """
    Run parallel frequency sweep test.
    
    Parameters
    ----------
    n_workers : int
        Number of parallel workers
    n_trials : int
        Trials per frequency
    freq_list : list, optional
        Frequencies to test (default: [10, 50, 100, 150, 200])
    
    Returns
    -------
    dict : Results organized by frequency
    """
    
    if freq_list is None:
        freq_list = [10, 50, 100, 150, 200]
    
    print("=" * 70)
    print("Parallel Test: Experiment 1 - Sugar GRN Activation")
    print("=" * 70)
    
    # Load data
    print("\n[1/4] Loading connectivity data...")
    t0 = time.time()
    data = load_all_data(CONFIG)
    print(f"      Loaded: {data['n_neurons']:,} neurons, {len(data['df_conn']):,} connections")
    print(f"      Time: {time.time()-t0:.1f}s")
    
    # Identify column names
    print("\n[2/4] Preparing configuration...")
    df_conn = data['df_conn']
    cols = df_conn.columns.tolist()
    
    config_dict = {
        'pre_col': next(c for c in cols if 'pre' in c.lower() and 'root' in c.lower()),
        'post_col': next(c for c in cols if 'post' in c.lower() and 'root' in c.lower()),
        'weight_col': next(c for c in cols if 'syn_count' in c.lower()),
        'nt_prob_cols': {
            'gaba': next((c for c in cols if 'gaba' in c.lower()), None),
            'glut': next((c for c in cols if 'glut' in c.lower()), None),
        }
    }
    
    # Prepare tasks
    tasks = [
        (freq, NEU_SUGAR_LEFT, data, config_dict, DEFAULT_PARAMS, n_trials)
        for freq in freq_list
    ]
    
    print(f"\n[3/4] Running parallel simulation...")
    print(f"      Frequencies: {freq_list}")
    print(f"      Trials/freq: {n_trials}")
    print(f"      Workers: {n_workers}")
    print(f"      Total tasks: {len(tasks)}")
    print(f"\n      Note: Each worker rebuilds network (~8s overhead)")
    
    # Run parallel
    t_start = time.time()
    
    results = Parallel(n_jobs=n_workers, verbose=10)(
        delayed(run_frequency_worker)(*task)
        for task in tasks
    )
    
    parallel_time = time.time() - t_start
    
    # Organize results
    results_dict = dict(results)
    
    # Print results
    print("\n" + "=" * 70)
    print("[4/4] Results Summary")
    print("=" * 70)
    
    print(f"\nPerformance:")
    print(f"  Total time: {parallel_time/60:.2f} minutes")
    print(f"  Time/freq:  {parallel_time/len(freq_list):.1f} seconds")
    
    print(f"\nResponsive neurons by frequency:")
    for freq in freq_list:
        res = results_dict[freq]
        print(f"  {freq:3d} Hz: {res['n_active']:4d} neurons, {res['n_spikes']:6d} spikes")
    
    # Calculate baseline comparison (from notebook: 9.3 min for 5 freqs × 2 trials)
    baseline_time = 9.3  # minutes
    speedup = baseline_time / (parallel_time / 60)
    
    print(f"\nSpeedup Analysis:")
    print(f"  Serial baseline:  {baseline_time:.1f} minutes")
    print(f"  Parallel ({n_workers} cores): {parallel_time/60:.2f} minutes")
    print(f"  Speedup:          {speedup:.2f}×")
    
    if speedup > 3.0:
        print(f"  ✅ Excellent speedup!")
    elif speedup > 2.0:
        print(f"  ✓ Good speedup")
    else:
        print(f"  ⚠ Speedup lower than expected (overhead may be high)")
    
    # Extrapolate to cloud
    cloud_cores = 48
    extrapolated_speedup = speedup * (cloud_cores / n_workers)
    cloud_time = baseline_time / extrapolated_speedup
    
    print(f"\n48-core cloud prediction:")
    print(f"  Expected speedup: {extrapolated_speedup:.1f}× (linear scaling)")
    print(f"  Expected time:    {cloud_time:.2f} minutes")
    
    return results_dict


# =============================================================================
# Script Entry Point
# =============================================================================

def main():
    """Main entry point with argument parsing."""
    
    parser = argparse.ArgumentParser(
        description='Parallel test for Experiment 1'
    )
    parser.add_argument(
        '--workers', type=int, default=6,
        help='Number of parallel workers (default: 6)'
    )
    parser.add_argument(
        '--trials', type=int, default=2,
        help='Trials per frequency (default: 2)'
    )
    parser.add_argument(
        '--freqs', type=int, nargs='+', default=[10, 50, 100, 150, 200],
        help='Frequencies to test (default: 10 50 100 150 200)'
    )
    parser.add_argument(
        '--output', type=str, default='test_parallel_results.pkl',
        help='Output file for results'
    )
    
    args = parser.parse_args()
    
    # Run test
    results = run_parallel_test(
        n_workers=args.workers,
        n_trials=args.trials,
        freq_list=args.freqs
    )
    
    # Save results
    output_path = Path(args.output)
    with open(output_path, 'wb') as f:
        pickle.dump(results, f)
    
    print(f"\n✅ Results saved to: {output_path}")
    print("\nTest complete!")


if __name__ == '__main__':
    main()