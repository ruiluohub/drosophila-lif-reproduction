"""
Parallel Experiment Functions for Exp2 and Exp3

Implements sufficiency and necessity tests with joblib parallelization.
Each worker independently builds network and processes one neuron-frequency pair.
"""

import numpy as np
import pandas as pd
from brian2 import Hz, ms
from time import time
import gc


# =============================================================================
# Worker Functions (for joblib parallel)
# =============================================================================

def run_exp2_worker(neuron_id, freq, data, params, target_neurons, n_trials):
    """
    Exp2 worker: Test if single neuron activates target.
    
    Each worker independently builds network and runs simulation.
    Optimized data (68MB) makes network building faster (~7s vs ~28s).
    
    Parameters
    ----------
    neuron_id : int
        FlyWire ID of neuron to activate
    freq : int
        Activation frequency (Hz)
    data : dict
        Preprocessed data from load_simulation_data()
    params : dict
        Simulation parameters
    target_neurons : list
        Target neuron IDs to monitor (e.g., MN9)
    n_trials : int
        Number of trials
    
    Returns
    -------
    tuple : (neuron_id, freq, {'mean': float, 'std': float})
    """
    from flylif.core.network import build_network
    from flylif.core.simulation import run_simulation
    
    # Build network (unavoidable in multiprocessing)
    columns = data['columns']
    net = build_network(
        data=data,
        pre_col=columns['pre_col'],
        post_col=columns['post_col'],
        weight_col=columns['weight_col'],
        nt_prob_cols=columns.get('nt_prob_cols', {}),
        params=params,
        syn_threshold=5,
        verbose=False
    )
    
    # Run simulation
    result = run_simulation(
        net_components=net,
        neu_exc=[neuron_id],
        params={'r_poi': freq * Hz},
        n_trials=n_trials,
        verbose=False
    )
    
    df = result['df']
    duration_s = float(params['t_run'] / ms) / 1000
    
    # Calculate target neuron firing rates per trial
    target_rates = []
    for trial in range(n_trials):
        trial_df = df[df['trial'] == trial]
        count = 0
        for tid in target_neurons:
            count += len(trial_df[trial_df['flywire_id'] == tid])
        target_rates.append(count / duration_s)
    
    stats = {
        'mean': np.mean(target_rates),
        'std': np.std(target_rates),
    }
    
    # Cleanup
    del net
    del result
    gc.collect()
    
    return (neuron_id, freq, stats)


def run_exp3_worker(neuron_to_silence, freq, neu_exc, data, params, 
                    target_neurons, n_trials, control_mean):
    """
    Exp3 worker: Test if neuron is required (silencing experiment).
    
    Activates neu_exc continuously, silences one neuron, measures target response.
    
    Parameters
    ----------
    neuron_to_silence : int
        Neuron to silence
    freq : int
        Activation frequency (Hz)
    neu_exc : list
        Neurons to continuously activate (e.g., 21 Sugar GRNs)
    data : dict
        Preprocessed data
    params : dict
        Parameters
    target_neurons : list
        Target neurons to monitor
    n_trials : int
        Number of trials
    control_mean : float
        Control MN9 firing rate (for relative % calculation)
    
    Returns
    -------
    tuple : (neuron_id, freq, {'mean': float, 'std': float, 'relative_%': float})
    """
    from flylif.core.network import build_network
    from flylif.core.simulation import run_simulation
    
    # Build network
    columns = data['columns']
    net = build_network(
        data=data,
        pre_col=columns['pre_col'],
        post_col=columns['post_col'],
        weight_col=columns['weight_col'],
        nt_prob_cols=columns.get('nt_prob_cols', {}),
        params=params,
        syn_threshold=5,
        verbose=False
    )
    
    # Run simulation with silencing
    result = run_simulation(
        net_components=net,
        neu_exc=neu_exc,
        neu_slnc=[neuron_to_silence],
        params={'r_poi': freq * Hz},
        n_trials=n_trials,
        verbose=False
    )
    
    df = result['df']
    duration_s = float(params['t_run'] / ms) / 1000
    
    # Calculate target rates
    target_rates = []
    for trial in range(n_trials):
        trial_df = df[df['trial'] == trial]
        count = 0
        for tid in target_neurons:
            count += len(trial_df[trial_df['flywire_id'] == tid])
        target_rates.append(count / duration_s)
    
    slnc_mean = np.mean(target_rates)
    slnc_std = np.std(target_rates)
    
    # Calculate relative %
    if control_mean > 0:
        relative_pct = (slnc_mean / control_mean) * 100
    else:
        relative_pct = 100.0
    
    stats = {
        'mean': slnc_mean,
        'std': slnc_std,
        'relative_%': relative_pct,
    }
    
    # Cleanup
    del net
    del result
    gc.collect()
    
    return (neuron_to_silence, freq, stats)


# =============================================================================
# Main Functions (parallel orchestration)
# =============================================================================

def run_exp2_parallel(data, neurons_to_test, freqs, target_neurons, 
                      params, n_trials=10, n_workers=3, verbose=True):
    """
    Run Exp2 (sufficiency test) with parallel execution.
    
    Tests if individual neurons can activate target neurons when stimulated.
    Parallelizes across all neuron-frequency combinations.
    
    Parameters
    ----------
    data : dict
        Preprocessed data from load_simulation_data()
    neurons_to_test : list
        Neuron IDs to test (e.g., top_200_neurons)
    freqs : list
        Frequencies to test (Hz)
    target_neurons : list
        Target neurons to monitor (e.g., MN9)
    params : dict
        DEFAULT_PARAMS
    n_trials : int, default=10
        Trials per condition
    n_workers : int, default=3
        Number of parallel workers
    verbose : bool
    
    Returns
    -------
    dict : {
        freq: {neuron_id: {'mean': float, 'std': float}}
    }
    
    Examples
    --------
    >>> results = run_exp2_parallel(
    ...     DATA, top_200[:20], [50, 100], [MN9_ID], 
    ...     DEFAULT_PARAMS, n_trials=1, n_workers=3
    ... )
    """
    from joblib import Parallel, delayed
    
    if verbose:
        print("\n" + "=" * 70)
        print("Exp2: Sufficiency Test (Parallel)")
        print("=" * 70)
        print(f"  Neurons: {len(neurons_to_test)}")
        print(f"  Frequencies: {freqs}")
        print(f"  Trials/condition: {n_trials}")
        print(f"  Workers: {n_workers}")
        print(f"  Total tasks: {len(neurons_to_test) * len(freqs)}")
    
    # Create tasks
    tasks = [(nid, freq, data, params, target_neurons, n_trials)
             for freq in freqs
             for nid in neurons_to_test]
    
    if verbose:
        print(f"\n  Running {len(tasks)} tasks...")
        t0 = time()
    
    # Parallel execution
    results_list = Parallel(n_jobs=n_workers, verbose=5)(
        delayed(run_exp2_worker)(*task) for task in tasks
    )
    
    if verbose:
        elapsed = time() - t0
        print(f"\n  ✅ Complete! Total: {elapsed/60:.1f} min")
    
    # Organize results by frequency
    results = {freq: {} for freq in freqs}
    for neuron_id, freq, stats in results_list:
        results[freq][neuron_id] = stats
    
    return results


def run_exp3_parallel(data, neu_exc, neurons_to_silence, freqs, 
                      target_neurons, params, n_trials=10, 
                      n_workers=3, verbose=True):
    """
    Run Exp3 (necessity test) with parallel execution.
    
    Activates neu_exc continuously, silences neurons individually.
    First obtains control (no silencing), then parallelizes silencing tests.
    
    Parameters
    ----------
    data : dict
        Preprocessed data
    neu_exc : list
        Neurons to continuously activate (e.g., Sugar GRNs)
    neurons_to_silence : list
        Neurons to individually silence
    freqs : list
        Frequencies (Hz)
    target_neurons : list
        Target neurons to monitor
    params : dict
        Parameters
    n_trials : int
        Trials per condition
    n_workers : int
        Number of workers
    verbose : bool
    
    Returns
    -------
    dict : {
        freq: {
            'control': {'mean': float, 'std': float},
            'silencing': {neuron_id: {'mean': float, 'std': float, 'relative_%': float}}
        }
    }
    """
    from joblib import Parallel, delayed
    from flylif.core.network import build_network
    from flylif.core.simulation import run_simulation
    
    if verbose:
        print("\n" + "=" * 70)
        print("Exp3: Necessity Test (Parallel)")
        print("=" * 70)
        print(f"  Activated: {len(neu_exc)} neurons")
        print(f"  To silence: {len(neurons_to_silence)}")
        print(f"  Frequencies: {freqs}")
        print(f"  Trials/condition: {n_trials}")
        print(f"  Workers: {n_workers}")
    
    results = {}
    duration_s = float(params['t_run'] / ms) / 1000
    
    # Step 1: Get controls (serial, fast)
    if verbose:
        print(f"\n  [1/2] Getting controls...")
        t0 = time()
    
    # Build network once for controls
    columns = data['columns']
    net = build_network(
        data=data,
        pre_col=columns['pre_col'],
        post_col=columns['post_col'],
        weight_col=columns['weight_col'],
        nt_prob_cols=columns.get('nt_prob_cols', {}),
        params=params,
        syn_threshold=5,
        verbose=False
    )
    
    for freq in freqs:
        result = run_simulation(
            net_components=net,
            neu_exc=neu_exc,
            params={'r_poi': freq * Hz},
            n_trials=n_trials,
            verbose=False
        )
        
        df = result['df']
        ctrl_rates = []
        for trial in range(n_trials):
            trial_df = df[df['trial'] == trial]
            count = sum(len(trial_df[trial_df['flywire_id'] == tid]) 
                       for tid in target_neurons)
            ctrl_rates.append(count / duration_s)
        
        control_mean = np.mean(ctrl_rates)
        control_std = np.std(ctrl_rates)
        
        results[freq] = {
            'control': {'mean': control_mean, 'std': control_std},
            'silencing': {}
        }
        
        if verbose:
            print(f"    {freq} Hz: {control_mean:.1f} ± {control_std:.1f} Hz")
    
    del net
    gc.collect()
    
    if verbose:
        print(f"  ✅ Controls done ({time()-t0:.1f}s)")
    
    # Step 2: Silencing tests (parallel)
    if verbose:
        print(f"\n  [2/2] Silencing tests...")
        print(f"    Total tasks: {len(neurons_to_silence) * len(freqs)}")
        t0 = time()
    
    # Create tasks
    tasks = []
    for freq in freqs:
        control_mean = results[freq]['control']['mean']
        for nid in neurons_to_silence:
            tasks.append((nid, freq, neu_exc, data, params, 
                         target_neurons, n_trials, control_mean))
    
    # Parallel execution
    results_list = Parallel(n_jobs=n_workers, verbose=5)(
        delayed(run_exp3_worker)(*task) for task in tasks
    )
    
    if verbose:
        elapsed = time() - t0
        print(f"\n  ✅ Complete! Silencing: {elapsed/60:.1f} min")
    
    # Organize results
    for neuron_id, freq, stats in results_list:
        results[freq]['silencing'][neuron_id] = stats
    
    return results


# =============================================================================
# Baseline comparison helpers
# =============================================================================

def run_exp2_baseline_unoptimized(data, neurons_to_test, freqs, target_neurons,
                                  params, n_trials=1, verbose=True):
    """
    Baseline: Unoptimized version that rebuilds network every time.
    
    Only for performance comparison. DO NOT use for actual experiments.
    
    Returns
    -------
    float : Total time in seconds
    """
    from flylif.core.network import build_network
    from flylif.core.simulation import run_simulation
    
    if verbose:
        print("\n" + "=" * 70)
        print("Baseline: Unoptimized (Rebuild Every Time)")
        print("=" * 70)
        print(f"  Neurons: {len(neurons_to_test)}")
        print(f"  Frequencies: {freqs}")
        print(f"  Trials: {n_trials}")
        print(f"  ⚠️  This will be SLOW (rebuilds {len(neurons_to_test)*len(freqs)} times)")
    
    t0_total = time()
    columns = data['columns']
    
    count = 0
    total_tasks = len(neurons_to_test) * len(freqs)
    
    for freq in freqs:
        for neuron_id in neurons_to_test:
            count += 1
            
            # Rebuild network every time (inefficient!)
            net = build_network(
                data=data,
                pre_col=columns['pre_col'],
                post_col=columns['post_col'],
                weight_col=columns['weight_col'],
                nt_prob_cols=columns.get('nt_prob_cols', {}),
                params=params,
                syn_threshold=5,
                verbose=False
            )
            
            result = run_simulation(
                net_components=net,
                neu_exc=[neuron_id],
                params={'r_poi': freq * Hz},
                n_trials=n_trials,
                verbose=False
            )
            
            del net
            gc.collect()
            
            if verbose and count % 10 == 0:
                elapsed = time() - t0_total
                eta = elapsed / count * (total_tasks - count)
                print(f"\r  Progress: {count}/{total_tasks} "
                      f"({elapsed:.0f}s, ETA: {eta:.0f}s)", end='', flush=True)
    
    total_time = time() - t0_total
    
    if verbose:
        print(f"\n  ✅ Baseline complete: {total_time/60:.1f} min")
    
    return total_time


# =============================================================================
# Testing
# =============================================================================

if __name__ == '__main__':
    print("\n" + "=" * 70)
    print("Testing experiments.py (Parallel Version)")
    print("=" * 70)
    print("\n⚠️  This requires DATA and params")
    print("   Run from exp1_clean_test.ipynb after loading data")
    print("\nExample usage:")
    print("""
from flylif.core.experiments import run_exp2_parallel

results = run_exp2_parallel(
    DATA, top_200[:20], [50, 100], [MN9_ID],
    DEFAULT_PARAMS, n_trials=1, n_workers=3
)
""")