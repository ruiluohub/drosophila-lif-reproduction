"""
Parallel Experiment Functions for Exp2 and Exp3

Implements sufficiency and necessity tests with joblib parallelization.
Each worker independently builds network and processes one neuron-frequency pair.

CHANGELOG:
- Added CheckpointManager integration for resume capability
- Added start_scope() in workers for memory isolation
- Added task_key tracking for checkpoint identification
"""

import numpy as np
import pandas as pd
from brian2 import Hz, ms, start_scope
from time import time
import gc


# =============================================================================
# Worker Functions (for joblib parallel)
# =============================================================================
# Worker function with memory optimization
def run_exp1_worker(freq, neu_exc, data, params, n_trials):
    """
    Exp1 worker: Single frequency simulation.
    
    Optimized with start_scope() and minimal return data.
    """

    start_scope()  # Clear Brian2 global state
    
    from flylif.core.network import build_network
    from flylif.core.simulation import run_simulation
    
    # Build network
    columns = data['columns']
    net_components = build_network(
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
        net_components=net_components,
        neu_exc=neu_exc,
        params={'r_poi': freq * Hz},
        n_trials=n_trials,
        verbose=False
    )
    
    # Extract only essential data
    result_clean = {
        'n_active': result['n_active'],
        'n_spikes': result['n_spikes'],
        'df': result['df'].copy(),
    }
    
    # Explicit cleanup
    del net_components
    del result
    gc.collect()
    
    return (freq, result_clean)




# Worker function
def shuffled_control_worker(sim_idx, shuffle, data, params, 
                            neu_exc, target_id, task_key=None):
    """
    Shuffled connectivity control worker (corrected).
    
    Shuffles connection topology by randomizing post-synaptic targets
    while maintaining global connectivity statistics.
    
    Parameters
    ----------
    sim_idx : int
        Simulation index (0-99)
    shuffle : bool
        If True, shuffle post-synaptic targets in df_conn
    data : dict
        Original DATA from load_simulation_data()
    params : dict
        DEFAULT_PARAMS
    neu_exc : list
        Neurons to activate (Sugar GRNs)
    target_id : int
        Target neuron to monitor (MN9)
    task_key : str
        Checkpoint key
    
    Returns
    -------
    tuple : (sim_idx, shuffle, mn9_activated, task_key)
    """
    from brian2 import start_scope, Hz
    start_scope()
    
    from flylif.core.network import build_network
    from flylif.core.simulation import run_simulation
    import gc
    
    # === Shuffle at data level (changes topology) ===
    if shuffle:
        # Create shuffled data copy
        data_use = data.copy()
        df_conn_shuffled = data['df_conn'].copy()
        
        # Shuffle post-synaptic targets
        # This changes WHO connects to WHOM
        post_col = data['columns']['post_col']
        post_ids = df_conn_shuffled[post_col].values.copy()
        
        np.random.seed(sim_idx)  # Reproducible shuffle
        np.random.shuffle(post_ids)
        
        df_conn_shuffled[post_col] = post_ids
        data_use['df_conn'] = df_conn_shuffled
    else:
        # Use original data
        data_use = data
    
    # Build network (with original or shuffled connectivity)
    columns = data_use['columns']
    net = build_network(
        data=data_use,
        pre_col=columns['pre_col'],
        post_col=columns['post_col'],
        weight_col=columns['weight_col'],
        nt_prob_cols=columns.get('nt_prob_cols', {}),
        params=params,
        syn_threshold=5,
        verbose=False
    )
    
    # Run simulation (100 Hz sugar activation, 1 trial)
    result = run_simulation(
        net_components=net,
        neu_exc=neu_exc,
        params={'r_poi': 100 * Hz},
        n_trials=1,
        verbose=False
    )
    
    # Check if MN9 activated (>0 Hz)
    df = result['df']
    mn9_activated = len(df[df['flywire_id'] == target_id]) > 0
    
    # Cleanup
    del net, result
    if shuffle:
        del data_use, df_conn_shuffled
    gc.collect()
    
    return (sim_idx, shuffle, mn9_activated, task_key)

def compute_control_worker(freq, neu_exc, data, params, target_neurons, n_trials):
    """
    Worker for computing single control baseline.
    Each worker builds its own network.
    """
    start_scope()  # Clean state
    
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
    
    # Run simulation
    result = run_simulation(
        net_components=net,
        neu_exc=neu_exc,
        params={'r_poi': freq * Hz},
        n_trials=n_trials,
        verbose=False
    )
    
    df = result['df']
    duration_s = float(params['t_run'] / ms) / 1000
    
    # Calculate target rates
    ctrl_rates = []
    for trial in range(n_trials):
        trial_df = df[df['trial'] == trial]
        count = sum(len(trial_df[trial_df['flywire_id'] == tid]) 
                   for tid in target_neurons)
        ctrl_rates.append(count / duration_s)
    
    control_mean = np.mean(ctrl_rates)
    control_std = np.std(ctrl_rates)
    
    # Cleanup
    del net
    del result
    gc.collect()
    
    return (freq, {'mean': control_mean, 'std': control_std})



def run_exp2_worker(neuron_id, freq, data, params, target_neurons, n_trials, task_key=None):
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
    task_key : str, optional
        Unique identifier for checkpoint tracking
    
    Returns
    -------
    tuple : (neuron_id, freq, {'mean': float, 'std': float}, task_key)
    """
    start_scope()  # Clear Brian2 global state for memory isolation
    
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
    
    return (neuron_id, freq, stats, task_key)


def run_exp3_worker(neuron_to_silence, freq, neu_exc, data, params, 
                    target_neurons, n_trials, control_mean, task_key=None):
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
    task_key : str, optional
        Unique identifier for checkpoint tracking
    
    Returns
    -------
    tuple : (neuron_id, freq, {'mean': float, 'std': float, 'relative_%': float}, task_key)
    """
    start_scope()  # Clear Brian2 global state for memory isolation
    
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
    
    return (neuron_to_silence, freq, stats, task_key)




# =============================================================================
# Main Functions (parallel orchestration with checkpoint support)
# =============================================================================
# Batch runner with checkpoirun_exp1_batchnt and memory monitoring
def run_exp1_batch(freq_list, batch_name, n_workers=3, n_trials=10, 
              checkpoint_dir='./checkpoints/exp1'):
    """
    Run batch with checkpoint and memory monitoring.
    
    Parameters
    ----------
    freq_list : list
        List of frequencies to test
    batch_name : str
        Name of the batch (for logging)
    n_workers : int
        Number of parallel workers
    n_trials : int
        Trials per frequency
    checkpoint_dir : str or Path
        Checkpoint directory
    
    Returns
    -------
    dict : {freq: result}
    """
    from flylif.utils.checkpoint import CheckpointManager
    
    print("\n" + "=" * 70)
    print(f"Batch: {batch_name}")
    print("=" * 70)
    
    # Initialize checkpoint
    ckpt = CheckpointManager(checkpoint_dir)
    
    # Check remaining tasks
    remaining = ckpt.get_remaining(freq_list)
    
    if not remaining:
        print(f"✅ All frequencies already completed, loading from checkpoint...")
        return ckpt.load_all_completed()
    
    print(f"\nConfiguration:")
    print(f"  Frequencies: {remaining}")
    print(f"  Trials/freq: {n_trials}")
    print(f"  Workers: {n_workers}")
    print(f"  Checkpoint: {checkpoint_dir}")
    
    # Memory check
    is_safe, msg = check_memory_safe(threshold=80.0, swap_threshold=0.5)
    print(f"\n  Memory status: {msg}")
    
    if not is_safe:
        print(f"  ⚠️  Memory not safe, recommend cleanup first")
        memory_cleanup()
        print_memory("    After cleanup: ")
    
    # Prepare tasks
    tasks = [(freq, NEU_SUGAR_LEFT, DATA, DEFAULT_PARAMS, n_trials) 
             for freq in remaining]
    
    print(f"\n  Running {len(tasks)} tasks...")
    
    with MemoryMonitor(label=batch_name):
        t0 = time()
        
        results = Parallel(n_jobs=n_workers, verbose=5, max_nbytes='100M')(
            delayed(run_exp1_worker)(*task) for task in tasks
        )
        
        batch_time = time() - t0
    
    # Save to checkpoint
    for freq, result in results:
        ckpt.save(freq, result)
    
    print(f"\n  ✅ Batch complete in {batch_time/60:.2f} min")
    print_memory("  Final memory: ")
    
    # Cleanup
    del results
    gc.collect()
    
    return ckpt.load_all_completed()


    
def run_exp2_parallel(data, neurons_to_test, freqs, target_neurons, 
                      params, n_trials=10, n_workers=3, 
                      checkpoint_dir=None, verbose=True):
    """
    Run Exp2 (sufficiency test) with parallel execution and checkpoint support.
    
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
    checkpoint_dir : str or Path, optional
        Directory for checkpoint files. If provided, enables resume capability.
    verbose : bool
    
    Returns
    -------
    dict : {
        freq: {neuron_id: {'mean': float, 'std': float}}
    }
    
    Examples
    --------
    >>> results = run_exp2_parallel(
    ...     DATA, top_200, [50, 100], [MN9_ID], 
    ...     DEFAULT_PARAMS, n_trials=10, n_workers=3,
    ...     checkpoint_dir='./checkpoints/exp2'  # Enable checkpoint
    ... )
    """
    from joblib import Parallel, delayed
    
    if verbose:
        print("\n" + "=" * 70)
        print("Exp2: Sufficiency Test (Parallel with Checkpoint)")
        print("=" * 70)
        print(f"  Neurons: {len(neurons_to_test)}")
        print(f"  Frequencies: {freqs}")
        print(f"  Trials/condition: {n_trials}")
        print(f"  Workers: {n_workers}")
        print(f"  Total tasks: {len(neurons_to_test) * len(freqs)}")
        if checkpoint_dir:
            print(f"  Checkpoint: {checkpoint_dir}")
    
    # Initialize checkpoint manager
    if checkpoint_dir:
        from flylif.utils.checkpoint import CheckpointManager
        from pathlib import Path
        ckpt = CheckpointManager(Path(checkpoint_dir))
    else:
        ckpt = None
    
    # Create tasks with checkpoint filtering
    all_task_specs = []
    for freq in freqs:
        for nid in neurons_to_test:
            task_key = f'{nid}_{freq}'
            
            # Skip if already completed
            if ckpt and ckpt.is_completed(task_key):
                if verbose:
                    print(f"  ⏭️  Skip {task_key}")
                continue
            
            all_task_specs.append((nid, freq, data, params, target_neurons, n_trials, task_key))
    
    tasks = all_task_specs
    
    if verbose:
        n_total = len(neurons_to_test) * len(freqs)
        n_remaining = len(tasks)
        n_completed = n_total - n_remaining
        print(f"\n  Progress: {n_completed}/{n_total} completed, {n_remaining} remaining")
    
    # Early exit if all completed
    if len(tasks) == 0:
        if verbose:
            print(f"\n  ✅ All tasks already completed, loading from checkpoint...")
        return _load_exp2_results_from_checkpoint(ckpt, freqs, neurons_to_test)
    
    if verbose:
        print(f"\n  Running {len(tasks)} tasks...")
        t0 = time()
    
    if verbose:
        print(f"\n  Running {len(tasks)} tasks...")
        t0 = time()
    
    # Parallel execution with real-time checkpoint saving
    results_list = []
    
    with Parallel(n_jobs=n_workers, verbose=5, max_nbytes='100M',
                  return_as='generator') as parallel:
        
        generator = parallel(delayed(run_exp2_worker)(*task) for task in tasks)
        
        for result in generator:
            neuron_id, freq, stats, task_key = result
            
            # Save checkpoint immediately after each task
            if ckpt:
                ckpt.save(task_key, {
                    'neuron_id': neuron_id,
                    'freq': freq,
                    'stats': stats
                })
            
            results_list.append(result)
    
    if verbose:
        elapsed = time() - t0
        print(f"\n  ✅ Complete! Total: {elapsed/60:.1f} min")
        if ckpt:
            print(f"     Checkpoints saved: {len(results_list)}")
    
    # Organize results (include both new and previously completed)
    results = {freq: {} for freq in freqs}
    
    # Add new results
    for neuron_id, freq, stats, _ in results_list:
        results[freq][neuron_id] = stats
    
    # Add previously completed results
    if ckpt:
        all_completed = ckpt.load_all_completed()
        for task_key, data in all_completed.items():
            nid = data['neuron_id']
            freq_val = data['freq']
            stats = data['stats']
            
            if freq_val in results:
                results[freq_val][nid] = stats
    
    return results

def run_exp3_parallel(data, neu_exc, neurons_to_silence, freqs, 
                      target_neurons, params, n_trials=10, 
                      n_workers=3, checkpoint_dir=None, 
                      precomputed_controls=None, verbose=True):
    """
    Run Exp3 (necessity test) with parallel execution and checkpoint support.
    
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
    checkpoint_dir : str or Path, optional
        Directory for checkpoint files. If provided, enables resume capability.
    precomputed_controls : dict, optional
        Pre-computed control baselines: {freq: {'mean': float, 'std': float}}
        If provided, skips control calculation (saves time).
    verbose : bool
    
    Returns
    -------
    dict : {
        freq: {
            'control': {'mean': float, 'std': float},
            'silencing': {neuron_id: {'mean': float, 'std': float, 'relative_%': float}}
        }
    }
    
    Examples
    --------
    >>> # With precomputed controls (recommended):
    >>> controls = {50: {'mean': 80.5, 'std': 4.2}, ...}
    >>> results = run_exp3_parallel(
    ...     DATA, SUGAR_GRNs, top_200, [50, 100], [MN9_ID],
    ...     DEFAULT_PARAMS, n_trials=10, n_workers=3,
    ...     checkpoint_dir='./checkpoints/exp3',
    ...     precomputed_controls=controls
    ... )
    """
    from joblib import Parallel, delayed
    from flylif.core.network import build_network
    from flylif.core.simulation import run_simulation
    
    if verbose:
        print("\n" + "=" * 70)
        print("Exp3: Necessity Test (Parallel with Checkpoint)")
        print("=" * 70)
        print(f"  Activated: {len(neu_exc)} neurons")
        print(f"  To silence: {len(neurons_to_silence)}")
        print(f"  Frequencies: {freqs}")
        print(f"  Trials/condition: {n_trials}")
        print(f"  Workers: {n_workers}")
        if checkpoint_dir:
            print(f"  Checkpoint: {checkpoint_dir}")
        if precomputed_controls:
            print(f"  Using precomputed controls: ✅")
    
    # Initialize checkpoint manager
    if checkpoint_dir:
        from flylif.utils.checkpoint import CheckpointManager
        from pathlib import Path
        ckpt = CheckpointManager(Path(checkpoint_dir))
    else:
        ckpt = None
    
    results = {}
    duration_s = float(params['t_run'] / ms) / 1000
    
    # ========================================================================
    # Step 1: Get controls
    # ========================================================================
    
    if precomputed_controls:
        # Use provided controls (skip computation)
        if verbose:
            print(f"\n  [1/2] Using precomputed controls...")
        
        for freq in freqs:
            if freq not in precomputed_controls:
                raise ValueError(
                    f"Missing control for {freq}Hz in precomputed_controls. "
                    f"Available: {list(precomputed_controls.keys())}"
                )
            
            results[freq] = {
                'control': precomputed_controls[freq],
                'silencing': {}
            }
            
            if verbose:
                ctrl = precomputed_controls[freq]
                print(f"    {freq} Hz: {ctrl['mean']:.1f} ± {ctrl['std']:.1f} Hz")
        
        if verbose:
            print(f"  ✅ Controls loaded (skipped computation)")
    
    else:
        # Compute controls (original logic)
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
    
    # ========================================================================
    # Step 2: Silencing tests (parallel) with checkpoint filtering
    # ========================================================================
    
    if verbose:
        print(f"\n  [2/2] Silencing tests...")
    
    # Create tasks with checkpoint filtering
    all_task_specs = []
    for freq in freqs:
        control_mean = results[freq]['control']['mean']
        for nid in neurons_to_silence:
            task_key = f'{nid}_{freq}'
            
            # Skip if already completed
            if ckpt and ckpt.is_completed(task_key):
                if verbose:
                    print(f"    ⏭️  Skip {task_key}")
                continue
            
            all_task_specs.append((nid, freq, neu_exc, data, params, 
                                  target_neurons, n_trials, control_mean, task_key))
    
    tasks = all_task_specs
    
    if verbose:
        n_total = len(neurons_to_silence) * len(freqs)
        n_remaining = len(tasks)
        n_completed = n_total - n_remaining
        print(f"    Total tasks: {n_total}")
        print(f"    Completed: {n_completed}")
        print(f"    Remaining: {n_remaining}")
    
    # Early exit if all completed
    if len(tasks) == 0:
        if verbose:
            print(f"\n  ✅ All silencing tasks already completed, loading from checkpoint...")
        return _load_exp3_results_from_checkpoint(ckpt, results, freqs, neurons_to_silence)
    
    if verbose:
        t0 = time()
    
    # Parallel execution with real-time checkpoint saving
    results_list = []
    
    with Parallel(n_jobs=n_workers, verbose=5, max_nbytes='100M',
                  return_as='generator') as parallel:
        
        generator = parallel(delayed(run_exp3_worker)(*task) for task in tasks)
        
        for result in generator:
            neuron_id, freq, stats, task_key = result
            
            # Save checkpoint immediately after each task
            if ckpt:
                ckpt.save(task_key, {
                    'neuron_id': neuron_id,
                    'freq': freq,
                    'stats': stats
                })
            
            results_list.append(result)
    
    if verbose:
        elapsed = time() - t0
        print(f"\n  ✅ Complete! Silencing: {elapsed/60:.1f} min")
        if ckpt:
            print(f"     Checkpoints saved: {len(results_list)}")
    
    # Organize results (include both new and previously completed)
    # Add new results
    for neuron_id, freq, stats, _ in results_list:
        results[freq]['silencing'][neuron_id] = stats
    
    # Add previously completed results
    if ckpt:
        all_completed = ckpt.load_all_completed()
        for task_key, data_saved in all_completed.items():
            nid = data_saved['neuron_id']
            freq_val = data_saved['freq']
            stats = data_saved['stats']
            
            if freq_val in results and 'silencing' in results[freq_val]:
                results[freq_val]['silencing'][nid] = stats
    
    return results

# =============================================================================
# Helper Functions (for checkpoint loading)
# =============================================================================

def _load_exp2_results_from_checkpoint(ckpt, freqs, neurons_to_test):
    """Load all Exp2 results from checkpoint."""
    results = {freq: {} for freq in freqs}
    
    all_completed = ckpt.load_all_completed()
    for task_key, data in all_completed.items():
        nid = data['neuron_id']
        freq = data['freq']
        stats = data['stats']
        
        if freq in results:
            results[freq][nid] = stats
    
    return results


def _load_exp3_results_from_checkpoint(ckpt, results_with_controls, freqs, neurons_to_silence):
    """Load all Exp3 silencing results from checkpoint (controls already in results)."""
    all_completed = ckpt.load_all_completed()
    
    for task_key, data in all_completed.items():
        nid = data['neuron_id']
        freq = data['freq']
        stats = data['stats']
        
        if freq in results_with_controls and 'silencing' in results_with_controls[freq]:
            results_with_controls[freq]['silencing'][nid] = stats
    
    return results_with_controls


# =============================================================================
# Baseline comparison helpers (unchanged)
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
# Exp4: Dual Modality Experiments
# =============================================================================

def run_dual_modality_worker(freq1, freq2, neu_exc1, neu_exc2, data, params,
                              target_neurons, n_trials, task_key=None):
    """
    Worker for dual modality experiment.
    
    Activates neu_exc1 at freq1 (r_poi) and neu_exc2 at freq2 (r_poi2).
    Measures target neuron (MN9) response.
    
    Parameters
    ----------
    freq1, freq2 : int
        Activation frequencies (Hz)
    neu_exc1, neu_exc2 : list
        Neuron IDs to activate
    data : dict
        Preprocessed connectivity data
    params : dict
        Simulation parameters
    target_neurons : list
        Neurons to monitor (e.g., MN9)
    n_trials : int
        Number of trials
    task_key : str, optional
        Unique identifier for checkpoint
    
    Returns
    -------
    tuple : (freq1, freq2, {'mean': float, 'std': float}, task_key)
    """
    start_scope()  # Clean state
    
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
    
    # Run simulation with dual activation
    result = run_simulation(
        net_components=net,
        neu_exc=neu_exc1,   # Activated at r_poi
        neu_exc2=neu_exc2,  # Activated at r_poi2
        params={'r_poi': freq1 * Hz, 'r_poi2': freq2 * Hz},
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
    
    return (freq1, freq2, stats, task_key)


def run_exp4_parallel(data, neu_exc1, neu_exc2, freq_range1, freq_range2,
                      target_neurons, params, n_trials=10, n_workers=4,
                      checkpoint_dir=None, exp_name='exp4', verbose=True):
    """
    Run dual modality experiment with checkpoint support.
    
    Tests interaction between two GRN populations (e.g., Sugar × Bitter).
    
    Parameters
    ----------
    data : dict
        Preprocessed data
    neu_exc1 : list
        First population (e.g., Sugar GRNs), activated at freq_range1
    neu_exc2 : list
        Second population (e.g., Bitter GRNs), activated at freq_range2
    freq_range1, freq_range2 : list
        Frequency ranges to test (Hz)
    target_neurons : list
        Target neurons to monitor
    params : dict
        Simulation parameters
    n_trials : int
        Trials per condition
    n_workers : int
        Number of parallel workers
    checkpoint_dir : str or Path, optional
        Checkpoint directory (enables resume)
    exp_name : str
        Experiment label (for logging)
    verbose : bool
    
    Returns
    -------
    dict : {
        'results': {freq1: {freq2: {'mean': float, 'std': float}}},
        'matrix': numpy array (freq_range2 × freq_range1),
        'baselines': {freq1: {'mean': float, 'std': float}}  # freq2=0 conditions
    }
    
    Examples
    --------
    >>> result = run_exp4_parallel(
    ...     DATA, SUGAR_GRNs, BITTER_GRNs, [0,50,100], [0,50,100],
    ...     [MN9_ID], DEFAULT_PARAMS, checkpoint_dir='./checkpoints/exp4a'
    ... )
    """
    from joblib import Parallel, delayed
    
    if verbose:
        print("\n" + "=" * 70)
        print(f"{exp_name}: Dual Modality Test (Parallel with Checkpoint)")
        print("=" * 70)
        print(f"  Population 1: {len(neu_exc1)} neurons")
        print(f"  Population 2: {len(neu_exc2)} neurons")
        print(f"  Freq range 1: {freq_range1}")
        print(f"  Freq range 2: {freq_range2}")
        print(f"  Total conditions: {len(freq_range1) * len(freq_range2)}")
        print(f"  Workers: {n_workers}")
        if checkpoint_dir:
            print(f"  Checkpoint: {checkpoint_dir}")
    
    # Initialize checkpoint
    if checkpoint_dir:
        from flylif.utils.checkpoint import CheckpointManager
        from pathlib import Path
        ckpt = CheckpointManager(Path(checkpoint_dir))
    else:
        ckpt = None
    
    # Create tasks with checkpoint filtering
    all_task_specs = []
    
    for f1 in freq_range1:
        for f2 in freq_range2:
            task_key = f'{f1}_{f2}'
            
            # Skip if completed
            if ckpt and ckpt.is_completed(task_key):
                if verbose:
                    print(f"  ⏭️  Skip {task_key}")
                continue
            
            all_task_specs.append((f1, f2, neu_exc1, neu_exc2, data, params,
                                  target_neurons, n_trials, task_key))
    
    tasks = all_task_specs
    
    if verbose:
        n_total = len(freq_range1) * len(freq_range2)
        n_remaining = len(tasks)
        n_completed = n_total - n_remaining
        print(f"\n  Progress: {n_completed}/{n_total} completed, {n_remaining} remaining")
    
    # Early exit if all completed
    if len(tasks) == 0:
        if verbose:
            print(f"\n  ✅ All tasks completed, loading from checkpoint...")
        return _load_exp4_results_from_checkpoint(ckpt, freq_range1, freq_range2)
    
    if verbose:
        t0 = time()
    
    # Parallel execution with real-time checkpoint
    results_list = []
    
    with Parallel(n_jobs=n_workers, verbose=5, max_nbytes='100M',
                  return_as='generator') as parallel:
        
        generator = parallel(delayed(run_dual_modality_worker)(*task) for task in tasks)
        
        for result in generator:
            freq1, freq2, stats, task_key = result
            
            # Save checkpoint immediately
            if ckpt:
                ckpt.save(task_key, {
                    'freq1': freq1,
                    'freq2': freq2,
                    'stats': stats
                })
            
            results_list.append(result)
    
    if verbose:
        elapsed = time() - t0
        print(f"\n  ✅ Complete! Total: {elapsed/60:.1f} min")
        if ckpt:
            print(f"     Checkpoints saved: {len(results_list)}")
    
    # Organize results
    results_dict = {}
    for freq1, freq2, stats, _ in results_list:
        if freq1 not in results_dict:
            results_dict[freq1] = {}
        results_dict[freq1][freq2] = stats
    
    # Add previously completed results
    if ckpt:
        all_completed = ckpt.load_all_completed()
        for task_key, data_saved in all_completed.items():
            f1 = data_saved['freq1']
            f2 = data_saved['freq2']
            stats = data_saved['stats']
            
            if f1 not in results_dict:
                results_dict[f1] = {}
            results_dict[f1][f2] = stats
    
    # Create 2D matrix
    matrix = np.zeros((len(freq_range2), len(freq_range1)))
    for i, f2 in enumerate(freq_range2):
        for j, f1 in enumerate(freq_range1):
            if f1 in results_dict and f2 in results_dict[f1]:
                matrix[i, j] = results_dict[f1][f2]['mean']
    
    # Extract baselines (freq2=0)
    baselines = {}
    for f1 in freq_range1:
        if f1 in results_dict and 0 in results_dict[f1]:
            baselines[f1] = results_dict[f1][0]
    
    return {
        'results': results_dict,
        'matrix': matrix,
        'baselines': baselines,
        'freq_range1': freq_range1,
        'freq_range2': freq_range2,
    }


def _load_exp4_results_from_checkpoint(ckpt, freq_range1, freq_range2):
    """Load all Exp4 results from checkpoint."""
    all_completed = ckpt.load_all_completed()
    
    results_dict = {}
    for task_key, data in all_completed.items():
        f1 = data['freq1']
        f2 = data['freq2']
        stats = data['stats']
        
        if f1 not in results_dict:
            results_dict[f1] = {}
        results_dict[f1][f2] = stats
    
    # Create matrix
    matrix = np.zeros((len(freq_range2), len(freq_range1)))
    for i, f2 in enumerate(freq_range2):
        for j, f1 in enumerate(freq_range1):
            if f1 in results_dict and f2 in results_dict[f1]:
                matrix[i, j] = results_dict[f1][f2]['mean']
    
    # Extract baselines
    baselines = {}
    for f1 in freq_range1:
        if f1 in results_dict and 0 in results_dict[f1]:
            baselines[f1] = results_dict[f1][0]
    
    return {
        'results': results_dict,
        'matrix': matrix,
        'baselines': baselines,
        'freq_range1': freq_range1,
        'freq_range2': freq_range2,
    }

# =============================================================================
# Testing
# =============================================================================

if __name__ == '__main__':
    print("\n" + "=" * 70)
    print("Testing experiments.py (Parallel Version with Checkpoint)")
    print("=" * 70)
    print("\n⚠️  This requires DATA and params")
    print("   Run from notebook after loading data")
    print("\nExample usage:")
    print("""
from flylif.core.experiments import run_exp2_parallel, run_exp3_parallel

# Exp2 with checkpoint
results = run_exp2_parallel(
    DATA, top_200[:20], [50, 100], [MN9_ID],
    DEFAULT_PARAMS, n_trials=10, n_workers=3,
    checkpoint_dir='./checkpoints/exp2_test'
)

# Exp3 with checkpoint
results = run_exp3_parallel(
    DATA, SUGAR_GRNs, top_200[:20], [50, 100], [MN9_ID],
    DEFAULT_PARAMS, n_trials=10, n_workers=3,
    checkpoint_dir='./checkpoints/exp3_test'
)
""")