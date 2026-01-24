"""
Brian2 Simulation Execution Module

Core simulation functions for running LIF network experiments.
Supports activation and silencing experiments with multiple trials.
"""

import numpy as np
import pandas as pd
from brian2 import PoissonInput, SpikeMonitor, Network
from brian2 import ms, Hz, mV
from time import time


def run_simulation(net_components, neu_exc, neu_exc2=None, neu_slnc=None,
                   params=None, duration=None, n_trials=None, verbose=True):
    """
    Run LIF network simulation with neuron activation/silencing.
    
    Executes multiple trials of network simulation with:
    - Primary activation: neu_exc neurons driven by Poisson input at r_poi
    - Secondary activation: neu_exc2 neurons driven at r_poi2 (optional)
    - Silencing: neu_slnc neurons have output synapses set to weight=0
    
    Parameters
    ----------
    net_components : dict
        Network components from build_network(), must contain:
        - net: Brian2 Network object
        - neu: NeuronGroup
        - syn: Synapses
        - flyid2i, i2flyid: ID mappings
        - params: default parameters
    neu_exc : list
        FlyWire IDs of neurons to activate (Poisson input at r_poi)
    neu_exc2 : list, optional
        FlyWire IDs for secondary activation (Poisson at r_poi2)
    neu_slnc : list, optional
        FlyWire IDs of neurons to silence (set output weights to 0)
    params : dict, optional
        Override parameters (e.g., r_poi, t_run)
    duration : Quantity, optional
        Simulation duration (overrides params['t_run'])
    n_trials : int, optional
        Number of trials (overrides params['n_run'])
    verbose : bool, default=True
        Print progress information
    
    Returns
    -------
    dict :
        - df: DataFrame with spike times and neuron IDs
        - n_spikes: total spike count
        - n_active: number of active neurons
        - n_trials: number of trials run
        - duration: simulation duration
        - stim_neurons: activated neuron IDs
        - silenced_neurons: silenced neuron IDs
        - params_used: actual parameters used
    
    Examples
    --------
    >>> # Simple activation
    >>> result = run_simulation(NET, neu_exc=SUGAR_GRNs, n_trials=10)
    
    >>> # Dual modality
    >>> result = run_simulation(NET, neu_exc=SUGAR_GRNs, neu_exc2=BITTER_GRNs,
    ...                         params={'r_poi': 100*Hz, 'r_poi2': 50*Hz})
    
    >>> # Silencing experiment
    >>> result = run_simulation(NET, neu_exc=SUGAR_GRNs, neu_slnc=[neuron_id])
    """
    neu_exc2 = neu_exc2 or []
    neu_slnc = neu_slnc or []
    
    net = net_components['net']
    neu = net_components['neu']
    syn = net_components['syn']
    flyid2i = net_components['flyid2i']
    i2flyid = net_components['i2flyid']
    
    # Merge parameters
    sim_params = net_components['params'].copy()
    if params:
        sim_params.update(params)
    
    if duration is None:
        duration = sim_params['t_run']
    if n_trials is None:
        n_trials = sim_params['n_run']
    
    # Convert FlyWire IDs to indices
    exc = [flyid2i[n] for n in neu_exc if n in flyid2i]
    exc2 = [flyid2i[n] for n in neu_exc2 if n in flyid2i]
    slnc = [flyid2i[n] for n in neu_slnc if n in flyid2i]
    
    if verbose:
        print(f"\n>>> Simulation Configuration")
        print(f"    Activated: {len(exc)}, Secondary: {len(exc2)}, Silenced: {len(slnc)}")
        print(f"    Duration: {duration}, Trials: {n_trials}")
        print(f"    Frequency: {sim_params['r_poi']}")
    
    # Precompute silencing synapse indices
    if len(slnc) > 0 and 'pre_syn_indices' in net_components:
        pre_syn_indices = net_components['pre_syn_indices']
        weights_array = net_components['weights_array']
        
        slnc_syn_indices = []
        for neuron_idx in slnc:
            if neuron_idx in pre_syn_indices:
                slnc_syn_indices.extend(pre_syn_indices[neuron_idx])
        
        slnc_syn_indices = np.array(slnc_syn_indices, dtype=np.int64) if slnc_syn_indices else np.array([], dtype=np.int64)
        
        if verbose and len(slnc) > 0:
            print(f"    Silenced synapses: {len(slnc_syn_indices)}")
        
        # Store original weights for restoration
        if len(slnc_syn_indices) > 0:
            original_weights_slnc = weights_array[slnc_syn_indices].copy()
    else:
        slnc_syn_indices = np.array([], dtype=np.int64)
    
    # Run trials
    all_spike_data = []
    
    if verbose:
        print(f"    🚀 Running...")
        t0_total = time()
    
    for trial_idx in range(n_trials):
        t0_trial = time()
        
        # Step 1: Restore network state
        net.restore('initial')
        
        # Step 2: Apply silencing (direct weight array assignment)
        if len(slnc_syn_indices) > 0:
            syn.w[slnc_syn_indices] = 0 * mV
        
        # Step 3: Create monitors and inputs
        spk_mon = SpikeMonitor(neu)
        
        pois = []
        for i in exc:
            p = PoissonInput(
                target=neu[i], 
                target_var='v', 
                N=1,
                rate=sim_params['r_poi'], 
                weight=sim_params['w_syn'] * sim_params['f_poi']
            )
            neu[i].rfc = 0 * ms
            pois.append(p)
        
        for i in exc2:
            p = PoissonInput(
                target=neu[i], 
                target_var='v', 
                N=1,
                rate=sim_params['r_poi2'], 
                weight=sim_params['w_syn'] * sim_params['f_poi']
            )
            neu[i].rfc = 0 * ms
            pois.append(p)
        
        # Step 4: Run
        net.add(spk_mon, *pois)
        net.run(duration)
        
        # Step 5: Extract results (batch processing)
        spike_i = np.array(spk_mon.i)
        spike_t = np.array(spk_mon.t)
        
        if len(spike_i) > 0:
            trial_data = pd.DataFrame({
                't': spike_t,
                'trial': trial_idx,
                'neuron_idx': spike_i
            })
            all_spike_data.append(trial_data)
        
        # Step 6: Cleanup
        net.remove(spk_mon, *pois)
        
        trial_time = time() - t0_trial
        if verbose:
            print(f"\r    Trial {trial_idx + 1}/{n_trials} ({trial_time:.1f}s)", end='', flush=True)
    
    if verbose:
        total_time = time() - t0_total
        print(f"\n    ⏱️  Total: {total_time:.1f}s ({total_time/n_trials:.1f}s/trial)")
    
    # Merge all trial data
    if all_spike_data:
        df = pd.concat(all_spike_data, ignore_index=True)
        df['flywire_id'] = df['neuron_idx'].map(i2flyid)
        df = df[['t', 'trial', 'flywire_id']]
    else:
        df = pd.DataFrame(columns=['t', 'trial', 'flywire_id'])
    
    n_spikes = len(df)
    n_active = df['flywire_id'].nunique() if n_spikes > 0 else 0
    
    if verbose:
        print(f"    📊 Spikes: {n_spikes:,}, Active neurons: {n_active:,}")
    
    return {
        'df': df,
        'n_spikes': n_spikes,
        'n_active': n_active,
        'n_trials': n_trials,
        'duration': duration,
        'stim_neurons': neu_exc,
        'silenced_neurons': neu_slnc,
        'params_used': sim_params,
    }


def run_silencing_batch(net_components, neu_exc, neurons_to_silence, target_neurons,
                        freq, n_trials=1, verbose=True):
    """
    Batch silencing experiment for single frequency.
    
    Efficiently runs silencing tests for multiple neurons at once frequency.
    Optimized for experiments testing which neurons are necessary for a response.
    
    Parameters
    ----------
    net_components : dict
        Network components from build_network()
    neu_exc : list
        Neurons to continuously activate (e.g., Sugar GRNs)
    neurons_to_silence : list
        Neurons to individually silence and test
    target_neurons : list
        Target neurons to monitor (e.g., MN9)
    freq : float
        Activation frequency in Hz
    n_trials : int, default=1
        Trials per silencing condition
    verbose : bool
        Print progress
    
    Returns
    -------
    dict : {silenced_neuron_id: {target_id: firing_rate}}
    
    Examples
    --------
    >>> # Test which neurons are required for MN9 activation
    >>> results = run_silencing_batch(NET, SUGAR_GRNs, top_200, [MN9_ID], 
    ...                               freq=100, n_trials=10)
    >>> for neuron_id, rates in results.items():
    ...     if rates[MN9_ID] < baseline * 0.8:
    ...         print(f"Neuron {neuron_id} is required for MN9")
    """
    flyid2i = net_components['flyid2i']
    i2flyid = net_components['i2flyid']
    duration_s = float(net_components['params']['t_run'] / ms) / 1000
    
    results = {}
    n_total = len(neurons_to_silence)
    
    if verbose:
        print(f"\n  Batch silencing: {n_total} neurons @ {freq} Hz")
        t0 = time()
    
    # Convert target neuron indices
    target_indices = {flyid2i[t]: t for t in target_neurons if t in flyid2i}
    
    for idx, slnc_id in enumerate(neurons_to_silence):
        # Run silencing simulation
        result = run_simulation(
            net_components=net_components,
            neu_exc=neu_exc,
            neu_slnc=[slnc_id],
            params={'r_poi': freq * Hz},
            n_trials=n_trials,
            verbose=False
        )
        
        df = result['df']
        
        # Calculate target neuron firing rates
        target_rates = {}
        for tid in target_neurons:
            if len(df) > 0:
                count = len(df[df['flywire_id'] == tid])
                target_rates[tid] = count / (n_trials * duration_s)
            else:
                target_rates[tid] = 0.0
        
        results[slnc_id] = target_rates
        
        # Progress display
        if verbose and (idx + 1) % 10 == 0:
            elapsed = time() - t0
            speed = (idx + 1) / elapsed
            eta = (n_total - idx - 1) / speed
            print(f"\r    Progress: {idx+1}/{n_total} ({elapsed:.0f}s, ETA: {eta:.0f}s)", end='', flush=True)
    
    if verbose:
        print(f"\n    ✅ Complete ({time()-t0:.1f}s)")
    
    return results


# =============================================================================
# Testing
# =============================================================================

if __name__ == '__main__':
    """
    Test simulation functions with dummy network.
    """
    from flylif.core.parameters import DEFAULT_PARAMS
    from flylif.core.data_loader import load_simulation_data
    from flylif.core.network import build_network
    from pathlib import Path
    
    print("\n" + "=" * 70)
    print("Testing simulation.py")
    print("=" * 70)
    
    # Load data
    config = {
        'data_dir': Path('./data_783'),
        'connections_file': 'proofread_connections_783.feather',
        'root_ids_file': 'proofread_root_ids_783.npy',
    }
    
    print("\n[1/3] Loading data...")
    data = load_simulation_data(config, verbose=False)
    
    # Build network
    print("\n[2/3] Building network...")
    columns = data['columns']
    net = build_network(
        data=data,
        pre_col=columns['pre_col'],
        post_col=columns['post_col'],
        weight_col=columns['weight_col'],
        nt_prob_cols=columns.get('nt_prob_cols', {}),
        params=DEFAULT_PARAMS,
        verbose=False
    )
    
    # Test simulation with random neurons
    print("\n[3/3] Testing simulation...")
    test_neurons = list(data['flyid2i'].keys())[:5]  # First 5 neurons
    
    result = run_simulation(
        net_components=net,
        neu_exc=test_neurons,
        n_trials=1,
        verbose=True
    )
    
    print("\n" + "=" * 70)
    print("✅ Simulation test passed!")
    print("=" * 70)
    print(f"Result:")
    print(f"  Spikes: {result['n_spikes']:,}")
    print(f"  Active neurons: {result['n_active']:,}")
    print(f"  DataFrame shape: {result['df'].shape}")