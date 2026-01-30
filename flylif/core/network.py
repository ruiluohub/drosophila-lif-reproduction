"""
Brian2 Network Construction Module

Builds LIF neural network from FlyWire connectivity data.
Supports both preprocessed (with neuron_sign) and raw data formats.
"""

import numpy as np
import pandas as pd
from brian2 import NeuronGroup, Synapses, Network, start_scope
from brian2 import mV, ms
from time import time


def determine_neuron_types_fast(df_conn, pre_col, nt_prob_cols, flyid2i, verbose=True):
    """
    Determine neuron types (excitatory/inhibitory).
    
    Fast path: If 'neuron_sign' column exists, use it directly.
    Fallback: Calculate from GABA + Glutamate probabilities.
    
    Parameters
    ----------
    df_conn : pd.DataFrame
        Connectivity data
    pre_col : str
        Column name for presynaptic neuron IDs
    nt_prob_cols : dict
        Neurotransmitter probability column names (gaba, glut, etc.)
    flyid2i : dict
        FlyWire ID to index mapping
    verbose : bool
        Print progress
    
    Returns
    -------
    dict : {neuron_index: sign}, where sign is +1 (exc) or -1 (inh)
    """
    
    # === Fast path: Use preprocessed neuron_sign ===
    if 'neuron_sign' in df_conn.columns:
        if verbose:
            print(f"   Using preprocessed neuron_sign column")
            t0 = time()
        
        # Build mapping from presynaptic neuron to sign
        # Use mode (most common sign) for each neuron
        pre_ids = df_conn[pre_col].values
        signs = df_conn['neuron_sign'].values
        
        # Group by presynaptic neuron, take majority vote
        df_temp = pd.DataFrame({'pre_id': pre_ids, 'sign': signs})
        neuron_sign_mode = df_temp.groupby('pre_id')['sign'].agg(
            lambda x: 1 if (x == 1).sum() > (x == -1).sum() else -1
        )
        
        neuron_signs = {}
        n_exc, n_inh = 0, 0
        
        for pre_id, sign in neuron_sign_mode.items():
            pre_id_int = int(pre_id)
            if pre_id_int in flyid2i:
                neuron_idx = flyid2i[pre_id_int]
                neuron_signs[neuron_idx] = int(sign)
                if sign > 0:
                    n_exc += 1
                else:
                    n_inh += 1
        
        if verbose:
            print(f"   Excitatory: {n_exc:,}, Inhibitory: {n_inh:,} ({time()-t0:.1f}s)")
        
        return neuron_signs
    
    # === Fallback: Calculate from NT probabilities ===
    if verbose:
        print(f"   Calculating from NT probabilities...")
        t0 = time()
    
    gaba_col = nt_prob_cols.get('gaba')
    glut_col = nt_prob_cols.get('glut')
    
    if gaba_col is None and glut_col is None:
        if verbose:
            print(f"   ⚠️  No GABA/Glut columns, defaulting to excitatory")
        return {}
    
    pre_ids = df_conn[pre_col].values
    
    # Calculate inhibitory probability
    inhib_prob = np.zeros(len(df_conn), dtype=np.float32)
    if gaba_col and gaba_col in df_conn.columns:
        inhib_prob += df_conn[gaba_col].fillna(0).values.astype(np.float32)
    if glut_col and glut_col in df_conn.columns:
        inhib_prob += df_conn[glut_col].fillna(0).values.astype(np.float32)
    
    is_inhib_synapse = inhib_prob > 0.5
    
    # Aggregate by presynaptic neuron
    df_temp = pd.DataFrame({'pre_id': pre_ids, 'is_inhib': is_inhib_synapse})
    neuron_inhib_ratio = df_temp.groupby('pre_id')['is_inhib'].mean()
    
    neuron_signs = {}
    n_exc, n_inh = 0, 0
    
    for pre_id, ratio in neuron_inhib_ratio.items():
        pre_id_int = int(pre_id)
        if pre_id_int in flyid2i:
            neuron_idx = flyid2i[pre_id_int]
            if ratio > 0.5:
                neuron_signs[neuron_idx] = -1
                n_inh += 1
            else:
                neuron_signs[neuron_idx] = 1
                n_exc += 1
    
    if verbose:
        print(f"   Excitatory: {n_exc:,}, Inhibitory: {n_inh:,} ({time()-t0:.1f}s)")
    
    return neuron_signs


def build_network(data, pre_col, post_col, weight_col, nt_prob_cols, params, 
                  syn_threshold=5, verbose=True):
    """
    Build Brian2 network from connectivity data.
    
    Optimized version supporting preprocessed data:
    - If data has 'neuron_sign' column: use it directly (fast)
    - Otherwise: calculate from NT probabilities (fallback)
    
    Parameters
    ----------
    data : dict
        Data package from load_all_data(), must contain:
        - n_neurons: int
        - flyid2i, i2flyid: ID mappings
        - df_conn: connectivity DataFrame
    pre_col : str
        Presynaptic neuron ID column name
    post_col : str
        Postsynaptic neuron ID column name
    weight_col : str
        Synapse count column name
    nt_prob_cols : dict
        NT probability columns (only used if neuron_sign absent)
    params : dict
        Brian2 simulation parameters
    syn_threshold : int, default=5
        Minimum synapse count (only if data not pre-filtered)
    verbose : bool
        Print progress
    
    Returns
    -------
    dict with network components:
        - net: Brian2 Network object
        - neu: NeuronGroup
        - syn: Synapses
        - n_neurons, n_synapses
        - flyid2i, i2flyid: ID mappings
        - params: parameters used
        - (for silencing) i_pre, i_post, pre_syn_indices, weights_array
    """
    start_scope()
    
    if verbose:
        print("=" * 60)
        print("Building Brian2 Network")
        print("=" * 60)
    
    n_neurons = data['n_neurons']
    flyid2i = data['flyid2i']
    i2flyid = data['i2flyid']
    df_conn = data['df_conn']
    
    # === Step 1: Filter connections (if not pre-filtered) ===
    if verbose:
        print(f"\n[1/7] Checking connections...")
    
    n_before = len(df_conn)
    
    # Check if already filtered (min syn_count >= threshold)
    if weight_col in df_conn.columns:
        min_syn = df_conn[weight_col].min()
        if min_syn < syn_threshold:
            if verbose:
                print(f"   Filtering (syn_count >= {syn_threshold})...")
            df_conn_filtered = df_conn[df_conn[weight_col] >= syn_threshold].copy()
        else:
            if verbose:
                print(f"   Already filtered (min={min_syn} >= {syn_threshold})")
            df_conn_filtered = df_conn.copy()
    else:
        df_conn_filtered = df_conn.copy()
    
    n_after = len(df_conn_filtered)
    if verbose:
        print(f"   Connections: {n_before:,} -> {n_after:,} ({100*n_after/n_before:.1f}%)")
    
    # === Step 2: Determine neuron types (skip if preprocessed) ===
    if verbose:
        print(f"\n[2/7] Determining neuron types...")
    
    # Fast path: Skip if neuron_sign column exists
    if 'neuron_sign' in df_conn_filtered.columns:
        if verbose:
            print(f"   ✓ Using preprocessed neuron_sign (skipping computation)")
        neuron_signs = {}  # Empty dict, will use column directly in Step 4
    else:
        # Fallback: Calculate from NT probabilities
        neuron_signs = determine_neuron_types_fast(
            df_conn_filtered, pre_col, nt_prob_cols, flyid2i, verbose
        )
    
    # === Step 3: Process connections ===
    if verbose:
        print(f"\n[3/7] Processing connections...")
        t0 = time()
    
    pre_ids = df_conn_filtered[pre_col].values.astype(np.int64)
    post_ids = df_conn_filtered[post_col].values.astype(np.int64)
    
    # Validate connections
    valid_ids_set = set(flyid2i.keys())
    
    pre_valid = np.array([int(p) in valid_ids_set for p in pre_ids])
    post_valid = np.array([int(p) in valid_ids_set for p in post_ids])
    valid_mask = pre_valid & post_valid
    
    n_valid = np.sum(valid_mask)
    if verbose:
        print(f"   Valid connections: {n_valid:,} / {len(pre_ids):,}")
    
    valid_pre_ids = pre_ids[valid_mask]
    valid_post_ids = post_ids[valid_mask]
    
    # Map to indices
    i_pre = np.array([flyid2i[int(p)] for p in valid_pre_ids], dtype=np.int32)
    i_post = np.array([flyid2i[int(p)] for p in valid_post_ids], dtype=np.int32)
    
    if verbose:
        print(f"   ⏱️  {time()-t0:.1f}s")
    
    # === Step 4: Calculate weights ===
    if verbose:
        print(f"\n[4/7] Calculating synaptic weights...")
        t0 = time()
    
    if weight_col and weight_col in df_conn_filtered.columns:
        weights_raw = df_conn_filtered[weight_col].values[valid_mask].astype(np.float64)
    else:
        weights_raw = np.ones(n_valid, dtype=np.float64)
    
    # Get signs (vectorized)
    # Check if using preprocessed signs
    if 'neuron_sign' in df_conn_filtered.columns:
        # Direct from dataframe
        signs = df_conn_filtered['neuron_sign'].values[valid_mask].astype(np.float64)
    else:
        # From neuron_signs dict
        default_sign = 1
        signs = np.array([neuron_signs.get(idx, default_sign) for idx in i_pre], dtype=np.float64)
    
    weights = weights_raw * signs
    
    n_exc_syn = np.sum(signs > 0)
    n_inh_syn = np.sum(signs < 0)
    if verbose:
        print(f"   Excitatory: {n_exc_syn:,}, Inhibitory: {n_inh_syn:,}")
        print(f"   ⏱️  {time()-t0:.1f}s")
    
    # === Step 5: Precompute synapse indices ===
    if verbose:
        print(f"\n[5/7] Precomputing synapse indices...")
        t0 = time()
    
    syn_indices = np.arange(n_valid, dtype=np.int64)
    df_syn = pd.DataFrame({'pre': i_pre, 'syn_idx': syn_indices})
    grouped = df_syn.groupby('pre')['syn_idx']
    pre_syn_indices = {int(k): v.values for k, v in grouped}
    
    if verbose:
        n_pre_neurons = len(pre_syn_indices)
        avg_syn = n_valid / n_pre_neurons if n_pre_neurons > 0 else 0
        print(f"   Neurons with output: {n_pre_neurons:,}")
        print(f"   Avg synapses/neuron: {avg_syn:.1f}")
        print(f"   ⏱️  {time()-t0:.1f}s")
    
    # === Step 6: Create neuron group ===
    if verbose:
        print(f"\n[6/7] Creating neurons (n={n_neurons:,})...")
        t0 = time()
    
    neu = NeuronGroup(
        N=n_neurons,
        model=params['eqs'],
        method='linear',
        threshold=params['eq_th'],
        reset=params['eq_rst'],
        refractory='rfc',
        name='neurons',
        namespace=params,
    )
    neu.v = params['v_0']
    neu.g = 0
    neu.rfc = params['t_rfc']
    neu.silenced = 0  # ← NEW: initialize silencing gate (all neurons active by default)
    
    if verbose:
        print(f"   ⏱️  {time()-t0:.1f}s")
    
    # === Step 7: Create synapses ===
    if verbose:
        print(f"\n[7/7] Creating synapses (n={n_valid:,})...")
        t0 = time()
    
    syn = Synapses(neu, neu, 'w : volt', on_pre='g += w',
                   delay=params['t_dly'], name='synapses')
    syn.connect(i=i_pre, j=i_post)
    
    weights_final = weights * float(params['w_syn'] / mV) * mV
    syn.w = weights_final
    
    if verbose:
        print(f"   ⏱️  {time()-t0:.1f}s")
    
    # === Create Network and save initial state ===
    if verbose:
        print(f"\n[Network] Creating Network...")
        t0 = time()
    
    net = Network(neu, syn)
    net.store('initial')
    
    if verbose:
        print(f"   ⏱️  {time()-t0:.1f}s")
    
    weights_array = np.array(syn.w / mV, dtype=np.float64)
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"✅ Network built!")
        print(f"   Neurons: {n_neurons:,}, Synapses: {n_valid:,}")
        print(f"{'='*60}")
    
    return {
        'net': net,
        'neu': neu,
        'syn': syn,
        'n_neurons': n_neurons,
        'n_synapses': n_valid,
        'flyid2i': flyid2i,
        'i2flyid': i2flyid,
        'params': params.copy(),
        # For silencing experiments
        'i_pre': i_pre,
        'i_post': i_post,
        'pre_syn_indices': pre_syn_indices,
        'weights_original': weights,
        'weights_array': weights_array,
        'neuron_signs': neuron_signs,
        'syn_threshold': syn_threshold,
    }


# =============================================================================
# Testing
# =============================================================================

if __name__ == '__main__':
    """
    Test network building with dummy data.
    """
    from flylif.core.parameters import DEFAULT_PARAMS
    from flylif.core.data_loader import load_simulation_data
    from pathlib import Path
    
    print("\n" + "=" * 70)
    print("Testing network.py")
    print("=" * 70)
    
    # Load test data
    config = {
        'data_dir': Path('./data_783'),
        'connections_file': 'proofread_connections_783.feather',
        'root_ids_file': 'proofread_root_ids_783.npy',
    }
    
    print("\n[1/2] Loading data...")
    data = load_simulation_data(config, verbose=False)
    print(f"   Loaded: {data['n_neurons']:,} neurons, {len(data['df_conn']):,} connections")
    
    # Build network
    print("\n[2/2] Building network...")
    
    columns = data['columns']
    net_components = build_network(
        data=data,
        pre_col=columns['pre_col'],
        post_col=columns['post_col'],
        weight_col=columns['weight_col'],
        nt_prob_cols=columns.get('nt_prob_cols', {}),
        params=DEFAULT_PARAMS,
        syn_threshold=5,
        verbose=True
    )
    
    print("\n" + "=" * 70)
    print("✅ Network building test passed!")
    print("=" * 70)
    print(f"Network components:")
    print(f"  Neurons: {net_components['n_neurons']:,}")
    print(f"  Synapses: {net_components['n_synapses']:,}")
    print(f"  Has restore point: {'initial' in net_components['net']._stored_state}")