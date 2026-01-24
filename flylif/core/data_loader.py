"""
FlyWire Connectivity Data Loader (Optimized)

Key optimizations:
1. Filter syn_count >= 5 (84% reduction: 16.8M → 2.7M rows)
2. Remove 6 NT probability columns (768MB reduction)
3. Remove neuropil column (500MB reduction)
4. Precompute neuron_sign (+16MB)

Result: 1.1GB → 0.35GB (68% reduction)
"""

import numpy as np
import pandas as pd
import pyarrow.feather as feather
from pathlib import Path
from time import time


def load_connectivity_data(filepath, chunk_size=500_000, verbose=True):
    """Load connectivity from feather using chunk streaming."""
    table = feather.read_table(filepath)
    num_rows = table.num_rows
    col_names = table.column_names
    
    if verbose:
        print(f"   Total rows: {num_rows:,}")
    
    col_data = {col: [] for col in col_names}
    
    for i in range(0, num_rows, chunk_size):
        end_row = min(i + chunk_size, num_rows)
        chunk = table.slice(i, end_row - i)
        
        for col_name in col_names:
            col = chunk.column(col_name)
            arr = col.combine_chunks()
            
            if str(arr.type) == 'string':
                col_data[col_name].extend(arr.to_pylist())
            else:
                col_data[col_name].append(arr.to_numpy(zero_copy_only=False))
        
        if verbose:
            print(f"\r   Loading... {end_row / num_rows * 100:5.1f}%", end='', flush=True)
    
    if verbose:
        print()
    
    final_data = {}
    for col_name in col_names:
        if isinstance(col_data[col_name][0], (list, str)):
            final_data[col_name] = col_data[col_name]
        else:
            final_data[col_name] = np.concatenate(col_data[col_name])
    
    return pd.DataFrame(final_data)


def compute_neuron_signs(df_conn, verbose=True):
    """
    Compute neuron signs from GABA + Glut columns.
    Auto-finds column names.
    """
    if verbose:
        print(f"   Computing neuron signs...")
        t0 = time()
    
    cols = df_conn.columns.tolist()
    
    # Find GABA/Glut columns (flexible matching)
    gaba_col = next((c for c in cols if 'gaba' in c.lower()), None)
    glut_col = next((c for c in cols if 'glut' in c.lower()), None)
    
    if gaba_col is None and glut_col is None:
        if verbose:
            print(f"   No GABA/Glut columns, defaulting to excitatory")
        df_conn['neuron_sign'] = np.int8(1)
        return df_conn
    
    inhib_prob = np.zeros(len(df_conn), dtype=np.float32)
    
    if gaba_col:
        inhib_prob += df_conn[gaba_col].fillna(0).values.astype(np.float32)
    if glut_col:
        inhib_prob += df_conn[glut_col].fillna(0).values.astype(np.float32)
    
    df_conn['neuron_sign'] = np.where(inhib_prob > 0.5, np.int8(-1), np.int8(1))
    
    if verbose:
        n_exc = (df_conn['neuron_sign'] == 1).sum()
        n_inh = (df_conn['neuron_sign'] == -1).sum()
        print(f"   Excitatory: {n_exc:,}, Inhibitory: {n_inh:,} ({time()-t0:.1f}s)")
    
    return df_conn


def load_simulation_data(config, syn_threshold=5, verbose=True):
    """
    Load and optimize data for simulation.
    
    Optimizations:
    1. Filter syn_count >= syn_threshold (default 5)
    2. Remove neurotransmitter probability columns (6 columns)
    3. Remove neuropil column (string, large)
    4. Precompute neuron_sign column
    
    Result: ~2.7M rows, 4 columns, ~350MB
    
    Parameters
    ----------
    config : dict
        Must have 'data_dir', 'connections_file', 'root_ids_file'
    syn_threshold : int, default=5
        Minimum synapse count to keep
    verbose : bool
        Print progress
    
    Returns
    -------
    dict with optimized data
    """
    if verbose:
        print("=" * 60)
        print("Loading Data (Simulation Mode - Optimized)")
        print("=" * 60)
    
    data_dir = Path(config['data_dir'])
    
    # Load root IDs
    if verbose:
        print(f"\n[1/5] Loading root IDs...")
    root_ids = np.load(data_dir / config['root_ids_file'])
    n_neurons = len(root_ids)
    if verbose:
        print(f"   Neurons: {n_neurons:,}")
    
    # Create mappings
    if verbose:
        print(f"\n[2/5] Creating mappings...")
    flyid2i = {int(rid): i for i, rid in enumerate(root_ids)}
    i2flyid = {i: int(rid) for i, rid in enumerate(root_ids)}
    if verbose:
        print(f"   ✅ Done")
    
    # Load connectivity
    if verbose:
        print(f"\n[3/5] Loading connectivity...")
    df_conn = load_connectivity_data(
        data_dir / config['connections_file'], 
        verbose=verbose
    )
    original_size = df_conn.memory_usage(deep=True).sum() / 1e6
    original_rows = len(df_conn)
    
    # Compute signs
    if verbose:
        print(f"\n[4/5] Computing neuron signs...")
    df_conn = compute_neuron_signs(df_conn, verbose=verbose)
    
    # Optimize
    if verbose:
        print(f"\n[5/5] Optimizing data...")
    
    # Step 1: Find columns
    cols = df_conn.columns.tolist()
    pre_col = next((c for c in cols if 'pre' in c.lower() and 'root' in c.lower()), None)
    post_col = next((c for c in cols if 'post' in c.lower() and 'root' in c.lower()), None)
    weight_col = next((c for c in cols if 'syn' in c.lower() and 'count' in c.lower()), None)
    
    if verbose:
        print(f"   Identified columns: pre={pre_col}, post={post_col}, weight={weight_col}")
    
    # Step 2: Filter connections
    if weight_col and weight_col in df_conn.columns:
        if verbose:
            print(f"   Filtering {weight_col} >= {syn_threshold}...")
        mask = df_conn[weight_col] >= syn_threshold
        if verbose:
            print(f"   {original_rows:,} -> {mask.sum():,} ({100*mask.sum()/original_rows:.1f}%)")
    else:
        if verbose:
            print(f"   ⚠️ No weight column, keeping all rows")
        mask = np.ones(len(df_conn), dtype=bool)
    
    # Step 3: Create new DataFrame with only 4 columns (avoid drop issues)
    if verbose:
        print(f"   Creating optimized DataFrame (4 columns only)...")
    
    df_conn_opt = pd.DataFrame({
        'pre_pt_root_id': df_conn.loc[mask, pre_col].values,
        'post_pt_root_id': df_conn.loc[mask, post_col].values,
        'syn_count': df_conn.loc[mask, weight_col].values if weight_col else np.ones(mask.sum()),
        'neuron_sign': df_conn.loc[mask, 'neuron_sign'].values,
    })
    
    df_conn = df_conn_opt  # Replace with optimized version
    
    # Summary
    final_size = df_conn.memory_usage(deep=True).sum() / 1e6
    reduction = (1 - final_size / original_size) * 100
    
    if verbose:
        print(f"\n[Summary]")
        print(f"   Rows: {len(df_conn):,}")
        print(f"   Columns: {df_conn.columns.tolist()}")
        print(f"   Memory: {final_size:.0f} MB")
        print(f"   Reduction: {reduction:.1f}%")
    
    return {
        'root_ids': root_ids,
        'n_neurons': n_neurons,
        'flyid2i': flyid2i,
        'i2flyid': i2flyid,
        'df_conn': df_conn,
        'columns': {
            'pre_col': pre_col,
            'post_col': post_col,
            'weight_col': weight_col,
        }
    }


def load_all_data(config, optimize_for='simulation', syn_threshold=5, verbose=True):
    """
    Load data with mode selection.
    
    Currently only 'simulation' mode fully implemented.
    """
    if optimize_for == 'simulation':
        return load_simulation_data(config, syn_threshold, verbose)
    else:
        raise NotImplementedError(f"Mode '{optimize_for}' not yet implemented")


# For testing
if __name__ == '__main__':
    CONFIG = {
        'data_dir': Path('./data_783'),
        'connections_file': 'proofread_connections_783.feather',
        'root_ids_file': 'proofread_root_ids_783.npy',
    }
    
    data = load_simulation_data(CONFIG)
    print(f"\n✅ Test: {len(data['df_conn']):,} rows, {len(data['df_conn'].columns)} cols")
    assert len(data['df_conn'].columns) == 4
    assert len(data['df_conn']) < 3_000_000