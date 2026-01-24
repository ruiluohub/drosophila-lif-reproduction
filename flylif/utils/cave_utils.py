"""
FlyWire ID Version Conversion Utilities

Tools for converting neuron IDs between different FlyWire connectome versions.
Uses CAVEclient to map IDs through supervoxels.
"""

from concurrent.futures import ThreadPoolExecutor
import numpy as np
import pickle
from pathlib import Path


def cave_id(old_list, new_ver=783, n_workers=10, client=None, verbose=True):
    """
    Convert FlyWire neuron IDs from one version to another.
    
    Uses supervoxel mapping through CAVE materialization timestamps.
    Parallelized for faster batch conversion.
    
    Parameters
    ----------
    old_list : list
        List of neuron root IDs from old version
    new_ver : int, default=783
        Target FlyWire version number
    n_workers : int, default=10
        Number of parallel workers for conversion
    client : CAVEclient, optional
        Initialized CAVEclient. If None, creates new client.
    verbose : bool, default=True
        Print progress
    
    Returns
    -------
    list : Converted neuron IDs in new version
    
    Examples
    --------
    >>> # Convert v630 IDs to v783
    >>> old_ids = [720575940624963786, ...]
    >>> new_ids = cave_id(old_ids, new_ver=783)
    
    >>> # With custom client
    >>> from caveclient import CAVEclient
    >>> client = CAVEclient('flywire_fafb_public')
    >>> new_ids = cave_id(old_ids, client=client)
    """
    # Initialize client if not provided
    if client is None:
        try:
            from caveclient import CAVEclient
            client = CAVEclient('flywire_fafb_public')
        except ImportError:
            raise ImportError("CAVEclient required. Install: pip install caveclient")
    
    if verbose:
        print(f"Converting {len(old_list)} IDs to version {new_ver}...")
    
    # Get timestamp for target version
    timestamp = client.materialize.get_timestamp(new_ver)
    
    def convert_single(old_root):
        """Convert single ID through supervoxel."""
        try:
            # Get supervoxel
            svs = client.chunkedgraph.get_leaves(old_root, stop_layer=1)
            sv = svs[0]
            
            # Map to new root ID at target timestamp
            new_root = client.chunkedgraph.get_root_id(
                supervoxel_id=sv, 
                timestamp=timestamp
            )
            return new_root
        except Exception as e:
            if verbose:
                print(f"\n  Warning: Failed to convert {old_root}: {e}")
            return None
    
    # Parallel conversion
    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        new_list = list(executor.map(convert_single, old_list))
    
    # Check for failures
    failed = sum(1 for x in new_list if x is None)
    if failed > 0 and verbose:
        print(f"  ⚠️  {failed}/{len(old_list)} conversions failed")
    
    # Remove None values
    new_list = [x for x in new_list if x is not None]
    
    if verbose:
        print(f"  ✅ Converted: {len(new_list)}/{len(old_list)} IDs")
    
    return new_list


def convert_neuron_list_cached(old_list, cache_file, new_ver=783, force_update=False, verbose=True):
    """
    Convert neuron IDs with caching to avoid repeated API calls.
    
    Parameters
    ----------
    old_list : list
        Old version IDs
    cache_file : str or Path
        Path to cache file (.pkl)
    new_ver : int
        Target version
    force_update : bool, default=False
        If True, ignore cache and reconvert
    verbose : bool
        Print status
    
    Returns
    -------
    list : Converted IDs
    """
    cache_file = Path(cache_file)
    
    # Load from cache if available
    if cache_file.exists() and not force_update:
        if verbose:
            print(f"Loading cached ID conversion from {cache_file}")
        with open(cache_file, 'rb') as f:
            cache = pickle.load(f)
        
        if cache.get('version') == new_ver and cache.get('old_ids') == old_list:
            if verbose:
                print(f"  ✅ Using cached conversion ({len(cache['new_ids'])} IDs)")
            return cache['new_ids']
        else:
            if verbose:
                print(f"  Cache outdated, reconverting...")
    
    # Convert
    new_list = cave_id(old_list, new_ver=new_ver, verbose=verbose)
    
    # Save cache
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_file, 'wb') as f:
        pickle.dump({
            'old_ids': old_list,
            'new_ids': new_list,
            'version': new_ver,
        }, f)
    
    if verbose:
        print(f"  ✅ Cached to {cache_file}")
    
    return new_list


def create_id_mapping(old_ids, new_ids):
    """
    Create bidirectional ID mapping.
    
    Parameters
    ----------
    old_ids : list
        Old version IDs
    new_ids : list
        New version IDs (same order)
    
    Returns
    -------
    dict : {old_id: new_id}
    dict : {new_id: old_id}
    """
    old_to_new = {old: new for old, new in zip(old_ids, new_ids)}
    new_to_old = {new: old for old, new in zip(old_ids, new_ids)}
    
    return old_to_new, new_to_old


# =============================================================================
# Testing
# =============================================================================

if __name__ == '__main__':
    """
    Test ID conversion with small sample.
    """
    from caveclient import CAVEclient
    
    print("\n" + "=" * 70)
    print("Testing cave_utils.py")
    print("=" * 70)
    
    # Initialize client
    print("\nInitializing CAVEclient...")
    client = CAVEclient('flywire_fafb_public')
    print("  ✅ Connected")
    
    # Test with 3 IDs
    test_old_ids = [
        720575940624963786,
        720575940630233916,
        720575940637568838,
    ]
    
    print(f"\nTest: Converting {len(test_old_ids)} IDs (v630 → v783)")
    new_ids = cave_id(test_old_ids, new_ver=783, client=client, n_workers=3)
    
    print(f"\nResults:")
    for old, new in zip(test_old_ids, new_ids):
        print(f"  {old} → {new}")
    
    # Test caching
    print(f"\nTest: Caching mechanism")
    cache_file = Path('./test_cache.pkl')
    new_ids_cached = convert_neuron_list_cached(
        test_old_ids, cache_file, new_ver=783
    )
    
    print(f"\nTest: Load from cache")
    new_ids_cached2 = convert_neuron_list_cached(
        test_old_ids, cache_file, new_ver=783
    )
    
    assert new_ids == new_ids_cached == new_ids_cached2
    print("  ✅ All tests passed")
    
    # Cleanup
    cache_file.unlink()