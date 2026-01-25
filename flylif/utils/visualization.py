"""
Visualization Tools for Experiment Results

Provides flexible plotting functions for LIF simulation results.
Supports both automatic data-driven defaults and full manual control.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from scipy.stats import pearsonr


def plot_correlation(rate_orig, rate_ours, n_trials_ours=10, freq=100,
                     figsize=(6, 6), save_path=None, **kwargs):
    """
    Plot firing rate correlation scatter plot.
    
    Parameters
    ----------
    rate_orig : pd.Series
        Original firing rates (index=neuron_ids)
    rate_ours : pd.Series  
        Our firing rates
    n_trials_ours : int
        Number of trials in our simulation (for title)
    freq : int
        Frequency being compared (for title)
    figsize : tuple, optional
    save_path : str or Path, optional
    **kwargs : additional matplotlib arguments (color, alpha, s, etc.)
    
    Returns
    -------
    fig, ax, r : figure, axis, correlation coefficient
    """
    # Merge data
    df_compare = pd.concat([rate_orig, rate_ours], axis=1).fillna(0)
    
    # Calculate correlation
    r, p = pearsonr(df_compare.iloc[:, 0], df_compare.iloc[:, 1])
    
    # Plot
    fig, ax = plt.subplots(figsize=figsize)
    
    # Default kwargs
    plot_kwargs = {'alpha': 0.3, 's': 20, 'color': 'steelblue'}
    plot_kwargs.update(kwargs)
    
    ax.scatter(df_compare.iloc[:, 0], df_compare.iloc[:, 1], **plot_kwargs)
    
    # Identity line
    max_val = df_compare.max().max()
    ax.plot([0, max_val], [0, max_val], 'r--', linewidth=2, alpha=0.7, label='y = x')
    
    ax.set_xlabel(f'Original Firing Rate (Hz)', fontsize=12)
    ax.set_ylabel(f'Our Firing Rate (Hz)', fontsize=12)
    ax.set_title(f'{freq}Hz Correlation (r={r:.3f}, n={n_trials_ours} trials)', 
                 fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # Add stats text
    n_active_orig = (df_compare.iloc[:, 0] > 0).sum()
    n_active_ours = (df_compare.iloc[:, 1] > 0).sum()
    textstr = f'Original: {n_active_orig} neurons\nOurs: {n_active_ours} neurons'
    ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig, ax, r


def plot_response_heatmap(all_firing_rates, freq_list, top_n=None, 
                          cmap='viridis', figsize=None, preset='auto',
                          vmin=None, vmax=None, save_path=None):
    """
    Plot response heatmap (neurons × frequencies).
    
    Parameters
    ----------
    all_firing_rates : dict
        {freq: {neuron_id: rate}}
    freq_list : list
        Frequencies to plot
    top_n : int, optional
        Number of top neurons to show. If None, auto-detect.
    cmap : str, default='viridis'
        Colormap name
    figsize : tuple, optional
        Figure size. If None, auto-calculate based on data.
    preset : str, default='auto'
        - 'auto': Data-driven defaults
        - 'paper': Match original paper style
    vmin, vmax : float, optional
        Color scale limits. If None, use data range.
    save_path : str or Path, optional
    
    Returns
    -------
    fig, ax, top_neurons : figure, axis, list of top neuron IDs
    """
    # Auto-detect top_n
    if top_n is None:
        max_freq = max(freq_list)
        if max_freq in all_firing_rates:
            n_active = len([v for v in all_firing_rates[max_freq].values() if v > 0])
            top_n = min(n_active, 200)  # Cap at 200
        else:
            top_n = 200
    
    # Get top responsive neurons
    max_freq = max(freq_list)
    if max_freq in all_firing_rates:
        rates_at_max = pd.Series(all_firing_rates[max_freq])
        top_neurons = rates_at_max.nlargest(top_n).index.tolist()
    else:
        raise ValueError(f"No data for frequency {max_freq}")
    
    # Build matrix
    matrix = np.zeros((len(top_neurons), len(freq_list)))
    for j, freq in enumerate(freq_list):
        if freq in all_firing_rates:
            for i, neu_id in enumerate(top_neurons):
                matrix[i, j] = all_firing_rates[freq].get(neu_id, 0)
    
    # Auto figsize
    if figsize is None:
        width = max(10, len(freq_list) * 0.5)
        height = max(10, top_n / 25)
        figsize = (width, height)
    
    # Auto vmax (95th percentile to exclude outliers)
    if vmax is None:
        vmax = matrix.max()
        # vmax = np.percentile(matrix[matrix > 0], 95) if (matrix > 0).any() else matrix.max()
    if vmin is None:
        vmin = 0
    
    # Preset adjustments
    if preset == 'paper':
        cmap = 'viridis'
        top_n = 200
    
    # Plot
    fig, ax = plt.subplots(figsize=figsize)
    
    im = ax.imshow(matrix, aspect='auto', cmap=cmap, 
                   vmin=vmin, vmax=vmax, interpolation='nearest')
    
    ax.set_xlabel('Sugar GRN firing rate (Hz)', fontsize=12)
    ax.set_ylabel('Responsive Neurons (sorted by max response)', fontsize=12)
    ax.set_title(f'Neural Response Heatmap (Top {len(top_neurons)} neurons)', 
                 fontsize=13, fontweight='bold')
    
    # X ticks
    ax.set_xticks(range(len(freq_list)))
    ax.set_xticklabels(freq_list, rotation=45 if len(freq_list) > 10 else 0)
    
    # Y ticks
    n_yticks = min(5, len(top_neurons))
    y_positions = np.linspace(0, len(top_neurons)-1, n_yticks).astype(int)
    ax.set_yticks(y_positions)
    ax.set_yticklabels([f'{i+1}' for i in y_positions])
    
    # Colorbar
    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('Firing Rate (Hz)', fontsize=11)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig, ax, top_neurons


def plot_frequency_response_curve(all_firing_rates, freq_list, target_neurons,
                                   labels=None, figsize=(10, 6), 
                                   show_std=True, save_path=None, **kwargs):
    """
    Plot response curves for specific neurons across frequencies.
    
    Parameters
    ----------
    all_firing_rates : dict
        {freq: {neuron_id: rate}}
    freq_list : list
        Frequencies
    target_neurons : list
        Neuron IDs to plot (e.g., [MN9_left, MN9_right])
    labels : list, optional
        Labels for each neuron. If None, use IDs.
    figsize : tuple
    show_std : bool
        Show error bars (requires trial data)
    save_path : str or Path, optional
    **kwargs : line plot arguments
    
    Returns
    -------
    fig, ax
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    if labels is None:
        labels = [str(nid) for nid in target_neurons]
    
    for neuron_id, label in zip(target_neurons, labels):
        rates = []
        for freq in freq_list:
            rate = all_firing_rates.get(freq, {}).get(neuron_id, 0)
            rates.append(rate)
        
        # Default plot style
        plot_kwargs = {'marker': 'o', 'markersize': 8, 'linewidth': 2}
        plot_kwargs.update(kwargs)
        
        ax.plot(freq_list, rates, label=label, **plot_kwargs)
    
    ax.set_xlabel('Sugar GRN firing rate (Hz)', fontsize=12)
    ax.set_ylabel('Predicted MN9 firing rate (Hz)', fontsize=12)
    ax.set_title('Frequency Response Curves', fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig, ax


def plot_summary_statistics(df_summary, save_path=None):
    """
    Plot summary bar charts.
    
    Parameters
    ----------
    df_summary : pd.DataFrame
        Must have columns: Frequency, Active Neurons, Total Spikes
    save_path : optional
    
    Returns
    -------
    fig, axes
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Active neurons
    ax1 = axes[0]
    ax1.bar(df_summary['Frequency'], df_summary['Active Neurons'], 
            color='steelblue', edgecolor='black', alpha=0.8)
    ax1.set_xlabel('Frequency (Hz)', fontsize=11)
    ax1.set_ylabel('Active Neurons', fontsize=11)
    ax1.set_title('Responsive Neurons vs Frequency', fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Total spikes
    ax2 = axes[1]
    ax2.bar(df_summary['Frequency'], df_summary['Total Spikes'], 
            color='coral', edgecolor='black', alpha=0.8)
    ax2.set_xlabel('Frequency (Hz)', fontsize=11)
    ax2.set_ylabel('Total Spikes', fontsize=11)
    ax2.set_title('Network Activity vs Frequency', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig, axes


# =============================================================================
# Preset configurations
# =============================================================================

PLOT_PRESETS = {
    'paper': {
        'heatmap': {'cmap': 'viridis', 'top_n': 200},
        'correlation': {'alpha': 0.3, 's': 20},
    },
    'presentation': {
        'heatmap': {'cmap': 'plasma', 'figsize': (12, 10)},
        'correlation': {'alpha': 0.5, 's': 30, 'color': 'darkblue'},
    },
}


def apply_preset(func_name, preset='auto'):
    """Get preset kwargs for a function."""
    if preset == 'auto':
        return {}
    return PLOT_PRESETS.get(preset, {}).get(func_name, {})