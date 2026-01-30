"""
FlyLIF Model Parameters
=======================

Default parameters for the Drosophila brain LIF (Leaky Integrate-and-Fire) model.
Based on Shiu et al. (2024) "A Drosophila computational brain model reveals 
sensorimotor processing" Nature.

Parameters are organized by source and category:
- Simulation parameters (trials, duration)
- Neuronal dynamics (Kakaria & de Bivort 2017)
- Synaptic parameters (Jürgensen et al., Paul et al. 2015)
- Input stimulation (Poisson rates)
- Model equations (LIF with alpha synapse)

All physical quantities use Brian2 units (mV, ms, Hz).
"""

from brian2 import mV, ms, Hz
from textwrap import dedent

# =============================================================================
# Default Model Parameters
# =============================================================================

DEFAULT_PARAMS = {
    # -------------------------------------------------------------------------
    # Simulation Parameters
    # -------------------------------------------------------------------------
    't_run': 1000 * ms,              # Duration of a single trial
    'n_run': 30,                     # Number of trials to run
    
    # -------------------------------------------------------------------------
    # Neuronal Dynamics (Kakaria and de Bivort 2017)
    # -------------------------------------------------------------------------
    'v_0': -52 * mV,                 # Resting membrane potential
    'v_rst': -52 * mV,               # Reset potential after spike
    'v_th': -45 * mV,                # Threshold potential for spiking
    't_mbr': 20 * ms,                # Membrane time constant
    
    # -------------------------------------------------------------------------
    # Synaptic Dynamics
    # -------------------------------------------------------------------------
    'tau': 5 * ms,                   # Synaptic conductance decay time constant (Jürgensen et al.)
    't_rfc': 2.2 * ms,               # Absolute refractory period (Lazar et al.)
    't_dly': 1.8 * ms,               # Synaptic transmission delay (Paul et al. 2015)
    'w_syn': 0.275 * mV,             # Synaptic weight per synapse (free parameter)
    
    # -------------------------------------------------------------------------
    # Input Stimulation
    # -------------------------------------------------------------------------
    'r_poi': 150 * Hz,               # Primary Poisson input rate for activation
    'r_poi2': 0 * Hz,                # Secondary Poisson input rate (for dual-modality)
    'f_poi': 250,                    # Poisson input scaling factor (dimensionless)
    
    # -------------------------------------------------------------------------
    # Model Equations
    # -------------------------------------------------------------------------
    'eqs': dedent('''
        dv/dt = (v_0 - v + g) / t_mbr : volt (unless refractory)
        dg/dt = -g / tau               : volt (unless refractory) 
        rfc                            : second
        silenced                       : integer
    '''),
    
    'eq_th': 'v > v_th and silenced == 0',  # Threshold condition for spike generation MODIFIED: add active condition           
    'eq_rst': 'v = v_rst; w = 0; g = 0 * mV',  # Reset equations after spike
}


# Parameter documentation (separate from dictionary)
PARAM_DOCS = {
    # Simulation
    't_run': 'Duration of a single trial (default: 1000 ms)',
    'n_run': 'Number of trials to run (default: 30)',
    
    # Neuronal dynamics
    'v_0': 'Resting membrane potential (default: -52 mV)',
    'v_rst': 'Reset potential after spike (default: -52 mV)',
    'v_th': 'Threshold potential for spiking (default: -45 mV)',
    't_mbr': 'Membrane time constant (default: 20 ms)',
    
    # Synaptic
    'tau': 'Synaptic conductance decay time constant (default: 5 ms)',
    't_rfc': 'Absolute refractory period (default: 2.2 ms)',
    't_dly': 'Synaptic transmission delay (default: 1.8 ms)',
    'w_syn': 'Synaptic weight per synapse - free parameter (default: 0.275 mV)',
    
    # Stimulation
    'r_poi': 'Primary Poisson input rate for activation (default: 150 Hz)',
    'r_poi2': 'Secondary Poisson input rate for dual-modality (default: 0 Hz)',
    'f_poi': 'Poisson input scaling factor (dimensionless, default: 250)',
    
    # Equations
    'eqs': '''LIF neuron equations with alpha synapse model.
    Variables:
        v : membrane potential (volt)
        g : synaptic conductance (volt) 
        rfc : refractory period (second)
        active : silencing gate (boolean)  
    The model uses voltage-based synaptic input where incoming spikes
    increment g, which then exponentially decays with time constant tau.
    Neurons can be silenced by setting active=False.''',  # ← MODIFIED
    'eq_th': 'Threshold condition for spike generation',
    'eq_rst': 'Reset equations executed after spike',
}


# =============================================================================
# Parameter Validation
# =============================================================================

def validate_params(params):
    """
    Validate model parameters for consistency.
    
    Parameters
    ----------
    params : dict
        Parameter dictionary to validate
        
    Returns
    -------
    bool
        True if parameters are valid
        
    Raises
    ------
    ValueError
        If parameters are inconsistent or out of valid range
    """
    # Check voltage ordering
    if params['v_th'] <= params['v_rst']:
        raise ValueError(f"Threshold ({params['v_th']}) must be greater than reset ({params['v_rst']})")
    
    if params['v_rst'] < params['v_0']:
        raise ValueError(f"Reset potential ({params['v_rst']}) should not be below resting ({params['v_0']})")
    
    # Check time constants are positive
    for key in ['t_run', 't_mbr', 'tau', 't_rfc', 't_dly']:
        if params[key] <= 0 * ms:
            raise ValueError(f"Time constant '{key}' must be positive, got {params[key]}")
    
    # Check trial count
    if params['n_run'] < 1:
        raise ValueError(f"Number of runs must be >= 1, got {params['n_run']}")
    
    # Check Poisson rates are non-negative  
    if params['r_poi'] < 0 * Hz or params['r_poi2'] < 0 * Hz:
        raise ValueError("Poisson rates must be non-negative")
    
    return True


# =============================================================================
# Parameter Presets
# =============================================================================

PRESET_FAST_TEST = DEFAULT_PARAMS.copy()
PRESET_FAST_TEST.update({
    't_run': 100 * ms,
    'n_run': 2,
})

PRESET_FULL_EXPERIMENT = DEFAULT_PARAMS.copy()
PRESET_FULL_EXPERIMENT.update({
    't_run': 1000 * ms,
    'n_run': 30,
})


# =============================================================================
# Convenience Functions
# =============================================================================

def get_params(preset='default'):
    """
    Get parameter dictionary by preset name.
    
    Parameters
    ----------
    preset : str
        Preset name: 'default', 'fast_test', or 'full_experiment'
        
    Returns
    -------
    dict
        Copy of parameter dictionary
        
    Examples
    --------
    >>> params = get_params('fast_test')
    >>> params['t_run']
    100. * msecond
    """
    presets = {
        'default': DEFAULT_PARAMS,
        'fast_test': PRESET_FAST_TEST,
        'full_experiment': PRESET_FULL_EXPERIMENT,
    }
    
    if preset not in presets:
        raise ValueError(f"Unknown preset '{preset}'. Available: {list(presets.keys())}")
    
    params = presets[preset].copy()
    validate_params(params)
    return params


def print_params(params=None):
    """
    Pretty-print parameter values.
    
    Parameters
    ----------
    params : dict, optional
        Parameters to print. If None, uses DEFAULT_PARAMS
    """
    if params is None:
        params = DEFAULT_PARAMS
    
    print("=" * 60)
    print("FlyLIF Model Parameters")
    print("=" * 60)
    
    categories = {
        'Simulation': ['t_run', 'n_run'],
        'Neuronal Dynamics': ['v_0', 'v_rst', 'v_th', 't_mbr'],
        'Synaptic': ['tau', 't_rfc', 't_dly', 'w_syn'],
        'Stimulation': ['r_poi', 'r_poi2', 'f_poi'],
        'Equations': ['eqs', 'eq_th', 'eq_rst'],
    }
    
    for category, keys in categories.items():
        print(f"\n{category}:")
        for key in keys:
            if key in params:
                value = params[key]
                if isinstance(value, str) and '\n' in value:
                    print(f"  {key}: <multiline>")
                else:
                    print(f"  {key}: {value}")


def get_param_doc(key):
    """
    Get documentation for a specific parameter.
    
    Parameters
    ----------
    key : str
        Parameter name
        
    Returns
    -------
    str
        Parameter documentation
    """
    return PARAM_DOCS.get(key, 'No documentation available')