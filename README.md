# Drosophila Brain LIF Model Reproduction

**Status**: Modularized implementation with performance optimization (v0.1.0-alpha)  
**Completion Date**: Core validation 2026-01-10 | Modularization 2026-01-25  
**Based on**: Shiu et al. (2024) "A Drosophila computational brain model reveals sensorimotor processing" *Nature*

---

## Overview

This repository contains a modularized reproduction of the connectome-constrained Leaky Integrate-and-Fire (LIF) model for the *Drosophila melanogaster* brain.

**Current implementation:**
- ✅ Modular architecture (4 core modules + utilities)
- ✅ Network construction from FlyWire v783 connectome (139K neurons, 2.7M synapses)
- ✅ Data optimization: 95% reduction (1.4GB → 68MB)
- ✅ Parallel execution: 5× speedup validated (6-core), 3× on 4-core (16GB)
- ✅ **Exp1 validation**: r=0.93 (20×10T)
- ✅ **Exp2 implementation**: 200×2×1T in 32 min (3× vs baseline)


**Scope**: This reproduction validates the core methodology and provides an optimized, parallelizable implementation ready for cloud deployment.

---

## ⚡ Key Optimizations

### 1. Data Preprocessing (95% Size Reduction)
- Precompute neuron signs (excitatory/inhibitory) in main process
- Filter weak connections (syn_count < 5)
- Remove redundant neurotransmitter probability columns
- **Result**: 1.4GB → 68MB, enabling faster parallel worker transmission

### 2. Network Construction Adaptation
- Skip redundant neuron type calculation when using preprocessed data
- **Result**: Step 2 time 20.8s → <0.1s per worker

### 3. Parallel Execution Framework
- Joblib-based parallel simulation
- Each worker independently builds network and runs trials
- **Result**: 5.14× speedup on 6-core (1.8 min vs 9.3 min serial)

---

## 🚀 Quick Start

### Installation
```bash
# Clone repository
git clone https://github.com/ruiluohub/drosophila-lif-reproduction.git
cd drosophila-lif-reproduction

# Create conda environment
conda env create -f environment.yml
conda activate flylif

# Download FlyWire data (see Data section below)
```

### Basic Usage
```python
from flylif.core import load_simulation_data, build_network, run_simulation, DEFAULT_PARAMS
from pathlib import Path
from brian2 import Hz

# Configure paths
config = {
    'data_dir': Path('./data_783'),
    'connections_file': 'proofread_connections_783.feather',
    'root_ids_file': 'proofread_root_ids_783.npy',
}

# Load optimized data (68MB)
DATA = load_simulation_data(config)

# Build network (~7 seconds)
NET = build_network(
    data=DATA,
    pre_col=DATA['columns']['pre_col'],
    post_col=DATA['columns']['post_col'],
    weight_col=DATA['columns']['weight_col'],
    nt_prob_cols=DATA['columns'].get('nt_prob_cols', {}),
    params=DEFAULT_PARAMS
)

# Run simulation
result = run_simulation(
    net_components=NET,
    neu_exc=[neuron_ids...],
    params={'r_poi': 100*Hz},
    n_trials=10
)

print(f"Active neurons: {result['n_active']}")
```

---

## 📁 Repository Structure

```
drosophila-lif-reproduction/
├── flylif/                      # Main package
│   ├── core/                    # Core modules
│   │   ├── parameters.py        # LIF model parameters
│   │   ├── data_loader.py       # Optimized data loading
│   │   ├── network.py           # Brian2 network construction
│   │   └── simulation.py        # Simulation execution
│   └── utils/
│       └── cave_utils.py        # FlyWire ID version conversion
│
├── test_exp1_clean.ipynb        # Validation notebook (clean environment)
├── repo_flylif_model.ipynb      # Original development notebook
│
├── README.md                    # This file
├── PROGRESS.md                  # Development log
├── TODO.md                      # Planned improvements
├── environment.yml              # Conda environment
└── .gitignore
```

---

## 📊 Performance Validation

### Test Configuration
- Neurons: 21 Sugar GRNs (left hemisphere)
- Frequencies: 5 points (10, 50, 100, 150, 200 Hz)
- Trials: 3

### Results
| Metric | Serial | Parallel (6-core) | Speedup |
|--------|--------|-------------------|---------|
| Total time | 9.3 min | 1.8 min | **5.14×** |
| Data size | 1.4 GB | 68 MB | 20× smaller |
| Memory/worker | - | ~2 GB | - |

### Scientific Validation
- **Pearson correlation** (100 Hz vs Shiu et al.): **r = 0.84**
- **Top 50 neuron overlap**: 84%
- Expected improvement with 10 trials: r > 0.85

---

## 📈 Cloud Deployment Projection

Based on local 6-core testing (5.14× speedup):

| Hardware | Experiment 1 (19 freqs × 10 trials) | All 5 Experiments |
|----------|-------------------------------------|-------------------|
| Local 6-core | ~31 minutes | ~24 hours |
| AWS 48-core | **~1 minute** | **~1 hour** |

**Estimated AWS cost**: $2 (on-demand) or $0.60 (Spot instance)

---

## 💾 Data

FlyWire connectome data required (v783, not included):
- `proofread_connections_783.feather` (~3.5GB)
- `proofread_root_ids_783.npy` (~1MB)
- `classification.csv` (~20MB)

**Download instructions**: [To be added]

Place data files in `./data_783/` directory.

---

## 🔬 Experiments

### Implemented
- ✅ Experiment 1: Sugar GRN frequency sweep (validation complete)
- ✅ **Experiment 2**: Sufficiency test (200 neurons × 2 frequencies, 3× speedup)

### In Progress
- 🔄 **Experiment 3**: Necessity test (silencing, framework ready)

### Planned
- Experiment 4: Sugar + Bitter interaction
- Experiment 5: Sugar + Ir94e interaction


See `TODO.md` for implementation roadmap.

---

## 🐛 Known Issues

### Memory Limitation (16GB RAM)
- **Issue**: Parallel runs >12 tasks cause memory exhaustion (swap thrashing)
- **Impact**: Exponential slowdown after task 12
- **Workaround**: Use 3-4 workers or batch processing
- **Permanent fix**: Run on cloud instances with >32GB RAM

See `TODO.md` for detailed improvement plan.

---

## 📚 Documentation

- **PROGRESS.md**: Development log and optimization decisions
- **TODO.md**: Planned improvements and known issues
- **test_exp1_clean.ipynb**: Clean validation notebook

---

## 🎯 Project Milestones

- **2026-01-10**: Core reproduction completed (r=0.85 validation)
- **2026-01-23**: Repository created
- **2026-01-25**: Modularization complete, 68MB optimization, 5× parallel speedup
- **2026-01-27**: Tested 200 neurons for MN9 activation capability

---

## Citation

**Original paper:**
```bibtex
@article{shiu2024drosophila,
  title={A Drosophila computational brain model reveals sensorimotor processing},
  author={Shiu, Philip K and others},
  journal={Nature},
  year={2024}
}
```

**This implementation:**
```bibtex
@software{luo2026flylif,
  author={Luo, Rui},
  title={FlyLIF: Modularized Drosophila Brain LIF Model},
  year={2026},
  url={https://github.com/ruiluohub/drosophila-lif-reproduction}
}
```

---

## 👤 Contact

**Rui Luo, Ph.D.**   
📧 luorui.2016@tsinghua.org.cn

---

## 📄 License

[To be determined - likely MIT or match original paper]

---

**Repository Status**: Active development (0.2.0-alpha)  
**Last Updated**: 2026-01-27