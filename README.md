# Drosophila Brain LIF Model Reproduction

**Status**: Work in progress (documentation being refined)  
**Completion Date**: 2026-01-10  
**Based on**: Shiu et al. (2024) "A Drosophila computational brain model reveals sensorimotor processing" *Nature*

## Overview

This repository contains a reproduction of the connectome-constrained Leaky Integrate-and-Fire (LIF) model for the Drosophila brain.

**Current implementation:**
- ✅ Network construction from FlyWire v783 connectome (~139K neurons)
- ✅ Basic activation experiments (Sugar GRN stimulation)
- ✅ Validation against original paper (firing rate correlation r=0.85)

**Scope**: This reproduction validates the core methodology (network building, LIF implementation, basic simulations).

## Quick Start
```bash
# Install dependencies
pip install brian2 numpy pandas pyarrow caveclient matplotlib 

# View pre-computed results (recommended first)
jupyter notebook repo_flylif_model.ipynb
```

**Note**: The notebook includes all outputs from the validation run completed on 2026-01-10.

## Key Results

- **Network size**: 139,044 neurons, ~2M synapses (after filtering)
- **Firing rate correlation with original**: r = 0.85
- **Validation status**: Core methodology successfully reproduced

## Dependencies

- Brian2 >= 2.7.0
- Python >= 3.9
- FlyWire connectome data (v783)

## Data

FlyWire connectome data required (not included in repo):
- `proofread_connections_783.feather`
- `proofread_root_ids_783.npy`
- `classification.csv`

Place data files in `./data/` directory.

## Timeline

- **2026-01-10**: Core reproduction completed and validated
- **2026-01-23**: Repository created
- **Ongoing**: Documentation refinement

## Citation

Original paper:
```bibtex
@article{shiu2024drosophila,
  title={A Drosophila computational brain model reveals sensorimotor processing},
  author={Shiu, Philip K and others},
  journal={Nature},
  year={2024}
}
```

## Contact

  Rui Luo, Ph.D, Tsinghua University

  luorui.2016@tsinghua.org.cn

---

**Repository Status**: Private during documentation refinement.
