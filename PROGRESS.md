# Development Progress Log

## 2026-01-25: Modularization & Optimization

### Completed
1. **Core Module Extraction**
   - Extracted `parameters.py` from notebook Section 2
   - Extracted `data_loader.py` with optimization logic
   - Extracted `network.py` with preprocessing adaptation
   - Extracted `simulation.py` for experiment execution

2. **Data Optimization**
   - **Problem**: Original data 1.4GB caused parallel overhead
   - **Solution**: Precompute neuron signs, filter syn_count≥5, remove 6 NT columns
   - **Result**: 95% reduction (1.4GB → 68MB)
   - **Impact**: Parallel speedup 2.58× → 5.14× (2× improvement)

3. **Network Construction Optimization**
   - **Problem**: `determine_neuron_types_fast()` repeated in each worker (20.8s)
   - **Solution**: Skip if `neuron_sign` column exists (use preprocessed)
   - **Result**: Step 2 time 20.8s → <0.1s
   - **Impact**: Total build time ~28s → ~7s

4. **Parallel Testing**
   - Validated joblib parallel mechanism (8.39× on 3-core demo)
   - Real code parallel test: 5.14× (6-core, 5 frequencies × 3 trials)
   - Scientific validation: r = 0.84 vs Shiu et al. (excellent)

### Issues Encountered

#### Issue 1: Jupyter Notebook Multiprocessing
- **Error**: `AttributeError: Can't get attribute 'function' on __main__`
- **Cause**: Functions defined in notebook cells cannot be pickled
- **Solution**: Use joblib (cloudpickle support)
- **Lesson**: Modularization necessary for parallel computing

#### Issue 2: Brian2 Object Reuse
- **Error**: `RuntimeError: neurongroup has already been simulated`
- **Cause**: Attempted to reuse NeuronGroup in multiple trials
- **Solution**: Each trial rebuilds network objects
- **Lesson**: Brian2 object lifecycle constraints

#### Issue 3: Memory Exhaustion (16GB Mac)
- **Symptom**: Performance degradation (task 1-6: 16min, task 13-15: 41min)
- **Cause**: 6 workers × 2GB + accumulation → 19.5GB swap
- **Impact**: 10000× slowdown when swapping to SSD
- **Solution**: Reduce workers (6→3) or batch processing

### Performance Metrics

| Metric | Value | Notes |
|--------|-------|-------|
| Data size (optimized) | 68 MB | vs 1.4GB original |
| Network build time | ~7 sec | Skip neuron type calculation |
| Single trial (100Hz) | ~54 sec | vs ~95s unoptimized |
| Parallel speedup (6-core) | 5.14× | Test: 5 freqs × 3 trials |
| Correlation with paper | r = 0.84 | 3 trials vs original 30 |

### Next Steps

#### Immediate (Next Session)
1. **Memory leak investigation**
   - Profile Brian2 object lifecycle
   - Add explicit garbage collection in workers
   - Test batch processing (split 19 freqs into 3 batches)

2. **Worker optimization**
   - Reduce to 3-4 workers for stable memory
   - Add memory monitoring in worker function
   - Implement checkpoint/resume for long runs

3. **Full Experiment 1 (19×10T)**
   - Option A: Local with 3 workers (~60 min)
   - Option B: AWS 48-core (~3 min, ~$0.20)

#### Short-term (This Week)
1. Create experiment scripts (exp2-5)
2. Implement checkpoint mechanism
3. Cloud deployment testing (8-core validation)

#### Mid-term (Next Week)
1. Run full experiments on cloud (Trial=10)
2. Detailed comparison with original paper
3. Documentation completion

### Technical Decisions

#### Data Loading Strategy
- **Chosen**: Preprocess in `data_loader.py` (main process, once)
- **Rejected**: Calculate in `network.py` (each worker, repeated)
- **Rationale**: 68MB transfer vs 1.4GB = 20× less overhead

#### Parallel Strategy  
- **Chosen**: Frequency-level parallelization (19 tasks for 19 frequencies)
- **Alternative**: Trial-level (less effective for many frequencies)
- **Rationale**: Better load balancing for large experiments

#### C++ Optimization
- **Decision**: Deferred (low priority)
- **Test result**: Small gain (<2×) due to recompilation overhead
- **Rationale**: Parallel (5×) > C++ (1.5×), focus on parallel first

---

## Performance Roadmap
Current (6-core Mac):
├─ Exp1 (19×10T): 30.7 min (projected)
├─ Memory limit: Cannot run 6 workers stably
└─ Workaround: 3 workers or batch processing
Target (48-core AWS):
├─ Exp1 (19×10T): ~1 min
├─ All 5 experiments (Trial=10): ~1 hour
├─ Cost: $2 (on-demand) or $0.60 (Spot)
└─ Memory: 96GB (no swap issues)


---

## 2026-01-26: Memory Optimization & Full Exp1 Completion

### Completed

1. **Memory Management**
   - **Issue**: 6 workers caused 19.5GB swap, exponential slowdown
   - **Solution**: Reference leak cleanup (gc.collect() in workers)
   - **Configuration**: Reduced to 3 workers for 16GB Mac
   - **Result**: Stable execution, memory <65%, swap <1GB

2. **Batched Execution with Checkpoint**
   - Implemented checkpoint system (incremental save/resume)
   - Split 20 frequencies into 4 batches (5 freqs each)
   - Each batch: ~17 min, 3 workers
   - Total runtime: 68 minutes
   - **Benefit**: Restart kernel between batches, prevent memory accumulation

3. **Full Experiment 1 Data**
   - **Parameters**: 20 frequencies (10-200Hz) × 10 trials
   - **Runtime**: 68 minutes (4 batches on 3-core Mac)
   - **Data**: Complete firing rate matrix (neurons × frequencies)
   - **Checkpoint**: All results saved incrementally

4. **Efficient ID Conversion**
   - **Problem**: Blind Top 50 conversion wasteful
   - **Solution**: Identify actual discrepancies first (244 IDs needed)
   - **Method**: 
     - Find missing neurons (v630 not in v783): 85
     - Find large firing rate differences: 159
     - Convert only these 244 IDs (vs blind 50)
   - **Result**: r improved 0.8151 → 0.9263 (+13.7%)

5. **Visualization Module**
   - Created `flylif/utils/visualization.py`
   - 4 reusable plot functions:
     - `plot_correlation()` - scatter with r value
     - `plot_response_heatmap()` - neurons × frequencies
     - `plot_frequency_response_curve()` - MN9 bilateral response
     - `plot_summary_statistics()` - bar charts
   - Flexible parameters: auto-detect + manual override
   - Generated 4 publication-quality figures

### Performance Metrics (Updated)

| Metric | Value | vs Previous |
|--------|-------|-------------|
| Exp1 runtime (20×10T, 3-core) | 68 min | First complete run |
| Memory peak | <65% | vs 90% (6-core) |
| Swap usage | <1 GB | vs 19.5GB (6-core) |
| **Correlation with paper** | **r = 0.9263** | vs 0.8151 (before ID fix) |
| Top 50 overlap | 86% | vs 84% (before) |

### Key Findings

#### Memory Behavior
- 3 workers stable: memory oscillates 48-65%
- Restart kernel effective: returns to ~48% baseline
- Reference leak mitigated: gc.collect() prevents accumulation

#### ID Version Impact
- 244/404 neurons (60%) needed ID conversion v630→v783
- After conversion: correlation jumped +11.1% (0.82→0.93)
- **Conclusion**: Connectome version critically impacts comparison

#### Scientific Validation
- **r = 0.9263** with 10 trials (vs original 30 trials)
- Demonstrates: 10 trials sufficient for stable results
- Ready for: Publication-quality experiments

### Files Created

**New modules**:
- `flylif/utils/checkpoint.py` - Incremental save/resume
- `flylif/utils/memory_utils.py` - Memory monitoring
- `flylif/utils/visualization.py` - Plotting tools

**Updated**:
- `flylif/core/simulation.py` - Added gc.collect() cleanup

**Notebooks**:
- `exp1_clean_test.ipynb` - Complete Exp1 with batched execution

**Output**:
- `results/exp1_full/` - 4 figures, firing rate matrix, complete results
- `checkpoints/exp1/` - 20 checkpoint files + progress.json
- `cache/id_conversions/` - Cached ID mappings

### Next Steps

**Immediate**:
1. Small-scale Exp2 test (20 neurons × 2 freqs)
2. Small-scale Exp3 test (10 neurons × 2 freqs + silencing optimization)
3. Validate checkpoint/memory strategies for Exp2/3

**Tomorrow**:
1. Cloud deployment preparation
2. Or medium-scale local tests (100 neurons)

### Lessons Learned

1. **ID version matters**: 60% neurons changed v630→v783, +11% correlation
2. **Memory management critical**: 3-core safer than 6-core on 16GB
3. **Checkpoint essential**: 68min run needs fault tolerance
4. **Efficient conversion**: Identify discrepancies first (244 vs blind 50)



---

## 2026-01-27: Exp2 Sufficiency Test Implementation

### Completed

1. **Parallel Batch Implementation**
   - Created `flylif/core/experiments.py`
   - Functions: `run_exp2_parallel()`, `run_exp3_parallel()`
   - Key optimization: Parallel across neuron-frequency pairs (avoid repeated network build in serial)

2. **Exp2 Full-Scale Test (200×2×1T)**
   - **Configuration**: 200 neurons × 2 frequencies × 1 trial = 400 conditions
   - **Method**: Split into 5 batches (40 neurons each), 4 workers
   - **Runtime**: ~32 minutes (5 batches × 6-7 min)
   - **Baseline comparison**: 96.2 min (projected) → 32 min (actual) = **3× speedup**
   - **Results**: Validated sufficiency test, identified neurons that activate MN9

3. **Performance Metrics**

| Batch | Workers | Tasks | Time | Memory | Swap |
|-------|---------|-------|------|--------|------|
| 1 | 3 | 80 | 7.6 min | 54.5% | 1.7GB* |
| 2 | 3 | 80 | 7.0 min | 45.5% | 0.0GB |
| 3 | 4 | 80 | 5.9 min | 47.6% | 0.0GB |
| 4 | 4 | 80 | ~6 min | - | - |
| 5 | 4 | 80 | ~6 min | - | - |

*Batch 1 swap due to system residual, cleared after restart

4. **Visualization Enhancement**
   - Updated `plot_response_heatmap()` with `neuron_order` parameter
   - Enables consistent neuron ordering across Exp1/Exp2/Exp3
   - Maintains comparability for cross-experiment analysis

### Technical Decisions

**Parallelization Strategy**
- **Chosen**: Neuron-frequency level parallelism (400 tasks for 200×2×1T)
- **Workers**: 4 (optimal for 16GB RAM after testing 3→4 transition)
- **Batch size**: 40 neurons (memory-safe, ~6 min/batch)
- **Rejected**: Worker-internal batching (Brian2 objects cannot be pickled/shared)

**Data Optimization Impact**
- Preprocessed data (68MB) enables faster network build (~7s)
- Each worker builds network independently (unavoidable with multiprocessing)
- Total build overhead: 400 × 7s / 4 workers = ~12 minutes (37% of total time)

**Speedup Analysis**
- Baseline (200×2×1T, serial): 96.2 min
- Optimized (200×2×1T, 4 workers): 32 min
- Speedup: **3× on 4-core (expected: 2.5-3×, accounting for build overhead)**
- Data optimization contribution: Build time 28s → 7s (4× faster per build)

### Results

**Exp2 Key Findings** (200×2×1T):
- Tested 200 neurons for MN9 activation capability
- Multiple neurons identified as sufficient to activate MN9
- Results saved: `./results/exp2_test/`
  - Complete results (.pkl)
  - Summary statistics (.csv)
  - Firing rate matrix (200×2, .csv)
  - Visualizations (heatmap, response curves)

### Next Steps

**Immediate**:
1. Implement Exp3 (necessity test, 200×2×1T) using same parallel framework
2. Validate silencing mechanism performance

**Short-term**:
1. Scale to full parameters (8 frequencies × 10 trials)
2. Implement 4-batch execution for Exp2/3 full-scale
3. Prepare cloud deployment scripts (48-core)

### Issues Encountered

**Issue 1: Worker Memory Leak**
- **Symptom**: Swap accumulated to 1.7GB during Batch 1
- **Cause**: System residual swap from previous runs
- **Solution**: Restart computer cleared swap completely
- **Prevention**: Monitor swap before starting, restart if >1GB

**Issue 2: 4 Workers Validation**
- **Test**: Increased from 3 to 4 workers at Batch 3
- **Result**: Improved speed (7.0min → 5.9min, 1.19× faster)
- **Memory**: Stable (47.6%, 0.0GB swap)
- **Decision**: Use 4 workers for remaining batches

**Issue 3: Front-20 Neurons Low Response**
- **Observation**: First 20 neurons (GRNs themselves) show minimal MN9 activation
- **Explanation**: Single GRN insufficient to activate MN9 (need ensemble)
- **Impact**: Validated need for testing full 200 neurons (downstream neurons respond)

### Lessons Learned

1. **Brian2 multiprocessing limitation**: Cannot share Network objects across workers
   - Each worker must build network independently
   - Data optimization (68MB) is critical for parallel efficiency

2. **Worker count optimization**: 4 workers optimal for 16GB RAM
   - 3 workers: stable but slower
   - 4 workers: 20% faster, memory still safe (<50%, 0 swap)

3. **Batch size selection**: 40 neurons/batch balances runtime and memory
   - Too large (100+): memory risk
   - Too small (20): excessive batching overhead

4. **Swap monitoring critical**: 
   - 1.7GB swap → performance degradation risk
   - Clean restart essential for long runs

---

**Version**: 0.2.0-alpha  
**Contributors**: Rui Luo  
**Date**: 2026-01-27
