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

**Version**: 0.1.0-alpha  
**Date**: 2026-01-25  
**Contributors**: Rui Luo