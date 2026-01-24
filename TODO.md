# TODO: Improvements & Bug Fixes

## Critical (Before Full Run)

### 1. Memory Leak Fix
**Problem**: 6 workers cause memory exhaustion (19.5GB swap on 16GB Mac)

**Root cause**:
- Brian2 objects not properly released after each worker task
- Accumulation: 15 tasks → 19.5GB swap → 100× slowdown

**Solutions to test**:
1. Explicit garbage collection in workers
```python
   import gc
   del net_components
   gc.collect()
```

2. Reduce worker count
   - 6 workers → 3-4 workers (safer for 16GB RAM)
   - Trade parallelism for stability

3. Batch processing with kernel restart
```python
   # Batch 1: freq 10-70 (7 freqs)
   # Save results → Restart kernel
   # Batch 2: freq 80-140 (7 freqs)
   # Save results → Restart kernel  
   # Batch 3: freq 150-200 (5 freqs)
   # Merge results
```

**Priority**: HIGH  
**Estimated fix time**: 2 hours

---

### 2. Checkpoint/Resume Mechanism
**Problem**: 30-60 min runs vulnerable to interruption

**Implementation**:
```python
# checkpoint.py
class CheckpointManager:
    def save(self, freq, result):
        # Save individual frequency result
        
    def load_completed(self):
        # Return list of completed frequencies
        
    def resume_from(self, freq_list):
        # Skip completed, run remaining
```

**Benefits**:
- Resume after crash/interrupt
- Incremental progress saving
- Safe for long experiments

**Priority**: HIGH  
**Estimated time**: 3 hours

---

### 3. Memory Monitoring
**Problem**: Silent memory exhaustion

**Implementation**:
```python
def run_freq_worker(freq, ...):
    import psutil
    
    mem_before = psutil.virtual_memory().percent
    
    # ... run simulation ...
    
    mem_after = psutil.virtual_memory().percent
    
    if mem_after > 85:
        print(f"⚠️ Worker {freq}Hz: Memory {mem_after}%")
```

**Priority**: MEDIUM  
**Estimated time**: 1 hour

---

## Optimization Opportunities

### 4. Worker Memory Footprint
**Current**: ~2GB per worker

**Ideas**:
- Share read-only data across workers (not copied)
- Use memory-mapped arrays for connectivity
- Streaming results (don't hold all DataFrames)

**Priority**: MEDIUM  
**Estimated impact**: 30-40% memory reduction

---

### 5. Network Build Caching
**Current**: Each worker rebuilds network (~7s)

**Idea**: 
- Serialize built network once
- Workers load pre-built network
- Challenge: Brian2 objects may not pickle well

**Priority**: LOW (7s acceptable)  
**Estimated saving**: ~5s per worker

---

### 6. Code Elegance

**Issues identified**:
1. `determine_neuron_types_fast()` function still complex (20s if called)
   - Could simplify logic for preprocessed data path

2. Duplicate filtering check in `build_network()` Step 1
   - If data from `load_simulation_data()`, guaranteed filtered
   - Could add `pre_filtered` flag to skip check

3. Mixed language comments in original notebook
   - Clean English comments needed

**Priority**: LOW  
**For**: Code quality, maintainability

---

## Cloud Deployment TODO

### 7. AWS Setup Scripts
```bash
scripts/
├── cloud_setup.sh        # Install dependencies
├── download_data.sh      # Get FlyWire data
└── run_experiments.sh    # Execute all 5 experiments
```

### 8. Experiment Scripts
```python
scripts/
├── run_exp1_full.py      # 19 freqs × 10 trials
├── run_exp2_sufficiency.py
├── run_exp3_silencing.py
├── run_exp4_sugar_bitter.py
└── run_exp5_sugar_ir94e.py
```

**Priority**: HIGH (for next session)  
**Estimated time**: 4 hours

---

## Documentation TODO

### 9. Usage Documentation
- [ ] Installation guide
- [ ] Quick start tutorial
- [ ] API reference
- [ ] Experiment descriptions
- [ ] Cloud deployment guide

### 10. Scientific Validation
- [ ] Detailed comparison with original paper
- [ ] Statistical analysis of differences
- [ ] Discussion of v630 vs v783 connectome impact

---

## Testing TODO

### 11. Unit Tests
```python
tests/
├── test_data_loader.py
├── test_network.py
├── test_simulation.py
└── test_parallel.py
```

### 12. Integration Tests
- [ ] Full Experiment 1 (local 3-worker)
- [ ] Small cloud test (8-core)
- [ ] Memory stress test

---

## Known Issues

### Issue #1: Memory Leak in Parallel Workers
- **Severity**: High
- **Impact**: Cannot run >12 tasks on 16GB Mac
- **Workaround**: Batch processing or reduce workers
- **Fix needed**: Explicit cleanup in workers

### Issue #2: Slow Performance After Task 12
- **Symptom**: Exponential slowdown (task 15: 41min vs task 6: 16min)
- **Root cause**: Swap thrashing (19.5GB swap on 16GB RAM)
- **Temporary fix**: Restart kernel between batches

### Issue #3: ID Version Mismatch
- **Impact**: ~1% neurons may have wrong IDs (v630 vs v783)
- **Mitigation**: `cave_id()` conversion tool implemented
- **TODO**: Systematic conversion before all experiments

---

## Future Enhancements

- GPU acceleration (Brian2CUDA)
- Distributed computing (Dask/Ray)
- Real-time progress dashboard
- Automated result visualization
- Docker containerization

---

**Priority Legend**:
- HIGH: Blocks full experiment runs
- MEDIUM: Improves user experience
- LOW: Nice to have