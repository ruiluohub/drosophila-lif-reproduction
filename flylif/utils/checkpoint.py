"""
Checkpoint Manager for Resumable Experiments

Saves incremental progress to enable resume after interruption.
"""

import json
import pickle
from pathlib import Path
from datetime import datetime


class CheckpointManager:
    """
    Manage experiment checkpoints for resumable execution.
    
    Usage:
        ckpt = CheckpointManager('checkpoints/exp1')
        
        for freq in frequencies:
            if ckpt.is_completed(freq):
                continue
            result = run_simulation(freq)
            ckpt.save(freq, result)
    """
    
    def __init__(self, checkpoint_dir):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.progress_file = self.checkpoint_dir / 'progress.json'
        self.completed = self._load_progress()
    
    def _load_progress(self):
        """Load completed task list."""
        if self.progress_file.exists():
            with open(self.progress_file, 'r') as f:
                data = json.load(f)
            return set(data.get('completed', []))
        return set()
    
    def _save_progress(self):
        """Save progress file."""
        with open(self.progress_file, 'w') as f:
            json.dump({
                'completed': sorted(list(self.completed)),
                'last_update': datetime.now().isoformat(),
            }, f, indent=2)
    
    def is_completed(self, task_id):
        """Check if task already completed."""
        return task_id in self.completed
    
    def save(self, task_id, result):
        """Save result and mark as completed."""
        # Save result
        result_file = self.checkpoint_dir / f'task_{task_id}.pkl'
        with open(result_file, 'wb') as f:
            pickle.dump(result, f)
        
        # Update progress
        self.completed.add(task_id)
        self._save_progress()
    
    def load(self, task_id):
        """Load saved result."""
        result_file = self.checkpoint_dir / f'task_{task_id}.pkl'
        if result_file.exists():
            with open(result_file, 'rb') as f:
                return pickle.load(f)
        return None
    
    def load_all_completed(self):
        """Load all completed results."""
        results = {}
        for task_id in self.completed:
            result = self.load(task_id)
            if result:
                results[task_id] = result
        return results
    
    def get_remaining(self, all_tasks):
        """Get list of uncompleted tasks."""
        return [t for t in all_tasks if t not in self.completed]
    
    def clear(self):
        """Clear all checkpoints."""
        import shutil
        if self.checkpoint_dir.exists():
            shutil.rmtree(self.checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.completed = set()