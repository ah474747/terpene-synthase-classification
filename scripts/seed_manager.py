#!/usr/bin/env python3
"""
Seed Management System for Reproducible Research
===============================================

This module provides comprehensive seed management for ensuring reproducible results
across all machine learning experiments and data processing steps.

Addresses reviewer feedback:
- Fixed seeds across all training scripts for reproducibility
- Consistent random state management
- Reproducible cross-validation splits
- Deterministic model training

Author: Andrew Horwitz
Date: October 2024
"""

import numpy as np
import random
import torch
import os
from sklearn.utils import check_random_state

# Global seed configuration
GLOBAL_SEED = 42
RANDOM_STATE = GLOBAL_SEED

class SeedManager:
    """Comprehensive seed management for reproducible research."""
    
    def __init__(self, seed=GLOBAL_SEED):
        self.seed = seed
        self.random_state = check_random_state(seed)
        
    def set_all_seeds(self):
        """Set seeds for all random number generators."""
        
        print(f"🌱 Setting global seed to {self.seed} for reproducibility...")
        
        # Python random
        random.seed(self.seed)
        
        # NumPy
        np.random.seed(self.seed)
        
        # PyTorch (if available)
        try:
            torch.manual_seed(self.seed)
            torch.cuda.manual_seed(self.seed)
            torch.cuda.manual_seed_all(self.seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            print("   ✅ PyTorch seeds set")
        except ImportError:
            print("   ⚠️  PyTorch not available, skipping PyTorch seeds")
        
        # Environment variable for additional libraries
        os.environ['PYTHONHASHSEED'] = str(self.seed)
        
        print("   ✅ All seeds set successfully")
        
    def get_random_state(self):
        """Get the configured random state."""
        return self.random_state
    
    def get_seed(self):
        """Get the current seed value."""
        return self.seed

# Global seed manager instance
seed_manager = SeedManager(GLOBAL_SEED)

def set_reproducible_seeds(seed=GLOBAL_SEED):
    """Convenience function to set all seeds for reproducibility."""
    global seed_manager
    seed_manager = SeedManager(seed)
    seed_manager.set_all_seeds()
    return seed_manager

def get_reproducible_random_state():
    """Get a reproducible random state for scikit-learn functions."""
    return seed_manager.get_random_state()

def ensure_reproducibility():
    """Ensure all random operations are reproducible."""
    seed_manager.set_all_seeds()

# Common seed configurations for different libraries
SEED_CONFIGS = {
    'sklearn': {'random_state': RANDOM_STATE},
    'xgboost': {'random_state': RANDOM_STATE},
    'torch': {'manual_seed': RANDOM_STATE},
    'numpy': {'seed': RANDOM_STATE},
    'cv_folds': {'random_state': RANDOM_STATE, 'shuffle': True}
}

def get_seed_config(library):
    """Get seed configuration for a specific library."""
    return SEED_CONFIGS.get(library, {'random_state': RANDOM_STATE})

# Example usage patterns
def example_sklearn_usage():
    """Example of how to use seeds with scikit-learn."""
    from sklearn.model_selection import StratifiedKFold
    from sklearn.ensemble import RandomForestClassifier
    
    # Set seeds
    ensure_reproducibility()
    
    # Use with scikit-learn
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    model = RandomForestClassifier(random_state=RANDOM_STATE)
    
    return cv, model

def example_xgboost_usage():
    """Example of how to use seeds with XGBoost."""
    import xgboost as xgb
    
    # Set seeds
    ensure_reproducibility()
    
    # Use with XGBoost
    model = xgb.XGBClassifier(random_state=RANDOM_STATE)
    
    return model

def example_torch_usage():
    """Example of how to use seeds with PyTorch."""
    try:
        import torch
        
        # Set seeds
        ensure_reproducibility()
        
        # PyTorch operations will now be reproducible
        tensor = torch.randn(10, 10)
        
        return tensor
    except ImportError:
        print("PyTorch not available")
        return None

if __name__ == "__main__":
    print("🌱 SEED MANAGEMENT SYSTEM")
    print("=" * 40)
    
    # Demonstrate seed management
    seed_manager = set_reproducible_seeds(42)
    
    # Show that results are reproducible
    print("\n🔬 Testing reproducibility...")
    
    # Generate some random numbers
    np.random.seed(RANDOM_STATE)
    rand1 = np.random.random(5)
    
    # Reset and generate again
    np.random.seed(RANDOM_STATE)
    rand2 = np.random.random(5)
    
    print(f"First generation:  {rand1}")
    print(f"Second generation: {rand2}")
    print(f"Are they equal? {np.allclose(rand1, rand2)}")
    
    print("\n✅ Seed management system ready!")
    print("   Import this module in your scripts for reproducible results.")
