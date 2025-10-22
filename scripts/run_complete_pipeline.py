#!/usr/bin/env python3
"""
Complete Pipeline Runner for Ent-Kaurene Classification

This script runs the complete analysis pipeline from start to finish,
including ESM-2 embedding generation, ML benchmarking, and validation.

Author: Cursor AI
Date: October 17, 2024
"""

import os
import sys
import time
import subprocess
from pathlib import Path

def run_script(script_path, description):
    """Run a script and handle errors"""
    print(f"\n🚀 {description}")
    print("=" * 50)
    
    start_time = time.time()
    
    try:
        result = subprocess.run([
            sys.executable, script_path
        ], check=True, capture_output=True, text=True)
        
        end_time = time.time()
        duration = end_time - start_time
        
        print(f"✅ {description} completed successfully")
        print(f"⏱️  Duration: {duration:.1f} seconds")
        
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed")
        print(f"Error: {e}")
        print(f"stdout: {e.stdout}")
        print(f"stderr: {e.stderr}")
        return False

def check_file_exists(file_path):
    """Check if a file exists"""
    return Path(file_path).exists()

def main():
    """Run the complete pipeline"""
    print("🧬 ENT-KAURENE CLASSIFICATION COMPLETE PIPELINE")
    print("=" * 60)
    
    # Check if we're in the right directory
    if not check_file_exists("data/ent_kaurene_binary_dataset.csv"):
        print("❌ Dataset not found. Please run from the project root directory.")
        sys.exit(1)
    
    # Pipeline steps
    pipeline_steps = [
        ("scripts/generate_embeddings.py", "ESM-2 Embedding Generation"),
        ("scripts/ent_kaurene_benchmark.py", "ML Algorithm Benchmark"),
        ("scripts/holdout_validation.py", "Hold-out Validation"),
        ("scripts/corrected_traditional_benchmark.py", "Traditional Methods Comparison"),
        ("scripts/statistical_analysis.py", "Statistical Analysis"),
        ("scripts/dataset_characterization.py", "Dataset Characterization")
    ]
    
    # Run pipeline
    successful_steps = 0
    total_steps = len(pipeline_steps)
    
    for script_path, description in pipeline_steps:
        if not check_file_exists(script_path):
            print(f"⚠️  Skipping {description} - script not found: {script_path}")
            continue
            
        success = run_script(script_path, description)
        if success:
            successful_steps += 1
        else:
            print(f"⚠️  Pipeline failed at: {description}")
            break
    
    # Summary
    print(f"\n📊 PIPELINE SUMMARY")
    print("=" * 30)
    print(f"Completed steps: {successful_steps}/{total_steps}")
    
    if successful_steps == total_steps:
        print("🎉 Complete pipeline finished successfully!")
        print("\n📁 Check the 'results/' directory for outputs:")
        print("   • Performance tables (*.csv)")
        print("   • Detailed results (*.json)")
        print("   • Visualization figures (*.png)")
    else:
        print("⚠️  Pipeline incomplete. Check errors above.")
        sys.exit(1)

if __name__ == "__main__":
    main()
