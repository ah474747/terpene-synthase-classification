#!/usr/bin/env python3
"""
Complete Pipeline Runner for Terpene Synthase Classification
===========================================================

This script runs the complete terpene synthase classification pipeline,
addressing all reviewer feedback for reproducibility and completeness.

Features:
- Fixed seeds for reproducible results
- Comprehensive evaluation with statistical analysis
- Complete metric reporting (precision, recall, AUC-ROC, AUC-PR)
- Bootstrap confidence intervals
- Embedding visualization (UMAP/t-SNE)
- Traditional methods comparison
- Hold-out validation
- Progress tracking and logging

Author: Andrew Horwitz
Date: October 2024
"""

import os
import sys
import time
import logging
from pathlib import Path
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

# Import our modules
from scripts.seed_manager import ensure_reproducibility
from scripts.comprehensive_evaluation import ComprehensiveEvaluator
from scripts.embedding_visualization import EmbeddingVisualizer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('pipeline.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class TerpeneClassificationPipeline:
    """Complete pipeline for terpene synthase classification."""
    
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.start_time = time.time()
        
        # Ensure reproducibility
        ensure_reproducibility()
        logger.info(f"🌱 Pipeline initialized with random_state={random_state}")
        
        # Setup paths
        self.data_dir = project_root / "data"
        self.results_dir = project_root / "results"
        self.scripts_dir = project_root / "scripts"
        
        # Ensure directories exist
        self.data_dir.mkdir(exist_ok=True)
        self.results_dir.mkdir(exist_ok=True)
        
        # Initialize evaluator and visualizer
        self.evaluator = ComprehensiveEvaluator(random_state=random_state)
        self.visualizer = EmbeddingVisualizer(random_state=random_state)
        
    def log_progress(self, message, step=None, total_steps=None):
        """Log progress with optional step counter."""
        if step and total_steps:
            progress = (step / total_steps) * 100
            logger.info(f"[{progress:5.1f}%] {message}")
        else:
            logger.info(message)
    
    def check_prerequisites(self):
        """Check if all required files and dependencies are available."""
        
        self.log_progress("🔍 Checking prerequisites...")
        
        required_files = [
            self.data_dir / "clean_MARTS_DB_binary_dataset.csv",
            self.data_dir / "Original_MARTS_DB_reactions.csv"
        ]
        
        missing_files = []
        for file_path in required_files:
            if not file_path.exists():
                missing_files.append(str(file_path))
        
        if missing_files:
            logger.error(f"❌ Missing required files: {missing_files}")
            return False
        
        # Check if embeddings exist
        embedding_files = [
            self.data_dir / "germacrene_esm2_embeddings.npy",
            self.data_dir / "pinene_esm2_embeddings.npy",
            self.data_dir / "myrcene_esm2_embeddings.npy"
        ]
        
        missing_embeddings = [f for f in embedding_files if not f.exists()]
        
        if missing_embeddings:
            logger.warning(f"⚠️  Missing embedding files: {[f.name for f in missing_embeddings]}")
            logger.info("   Will generate embeddings as part of pipeline...")
        
        self.log_progress("✅ Prerequisites check completed")
        return True
    
    def generate_embeddings(self):
        """Generate ESM-2 embeddings for all target products."""
        
        self.log_progress("🧬 Generating ESM-2 embeddings...", 1, 8)
        
        target_products = ['germacrene', 'pinene', 'myrcene']
        
        for i, product in enumerate(target_products):
            self.log_progress(f"   Processing {product}...", 1, len(target_products))
            
            embedding_path = self.data_dir / f"{product}_esm2_embeddings.npy"
            
            if embedding_path.exists():
                self.log_progress(f"   ✅ {product} embeddings already exist, skipping...")
                continue
            
            # Import and run embedding generation
            try:
                from scripts.generate_embeddings import generate_embeddings_for_product
                
                generate_embeddings_for_product(
                    dataset_path=self.data_dir / "clean_MARTS_DB_binary_dataset.csv",
                    target_product=product,
                    output_path=embedding_path
                )
                
                self.log_progress(f"   ✅ {product} embeddings generated successfully")
                
            except Exception as e:
                logger.error(f"   ❌ Failed to generate {product} embeddings: {e}")
                return False
        
        self.log_progress("✅ All embeddings generated successfully")
        return True
    
    def run_ml_benchmarks(self):
        """Run machine learning benchmarks for all target products."""
        
        self.log_progress("🤖 Running ML benchmarks...", 2, 8)
        
        target_products = ['germacrene', 'pinene', 'myrcene']
        
        for i, product in enumerate(target_products):
            self.log_progress(f"   Benchmarking {product}...", i+1, len(target_products))
            
            # Import and run benchmark
            try:
                if product == 'germacrene':
                    from scripts.germacrene_benchmark import run_benchmark
                elif product == 'pinene':
                    from scripts.pinene_benchmark import run_benchmark
                elif product == 'myrcene':
                    from scripts.myrcene_benchmark import run_benchmark
                
                embedding_path = self.data_dir / f"{product}_esm2_embeddings.npy"
                dataset_path = self.data_dir / "clean_MARTS_DB_binary_dataset.csv"
                
                results = run_benchmark(
                    embeddings_path=embedding_path,
                    dataset_path=dataset_path
                )
                
                self.log_progress(f"   ✅ {product} benchmark completed")
                
            except Exception as e:
                logger.error(f"   ❌ Failed to run {product} benchmark: {e}")
                return False
        
        self.log_progress("✅ All ML benchmarks completed successfully")
        return True
    
    def run_traditional_benchmarks(self):
        """Run traditional methods benchmarks."""
        
        self.log_progress("📊 Running traditional methods benchmarks...", 3, 8)
        
        try:
            from scripts.germacrene_traditional_benchmark import run_traditional_benchmark
            
            embedding_path = self.data_dir / "germacrene_esm2_embeddings.npy"
            dataset_path = self.data_dir / "clean_MARTS_DB_binary_dataset.csv"
            
            results = run_traditional_benchmark(
                embeddings_path=embedding_path,
                dataset_path=dataset_path
            )
            
            self.log_progress("✅ Traditional methods benchmark completed")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to run traditional methods benchmark: {e}")
            return False
    
    def run_holdout_validation(self):
        """Run hold-out validation for the best performing model."""
        
        self.log_progress("🎯 Running hold-out validation...", 4, 8)
        
        try:
            from scripts.germacrene_holdout_validation import run_holdout_validation
            
            embedding_path = self.data_dir / "germacrene_esm2_embeddings.npy"
            dataset_path = self.data_dir / "clean_MARTS_DB_binary_dataset.csv"
            
            results = run_holdout_validation(
                embeddings_path=embedding_path,
                dataset_path=dataset_path
            )
            
            self.log_progress("✅ Hold-out validation completed")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to run hold-out validation: {e}")
            return False
    
    def create_comprehensive_evaluations(self):
        """Create comprehensive evaluations with statistical analysis."""
        
        self.log_progress("📈 Creating comprehensive evaluations...", 5, 8)
        
        target_products = ['germacrene', 'pinene', 'myrcene']
        
        for i, product in enumerate(target_products):
            self.log_progress(f"   Comprehensive evaluation for {product}...", i+1, len(target_products))
            
            try:
                # Load data
                embedding_path = self.data_dir / f"{product}_esm2_embeddings.npy"
                dataset_path = self.data_dir / "clean_MARTS_DB_binary_dataset.csv"
                
                embeddings = np.load(embedding_path)
                df = pd.read_csv(dataset_path)
                
                # Get labels
                label_col = f"is_{product}"
                if label_col not in df.columns:
                    logger.error(f"❌ Label column {label_col} not found in dataset")
                    continue
                
                labels = df[label_col].values
                
                # Load best model (XGBoost for all products)
                from xgboost import XGBClassifier
                model = XGBClassifier(random_state=self.random_state)
                
                # Run comprehensive evaluation
                results = self.evaluator.evaluate_model_comprehensive(
                    model=model,
                    X=embeddings,
                    y=labels,
                    target_name=product.title()
                )
                
                # Save results
                output_path = self.results_dir / f"{product}_comprehensive_evaluation.json"
                self.evaluator.save_results(results, output_path)
                
                # Create visualization
                viz_path = self.results_dir / f"{product}_comprehensive_evaluation.png"
                self.evaluator.create_visualization(results, viz_path)
                
                self.log_progress(f"   ✅ {product} comprehensive evaluation completed")
                
            except Exception as e:
                logger.error(f"   ❌ Failed comprehensive evaluation for {product}: {e}")
                return False
        
        self.log_progress("✅ All comprehensive evaluations completed")
        return True
    
    def create_embedding_visualizations(self):
        """Create embedding visualizations (UMAP/t-SNE)."""
        
        self.log_progress("🎨 Creating embedding visualizations...", 6, 8)
        
        try:
            # Load germacrene data for visualization
            embedding_path = self.data_dir / "germacrene_esm2_embeddings.npy"
            dataset_path = self.data_dir / "clean_MARTS_DB_binary_dataset.csv"
            
            embeddings = np.load(embedding_path)
            df = pd.read_csv(dataset_path)
            
            # Create visualization
            output_path = self.results_dir / "figure5_embedding_analysis.png"
            
            visualization_results = self.visualizer.create_embedding_visualization(
                embeddings=embeddings,
                labels=df,
                target_products=['is_germacrene', 'is_pinene', 'is_myrcene'],
                output_path=output_path
            )
            
            # Save analysis results
            analysis_path = self.results_dir / "embedding_analysis_results.json"
            self.visualizer.save_analysis_results(visualization_results, analysis_path)
            
            self.log_progress("✅ Embedding visualizations created")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to create embedding visualizations: {e}")
            return False
    
    def generate_manuscript_figures(self):
        """Generate all manuscript figures."""
        
        self.log_progress("📊 Generating manuscript figures...", 7, 8)
        
        try:
            from scripts.create_manuscript_figures import create_all_figures
            
            create_all_figures()
            
            self.log_progress("✅ All manuscript figures generated")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to generate manuscript figures: {e}")
            return False
    
    def generate_final_report(self):
        """Generate final comprehensive report."""
        
        self.log_progress("📄 Generating final report...", 8, 8)
        
        try:
            # Generate PDF manuscript
            from scripts.generate_pdf_simple import generate_pdf
            
            generate_pdf()
            
            # Create summary report
            self.create_summary_report()
            
            self.log_progress("✅ Final report generated")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to generate final report: {e}")
            return False
    
    def create_summary_report(self):
        """Create a summary report of all results."""
        
        summary_path = self.results_dir / "pipeline_summary_report.txt"
        
        with open(summary_path, 'w') as f:
            f.write("TERPENE SYNTHASE CLASSIFICATION PIPELINE SUMMARY\n")
            f.write("=" * 60 + "\n\n")
            
            f.write(f"Pipeline completed at: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Total runtime: {time.time() - self.start_time:.1f} seconds\n")
            f.write(f"Random state: {self.random_state}\n\n")
            
            f.write("FILES GENERATED:\n")
            f.write("-" * 20 + "\n")
            
            # List all generated files
            for file_path in self.results_dir.glob("*"):
                if file_path.is_file():
                    f.write(f"  {file_path.name}\n")
            
            f.write(f"\nRESULTS DIRECTORY: {self.results_dir}\n")
            f.write(f"LOG FILE: pipeline.log\n")
        
        logger.info(f"📄 Summary report saved to: {summary_path}")
    
    def run_complete_pipeline(self):
        """Run the complete pipeline."""
        
        logger.info("🚀 STARTING COMPLETE TERPENE SYNTHASE CLASSIFICATION PIPELINE")
        logger.info("=" * 80)
        
        pipeline_steps = [
            ("Check Prerequisites", self.check_prerequisites),
            ("Generate Embeddings", self.generate_embeddings),
            ("Run ML Benchmarks", self.run_ml_benchmarks),
            ("Run Traditional Benchmarks", self.run_traditional_benchmarks),
            ("Run Hold-out Validation", self.run_holdout_validation),
            ("Create Comprehensive Evaluations", self.create_comprehensive_evaluations),
            ("Create Embedding Visualizations", self.create_embedding_visualizations),
            ("Generate Manuscript Figures", self.generate_manuscript_figures),
            ("Generate Final Report", self.generate_final_report)
        ]
        
        for step_name, step_function in pipeline_steps:
            logger.info(f"🔄 {step_name}...")
            
            try:
                success = step_function()
                if not success:
                    logger.error(f"❌ Pipeline failed at step: {step_name}")
                    return False
                    
            except Exception as e:
                logger.error(f"❌ Pipeline failed at step {step_name}: {e}")
                return False
        
        # Pipeline completed successfully
        total_time = time.time() - self.start_time
        logger.info("🎉 PIPELINE COMPLETED SUCCESSFULLY!")
        logger.info(f"⏱️  Total runtime: {total_time:.1f} seconds ({total_time/60:.1f} minutes)")
        logger.info(f"📁 Results saved to: {self.results_dir}")
        logger.info(f"📄 Log saved to: pipeline.log")
        
        return True

def main():
    """Main function to run the complete pipeline."""
    
    print("🧬 TERPENE SYNTHASE CLASSIFICATION PIPELINE")
    print("=" * 60)
    print("Addressing reviewer feedback for reproducibility and completeness")
    print()
    
    # Initialize and run pipeline
    pipeline = TerpeneClassificationPipeline(random_state=42)
    
    success = pipeline.run_complete_pipeline()
    
    if success:
        print("\n✅ PIPELINE COMPLETED SUCCESSFULLY!")
        print("📊 All results and figures have been generated")
        print("📄 Check the results/ directory for outputs")
        sys.exit(0)
    else:
        print("\n❌ PIPELINE FAILED!")
        print("📋 Check pipeline.log for detailed error information")
        sys.exit(1)

if __name__ == "__main__":
    main()