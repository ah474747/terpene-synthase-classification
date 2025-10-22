#!/usr/bin/env python3
"""
Generate ESM-2 Embeddings for Ent-Kaurene Binary Classifier

This script generates ESM-2 protein language model embeddings for all sequences
in the ent-kaurene binary classification dataset.

Author: Cursor AI
Date: October 17, 2025
"""

import pandas as pd
import numpy as np
import torch
from transformers import EsmModel, EsmTokenizer
from tqdm import tqdm
import os
import time
from pathlib import Path

class ESM2EmbeddingGenerator:
    """Generate ESM-2 embeddings for protein sequences"""
    
    def __init__(self, model_name="facebook/esm2_t33_650M_UR50D"):
        """
        Initialize the ESM-2 embedding generator
        
        Args:
            model_name: ESM-2 model name to use
        """
        self.model_name = model_name
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        print(f"🔧 Device: {self.device}")
        print(f"📦 Loading ESM-2 model: {model_name}")
        
        # Load tokenizer and model
        self.tokenizer = EsmTokenizer.from_pretrained(model_name)
        self.model = EsmModel.from_pretrained(model_name)
        self.model.eval()
        
        # Move model to device
        self.model.to(self.device)
        
        print(f"✅ Model loaded successfully on {self.device}")
    
    def generate_embedding(self, sequence, max_length=1024):
        """
        Generate ESM-2 embedding for a single sequence
        
        Args:
            sequence: Protein sequence string
            max_length: Maximum sequence length to process
            
        Returns:
            numpy array: 1280-dimensional embedding
        """
        try:
            # Tokenize sequence
            inputs = self.tokenizer(
                sequence, 
                return_tensors="pt", 
                truncation=True, 
                max_length=max_length,
                padding=True
            )
            
            # Move inputs to device
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Generate embeddings
            with torch.no_grad():
                outputs = self.model(**inputs)
                # Use mean pooling of all token embeddings
                embeddings = outputs.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()
            
            return embeddings.astype(np.float32)
            
        except Exception as e:
            print(f"⚠️  Error generating embedding: {e}")
            # Return zero vector if error
            return np.zeros(1280, dtype=np.float32)
    
    def generate_embeddings_batch(self, sequences, batch_size=8, max_length=1024):
        """
        Generate embeddings for a batch of sequences
        
        Args:
            sequences: List of protein sequences
            batch_size: Batch size for processing
            max_length: Maximum sequence length
            
        Returns:
            numpy array: Array of embeddings (n_sequences, 1280)
        """
        print(f"🚀 Generating embeddings for {len(sequences)} sequences...")
        print(f"   Batch size: {batch_size}")
        print(f"   Max length: {max_length}")
        
        embeddings_list = []
        
        # Process in batches
        for i in tqdm(range(0, len(sequences), batch_size), desc="Processing batches"):
            batch_sequences = sequences[i:i + batch_size]
            
            # Tokenize batch
            inputs = self.tokenizer(
                batch_sequences,
                return_tensors="pt",
                truncation=True,
                max_length=max_length,
                padding=True
            )
            
            # Move to device
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Generate embeddings
            with torch.no_grad():
                outputs = self.model(**inputs)
                # Mean pooling
                batch_embeddings = outputs.last_hidden_state.mean(dim=1).cpu().numpy()
            
            embeddings_list.append(batch_embeddings)
        
        # Concatenate all embeddings
        all_embeddings = np.vstack(embeddings_list)
        print(f"✅ Generated embeddings shape: {all_embeddings.shape}")
        
        return all_embeddings.astype(np.float32)

def main():
    """Main function to generate ESM-2 embeddings"""
    print("🧬 ESM-2 EMBEDDING GENERATION FOR ENT-KAURENE BINARY CLASSIFIER")
    print("=" * 70)
    
    # Load the dataset
    data_file = "data/ent_kaurene_binary_dataset.csv"
    print(f"📂 Loading dataset: {data_file}")
    
    df = pd.read_csv(data_file)
    sequences = df['Sequence'].tolist()
    
    print(f"📊 Dataset loaded: {len(sequences)} sequences")
    print(f"   Sequence length range: {min(len(s) for s in sequences)}-{max(len(s) for s in sequences)} aa")
    print(f"   Average length: {np.mean([len(s) for s in sequences]):.1f} aa")
    
    # Initialize ESM-2 generator
    generator = ESM2EmbeddingGenerator()
    
    # Generate embeddings
    start_time = time.time()
    
    # Use batch processing for efficiency
    embeddings = generator.generate_embeddings_batch(
        sequences, 
        batch_size=8,  # Adjust based on memory
        max_length=1024  # ESM-2 max length
    )
    
    end_time = time.time()
    
    print(f"⏱️  Generation completed in {end_time - start_time:.1f} seconds")
    print(f"📊 Embeddings shape: {embeddings.shape}")
    
    # Save embeddings
    output_file = "data/esm2_embeddings.npy"
    np.save(output_file, embeddings)
    print(f"💾 Embeddings saved to: {output_file}")
    
    # Verify saved embeddings
    loaded_embeddings = np.load(output_file)
    print(f"✅ Verification: Loaded embeddings shape {loaded_embeddings.shape}")
    
    # Save metadata
    metadata = {
        'model_name': generator.model_name,
        'n_sequences': len(sequences),
        'embedding_dim': embeddings.shape[1],
        'generation_time': end_time - start_time,
        'device': str(generator.device),
        'batch_size': 8,
        'max_length': 1024
    }
    
    import json
    with open('results/embedding_metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"📋 Metadata saved to: results/embedding_metadata.json")
    
    print("\n🎉 ESM-2 embedding generation completed successfully!")
    print("   Ready for ML training!")

if __name__ == "__main__":
    main()
