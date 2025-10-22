#!/usr/bin/env python3
"""
Generate ESM-2 embeddings for the clean MARTS-DB dataset
Target: Germacrene binary classification (93 positive sequences)
"""

import pandas as pd
import numpy as np
import torch
from transformers import EsmModel, EsmTokenizer
import time
import json
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

def main():
    print('🧬 ESM-2 EMBEDDING GENERATION FOR GERMACRENE BINARY CLASSIFIER')
    print('=' * 70)
    
    # Load dataset
    print('📂 Loading clean MARTS-DB dataset...')
    df = pd.read_csv('data/clean_MARTS_DB_binary_dataset.csv')
    
    print(f'📊 Dataset loaded: {len(df)} sequences')
    print(f'   Germacrene sequences: {df["is_germacrene"].sum()}')
    print(f'   Non-germacrene sequences: {(df["is_germacrene"] == 0).sum()}')
    print(f'   Class balance: {df["is_germacrene"].mean()*100:.1f}% germacrene')
    
    # Check sequence lengths
    seq_lengths = df['Aminoacid_sequence'].str.len()
    print(f'   Sequence length range: {seq_lengths.min()}-{seq_lengths.max()} aa')
    print(f'   Average length: {seq_lengths.mean():.1f} aa')
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'🔧 Device: {device}')
    
    # Load ESM-2 model
    print('📦 Loading ESM-2 model: facebook/esm2_t33_650M_UR50D')
    model_name = "facebook/esm2_t33_650M_UR50D"
    
    try:
        tokenizer = EsmTokenizer.from_pretrained(model_name)
        model = EsmModel.from_pretrained(model_name)
        model.to(device)
        model.eval()
        print('✅ Model loaded successfully')
    except Exception as e:
        print(f'❌ Error loading model: {e}')
        return
    
    # Generate embeddings
    print(f'🚀 Generating embeddings for {len(df)} sequences...')
    print(f'   Batch size: 8')
    print(f'   Max length: 1024')
    
    batch_size = 8
    max_length = 1024
    embeddings = []
    
    start_time = time.time()
    
    with torch.no_grad():
        for i in range(0, len(df), batch_size):
            batch_sequences = df['Aminoacid_sequence'].iloc[i:i+batch_size].tolist()
            
            # Tokenize sequences
            inputs = tokenizer(
                batch_sequences,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_length
            )
            
            # Move to device
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            # Get embeddings
            outputs = model(**inputs)
            
            # Average pool the sequence representations (excluding padding tokens)
            attention_mask = inputs['attention_mask']
            token_embeddings = outputs.last_hidden_state
            
            # Mask out padding tokens
            masked_embeddings = token_embeddings * attention_mask.unsqueeze(-1)
            
            # Average pool
            pooled_embeddings = masked_embeddings.sum(dim=1) / attention_mask.sum(dim=1, keepdim=True)
            
            embeddings.append(pooled_embeddings.cpu().numpy())
            
            if (i + batch_size) % 100 == 0:
                elapsed = time.time() - start_time
                print(f'   Processed {min(i + batch_size, len(df))}/{len(df)} sequences ({elapsed:.1f}s)')
    
    # Concatenate all embeddings
    all_embeddings = np.concatenate(embeddings, axis=0)
    
    total_time = time.time() - start_time
    print(f'✅ Generated embeddings shape: {all_embeddings.shape}')
    print(f'⏱️  Generation completed in {total_time:.1f} seconds')
    
    # Save embeddings
    embeddings_path = 'data/germacrene_esm2_embeddings.npy'
    np.save(embeddings_path, all_embeddings.astype(np.float32))
    print(f'💾 Embeddings saved to: {embeddings_path}')
    
    # Verify embeddings
    loaded_embeddings = np.load(embeddings_path)
    print(f'✅ Verification: Loaded embeddings shape {loaded_embeddings.shape}')
    print(f'📊 Embeddings info:')
    print(f'   Data type: {loaded_embeddings.dtype}')
    print(f'   Memory usage: {loaded_embeddings.nbytes / 1024**2:.2f} MB')
    print(f'   Contains NaN: {np.isnan(loaded_embeddings).any()}')
    print(f'   Contains Inf: {np.isinf(loaded_embeddings).any()}')
    
    # Save metadata
    metadata = {
        'model_name': model_name,
        'dataset_size': len(df),
        'embedding_dim': all_embeddings.shape[1],
        'germacrene_sequences': int(df['is_germacrene'].sum()),
        'non_germacrene_sequences': int((df['is_germacrene'] == 0).sum()),
        'class_balance': float(df['is_germacrene'].mean()),
        'generation_time_seconds': total_time,
        'device_used': str(device),
        'batch_size': batch_size,
        'max_length': max_length
    }
    
    metadata_path = 'results/germacrene_embedding_metadata.json'
    Path('results').mkdir(exist_ok=True)
    
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f'📋 Metadata saved to: {metadata_path}')
    print(f'🎯 Ready for germacrene binary classification!')

if __name__ == "__main__":
    main()
