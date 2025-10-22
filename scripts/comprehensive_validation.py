#!/usr/bin/env python3
"""
Comprehensive Real-World Validation with Decoy Sequences

This script creates a robust validation dataset including:
1. External ent-kaurene synthase sequences (positive controls)
2. Non-ent-kaurene terpene synthase sequences (negative controls/decoys)
3. Non-terpene synthase sequences (additional negative controls)

This addresses the user's question about testing both false negatives and false positives.

Author: Cursor AI
Date: October 17, 2025
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
from Bio import Entrez, SeqIO
import time
import random

class ComprehensiveValidator:
    """Create comprehensive validation dataset with positive and negative controls"""
    
    def __init__(self):
        """Initialize the validator"""
        self.external_positive = []  # External ent-kaurene synthases
        self.decoy_negative = []     # Non-ent-kaurene terpene synthases
        self.control_negative = []   # Non-terpene synthase sequences
        
    def load_external_sequences(self):
        """Load external ent-kaurene synthase sequences"""
        print("📂 Loading external ent-kaurene synthase sequences...")
        
        external_df = pd.read_csv('data/external_validation_sequences.csv')
        
        for _, row in external_df.iterrows():
            self.external_positive.append({
                'sequence_id': row['source_id'],
                'sequence': row['sequence'],
                'organism': row['organism'],
                'description': row['description'],
                'source': 'NCBI_External',
                'label': 1,  # Positive (ent-kaurene synthase)
                'category': 'external_ent_kaurene'
            })
        
        print(f"✅ Loaded {len(self.external_positive)} external ent-kaurene sequences")
        return len(self.external_positive)
    
    def collect_decoy_sequences(self):
        """Collect non-ent-kaurene terpene synthase sequences as negative controls"""
        print("🔍 Collecting decoy sequences (non-ent-kaurene terpene synthases)...")
        
        # Search terms for different types of terpene synthases
        decoy_search_terms = [
            "germacrene synthase",
            "limonene synthase", 
            "pinene synthase",
            "myrcene synthase",
            "linalool synthase",
            "caryophyllene synthase",
            "farnesene synthase",
            "bisabolene synthase"
        ]
        
        decoy_sequences = []
        
        for search_term in decoy_search_terms[:3]:  # Limit to avoid too many API calls
            print(f"   Searching for: {search_term}")
            try:
                # Search NCBI
                handle = Entrez.esearch(db="protein", term=search_term, retmax=20)
                record = Entrez.read(handle)
                handle.close()
                
                if record["IdList"]:
                    # Fetch sequences
                    ids = ",".join(record["IdList"][:10])  # Limit to 10 per search
                    handle = Entrez.efetch(db="protein", id=ids, rettype="gb", retmode="text")
                    records = list(SeqIO.parse(handle, "genbank"))
                    handle.close()
                    
                    for record in records:
                        # Skip if sequence is too short or too long
                        if 200 <= len(record.seq) <= 1000:
                            decoy_sequences.append({
                                'sequence_id': record.id,
                                'sequence': str(record.seq),
                                'organism': self._get_organism(record),
                                'description': record.description,
                                'source': 'NCBI_Decoy',
                                'label': 0,  # Negative (not ent-kaurene synthase)
                                'category': f'terpene_synthase_{search_term.replace(" ", "_")}'
                            })
                
                time.sleep(1)  # Be nice to NCBI
                
            except Exception as e:
                print(f"   ⚠️  Error searching for {search_term}: {e}")
        
        # Sample a reasonable number of decoys
        if len(decoy_sequences) > 50:
            decoy_sequences = random.sample(decoy_sequences, 50)
        
        self.decoy_negative = decoy_sequences
        print(f"✅ Collected {len(self.decoy_negative)} decoy sequences")
        return len(self.decoy_negative)
    
    def collect_control_sequences(self):
        """Collect non-terpene synthase sequences as additional negative controls"""
        print("🔬 Collecting control sequences (non-terpene synthases)...")
        
        # Search for common non-terpene synthase proteins
        control_search_terms = [
            "cytochrome oxidase",
            "ribulose bisphosphate carboxylase",
            "ATP synthase",
            "histone",
            "actin"
        ]
        
        control_sequences = []
        
        for search_term in control_search_terms[:2]:  # Limit API calls
            print(f"   Searching for: {search_term}")
            try:
                handle = Entrez.esearch(db="protein", term=search_term, retmax=15)
                record = Entrez.read(handle)
                handle.close()
                
                if record["IdList"]:
                    ids = ",".join(record["IdList"][:8])
                    handle = Entrez.efetch(db="protein", id=ids, rettype="gb", retmode="text")
                    records = list(SeqIO.parse(handle, "genbank"))
                    handle.close()
                    
                    for record in records:
                        if 200 <= len(record.seq) <= 1000:
                            control_sequences.append({
                                'sequence_id': record.id,
                                'sequence': str(record.seq),
                                'organism': self._get_organism(record),
                                'description': record.description,
                                'source': 'NCBI_Control',
                                'label': 0,  # Negative (not terpene synthase)
                                'category': f'control_{search_term.replace(" ", "_")}'
                            })
                
                time.sleep(1)
                
            except Exception as e:
                print(f"   ⚠️  Error searching for {search_term}: {e}")
        
        if len(control_sequences) > 30:
            control_sequences = random.sample(control_sequences, 30)
        
        self.control_negative = control_sequences
        print(f"✅ Collected {len(self.control_negative)} control sequences")
        return len(self.control_negative)
    
    def _get_organism(self, record):
        """Extract organism name from NCBI record"""
        for feature in record.features:
            if feature.type == "source":
                return feature.qualifiers.get("organism", ["Unknown"])[0]
        return "Unknown"
    
    def create_validation_dataset(self):
        """Create comprehensive validation dataset"""
        print("🚀 Creating comprehensive validation dataset...")
        
        # Load external sequences
        n_external = self.load_external_sequences()
        
        # Collect decoy sequences
        n_decoy = self.collect_decoy_sequences()
        
        # Collect control sequences
        n_control = self.collect_control_sequences()
        
        # Combine all sequences
        all_sequences = self.external_positive + self.decoy_negative + self.control_negative
        
        # Create DataFrame
        validation_df = pd.DataFrame(all_sequences)
        
        # Shuffle the dataset
        validation_df = validation_df.sample(frac=1, random_state=42).reset_index(drop=True)
        
        # Save dataset
        validation_df.to_csv('data/comprehensive_validation_dataset.csv', index=False)
        
        print(f"\\n📊 COMPREHENSIVE VALIDATION DATASET CREATED:")
        print(f"   • External ent-kaurene sequences: {n_external}")
        print(f"   • Decoy terpene synthase sequences: {n_decoy}")
        print(f"   • Control non-terpene synthase sequences: {n_control}")
        print(f"   • Total validation sequences: {len(validation_df)}")
        print(f"   • Positive examples: {len(validation_df[validation_df['label'] == 1])}")
        print(f"   • Negative examples: {len(validation_df[validation_df['label'] == 0])}")
        
        # Save metadata
        metadata = {
            'n_external_ent_kaurene': n_external,
            'n_decoy_terpene_synthases': n_decoy,
            'n_control_non_terpene': n_control,
            'total_validation_sequences': len(validation_df),
            'positive_ratio': len(validation_df[validation_df['label'] == 1]) / len(validation_df),
            'validation_strategy': {
                'external_positive': 'NCBI sequences annotated as ent-kaurene synthase',
                'decoy_negative': 'NCBI sequences annotated as other terpene synthases',
                'control_negative': 'NCBI sequences annotated as non-terpene synthase proteins'
            }
        }
        
        with open('results/comprehensive_validation_metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)
        
        return validation_df, metadata

def main():
    """Main validation dataset creation"""
    print("🌍 COMPREHENSIVE REAL-WORLD VALIDATION DATASET CREATION")
    print("=" * 65)
    
    validator = ComprehensiveValidator()
    
    # Create comprehensive validation dataset
    validation_df, metadata = validator.create_validation_dataset()
    
    print(f"\\n💾 Files created:")
    print(f"   • data/comprehensive_validation_dataset.csv")
    print(f"   • results/comprehensive_validation_metadata.json")
    
    print(f"\\n🎯 VALIDATION STRATEGY EXPLAINED:")
    print(f"   1. EXTERNAL POSITIVE CONTROLS ({metadata['n_external_ent_kaurene']} sequences):")
    print(f"      - NCBI sequences annotated as ent-kaurene synthase")
    print(f"      - Test for FALSE NEGATIVES (should be predicted as positive)")
    print(f"      - Measures model's ability to generalize to unseen ent-kaurene sequences")
    print()
    print(f"   2. DECOY NEGATIVE CONTROLS ({metadata['n_decoy_terpene_synthases']} sequences):")
    print(f"      - NCBI sequences annotated as OTHER terpene synthases")
    print(f"      - Test for FALSE POSITIVES (should be predicted as negative)")
    print(f"      - Measures model's specificity for ent-kaurene vs. other terpenes")
    print()
    print(f"   3. CONTROL NEGATIVE CONTROLS ({metadata['n_control_non_terpene']} sequences):")
    print(f"      - NCBI sequences annotated as NON-terpene synthase proteins")
    print(f"      - Additional test for FALSE POSITIVES")
    print(f"      - Measures model's specificity vs. completely unrelated proteins")
    
    print(f"\\n✅ Ready for comprehensive real-world validation!")

if __name__ == "__main__":
    main()
