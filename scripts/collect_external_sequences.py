#!/usr/bin/env python3
"""
Collect External Ent-Kaurene Synthase Sequences for Real-World Validation

This script identifies and collects ent-kaurene synthase sequences from sources
not included in our MARTS-DB training set for testing model generalizability.

Sources:
1. Recent patent filings (2023-2025)
2. Academic publications with novel sequences
3. NCBI GenBank sequences not in MARTS-DB
4. Recently characterized enzymes

Author: Cursor AI
Date: October 17, 2025
"""

import pandas as pd
import numpy as np
import requests
from Bio import Entrez, SeqIO
import time
import json
from pathlib import Path

class ExternalSequenceCollector:
    """Collect ent-kaurene synthase sequences from external sources"""
    
    def __init__(self, training_df_path="data/ent_kaurene_binary_dataset.csv"):
        """Initialize the collector"""
        self.training_df = pd.read_csv(training_df_path)
        self.training_sequences = set(self.training_df['Source_ID'].tolist())
        self.collected_sequences = []
        
        # Set up NCBI access
        Entrez.email = "your_email@example.com"  # Replace with actual email
        
    def get_ncbi_sequences(self, search_term="ent-kaurene synthase", max_results=50):
        """Search NCBI for ent-kaurene synthase sequences"""
        print(f"🔍 Searching NCBI for: {search_term}")
        
        try:
            # Search for protein sequences
            handle = Entrez.esearch(db="protein", term=search_term, retmax=max_results)
            record = Entrez.read(handle)
            handle.close()
            
            if not record["IdList"]:
                print("No sequences found in NCBI")
                return []
            
            print(f"Found {len(record['IdList'])} sequences")
            
            # Fetch sequence details
            ids = ",".join(record["IdList"])
            handle = Entrez.efetch(db="protein", id=ids, rettype="gb", retmode="text")
            records = list(SeqIO.parse(handle, "genbank"))
            handle.close()
            
            external_sequences = []
            for record in records:
                # Check if sequence is not in our training set
                if record.id not in self.training_sequences:
                    # Extract relevant information
                    source = "NCBI"
                    description = record.description
                    
                    # Get organism if available
                    organism = "Unknown"
                    for feature in record.features:
                        if feature.type == "source":
                            organism = feature.qualifiers.get("organism", ["Unknown"])[0]
                            break
                    
                    external_sequences.append({
                        'source_id': record.id,
                        'source': source,
                        'organism': organism,
                        'description': description,
                        'sequence': str(record.seq),
                        'length': len(record.seq),
                        'is_ent_kaurene': 1,  # Assuming positive since found in search
                        'collection_method': 'NCBI_search'
                    })
            
            print(f"✅ Collected {len(external_sequences)} new sequences from NCBI")
            return external_sequences
            
        except Exception as e:
            print(f"❌ Error searching NCBI: {e}")
            return []
    
    def get_patent_sequences(self):
        """Collect sequences from patent filings (manual curation needed)"""
        print("📋 Patent sequences require manual curation")
        
        # Based on search results, here are key patent sequences to investigate:
        patent_sources = [
            {
                'patent_id': 'US11725223',
                'year': 2023,
                'title': 'Microorganisms for Diterpene Production',
                'seq_ids': ['SEQ ID NO: 1', 'SEQ ID NO: 3', 'SEQ ID NO: 5', 'SEQ ID NO: 7'],
                'note': 'Multiple ent-kaurene synthase sequences listed'
            },
            {
                'patent_id': 'US20120058535',
                'year': 2012,
                'title': 'Biofuel Production in Prokaryotes and Eukaryotes',
                'organism': 'Phaeosphaeria nodorum',
                'note': 'ent-kaurene synthase from fungal source'
            },
            {
                'patent_id': 'US20250290081',
                'year': 2025,
                'title': 'Microorganisms for Steviol Glycoside Production',
                'note': 'Recent patent with ent-kaurene synthase sequences'
            }
        ]
        
        print("📄 Key patent sources identified:")
        for patent in patent_sources:
            print(f"  • {patent['patent_id']} ({patent['year']}): {patent['title']}")
        
        return patent_sources
    
    def get_recent_publications(self):
        """Identify recent publications with novel ent-kaurene synthase sequences"""
        print("📚 Recent publication sources identified:")
        
        publications = [
            {
                'title': 'Structure, function, and inhibition of ent-kaurene synthase from Bradyrhizobium japonicum',
                'year': 2014,
                'journal': 'PNAS',
                'organism': 'Bradyrhizobium japonicum',
                'note': 'Bacterial ent-kaurene synthase - structural characterization'
            },
            {
                'title': 'A tandem array of ent-kaurene synthases in maize with roles in gibberellin and specialized metabolism',
                'year': 2016,
                'journal': 'Plant Physiology',
                'organism': 'Zea mays',
                'note': 'Multiple ent-kaurene synthase genes in maize'
            },
            {
                'title': 'A pair of threonines mark ent-kaurene synthases for phytohormone biosynthesis',
                'year': 2015,
                'journal': 'Plant Cell',
                'note': 'Key residues in ent-kaurene synthases identified'
            }
        ]
        
        for pub in publications:
            print(f"  • {pub['year']}: {pub['title']} ({pub['organism'] if 'organism' in pub else 'Various'})")
        
        return publications
    
    def create_validation_dataset(self):
        """Create a validation dataset from external sources"""
        print("🚀 Creating external validation dataset...")
        
        # Collect sequences from various sources
        ncbi_sequences = self.get_ncbi_sequences()
        patent_sources = self.get_patent_sequences()
        publication_sources = self.get_recent_publications()
        
        # Combine all external sequences
        all_external = ncbi_sequences.copy()
        
        # Add metadata about other sources
        external_metadata = {
            'ncbi_sequences': len(ncbi_sequences),
            'patent_sources': len(patent_sources),
            'publication_sources': len(publication_sources),
            'total_external_sources': len(ncbi_sequences) + len(patent_sources) + len(publication_sources)
        }
        
        print(f"\\n📊 External dataset summary:")
        print(f"  • NCBI sequences: {external_metadata['ncbi_sequences']}")
        print(f"  • Patent sources: {external_metadata['patent_sources']}")
        print(f"  • Publication sources: {external_metadata['publication_sources']}")
        print(f"  • Total external sources: {external_metadata['total_external_sources']}")
        
        # Save external sequences
        if all_external:
            external_df = pd.DataFrame(all_external)
            external_df.to_csv('data/external_validation_sequences.csv', index=False)
            print(f"\\n💾 External sequences saved to: data/external_validation_sequences.csv")
        
        # Save metadata
        with open('results/external_validation_metadata.json', 'w') as f:
            json.dump(external_metadata, f, indent=2)
        
        return all_external, external_metadata

def main():
    """Main collection function"""
    print("🔬 EXTERNAL ENT-KAURENE SYNTHASE SEQUENCE COLLECTION")
    print("=" * 60)
    
    collector = ExternalSequenceCollector()
    
    # Create validation dataset
    external_sequences, metadata = collector.create_validation_dataset()
    
    print(f"\\n🎯 NEXT STEPS FOR REAL-WORLD VALIDATION:")
    print("1. Generate ESM-2 embeddings for external sequences")
    print("2. Run predictions using our best model (XGBoost)")
    print("3. Calculate performance metrics on unseen data")
    print("4. Create Figure 6: Real-world validation results")
    print("5. Compare performance vs. cross-validation results")
    
    if external_sequences:
        print(f"\\n✅ Ready for real-world validation with {len(external_sequences)} external sequences!")
    else:
        print(f"\\n⚠️  Limited external sequences found - may need manual curation")

if __name__ == "__main__":
    main()
