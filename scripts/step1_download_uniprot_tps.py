#!/usr/bin/env python3
"""
Step 1: Download Terpene Synthase Sequences from UniProt
==========================================================

Downloads putative terpene synthase sequences from UniProt, excluding sequences
already in the MARTS-DB training set.

Usage:
    python step1_download_uniprot_tps.py --max_sequences 5000

Output:
    - data/uniprot_tps_sequences.fasta
    - data/uniprot_tps_metadata.csv
    
Next Step:
    Upload the FASTA file to Google Colab for ESM-2 embedding generation.
"""

import argparse
import time
import requests
import pandas as pd
from pathlib import Path
from typing import Set, List, Dict
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_marts_exclusion_list(file_path: str) -> Set[str]:
    """Load list of MARTS-DB Source IDs to exclude."""
    logger.info(f"Loading MARTS-DB exclusion list from {file_path}")
    with open(file_path, 'r') as f:
        ids = {line.strip() for line in f if line.strip()}
    logger.info(f"Loaded {len(ids)} MARTS-DB IDs to exclude")
    return ids


def download_uniprot_stream(query_url: str, description: str) -> List[str]:
    """
    Download sequences from UniProt using streaming API.
    
    Args:
        query_url: Full UniProt REST API URL
        description: Description for logging
        
    Returns:
        List of FASTA entries as strings
    """
    logger.info(f"Downloading {description}...")
    logger.info(f"URL: {query_url}")
    
    try:
        response = requests.get(query_url, stream=True)
        response.raise_for_status()
        
        # Handle compressed response
        if query_url.find('compressed=true') != -1:
            import gzip
            content = gzip.decompress(response.content).decode('utf-8')
        else:
            content = response.text
        
        # Parse FASTA entries
        fasta_entries = []
        current_entry = []
        
        for line in content.split('\n'):
            if line.startswith('>'):
                if current_entry:
                    fasta_entries.append('\n'.join(current_entry))
                current_entry = [line]
            elif line.strip():
                current_entry.append(line)
        
        # Add last entry
        if current_entry:
            fasta_entries.append('\n'.join(current_entry))
        
        logger.info(f"Downloaded {len(fasta_entries)} sequences")
        return fasta_entries
        
    except requests.exceptions.RequestException as e:
        logger.error(f"Error downloading from UniProt: {e}")
        return []


def parse_fasta_entry(fasta_text: str) -> Dict:
    """
    Parse a single FASTA entry into metadata dict.
    
    Example header:
    >sp|Q9XYZ1|TPS_SALOF Terpene synthase OS=Salvia officinalis OX=38868 GN=TPS PE=1 SV=1
    """
    lines = fasta_text.split('\n')
    header = lines[0]
    sequence = ''.join(lines[1:])
    
    # Parse header components
    parts = header[1:].split('|')
    
    if len(parts) >= 3:
        # SwissProt/TrEMBL format: sp|ACCESSION|ID ...
        accession = parts[1]
        rest = parts[2]
    else:
        # Simple format: >ACCESSION ...
        accession = parts[0].split()[0]
        rest = ' '.join(parts[0].split()[1:]) if len(parts[0].split()) > 1 else ''
    
    # Extract protein name (everything before OS=)
    protein_name = rest.split(' OS=')[0] if ' OS=' in rest else rest
    
    # Extract organism name
    organism = ''
    if ' OS=' in header:
        organism = header.split(' OS=')[1].split(' OX=')[0] if ' OX=' in header else header.split(' OS=')[1].split()[0]
    
    # Extract organism ID
    organism_id = ''
    if ' OX=' in header:
        ox_part = header.split(' OX=')[1]
        organism_id = ox_part.split()[0]
    
    # Extract gene name
    gene_name = ''
    if ' GN=' in header:
        gn_part = header.split(' GN=')[1]
        gene_name = gn_part.split()[0]
    
    return {
        'uniprot_id': accession,
        'protein_name': protein_name.strip(),
        'organism': organism.strip(),
        'organism_id': organism_id,
        'gene_name': gene_name,
        'length': len(sequence),
        'function': '',  # Not available from FASTA
        'sequence': sequence
    }


def query_uniprot_tps(max_results: int = 5000, exclusion_ids: Set[str] = None) -> List[Dict]:
    """
    Query UniProt for terpene synthase sequences using streaming API.
    
    Search strategy:
    1. Download ALL sequences from "terpene synthase family" (family classification)
    2. UniProt returns reviewed sequences first, then unreviewed
    3. Filter: no fragments
    4. EXCLUDE sequences in MARTS-DB training set
    5. Take first N sequences after exclusion (reviewed are prioritized automatically)
    
    Note: Using family:"terpene synthase family" ensures only true TPS enzymes,
    not false positives from keyword searches.
    """
    
    if exclusion_ids is None:
        exclusion_ids = set()
    
    all_sequences = []
    excluded_count = 0
    
    # Download from terpene synthase family (reviewed sequences come first automatically)
    family_url = (
        "https://rest.uniprot.org/uniprotkb/stream?"
        "compressed=true&format=fasta&"
        "query=%28%28family%3A%22terpene+synthase+family%22%29+AND+%28fragment%3Afalse%29%29"
    )
    
    logger.info("Downloading terpene synthase family sequences...")
    logger.info("Note: UniProt returns reviewed sequences first, then unreviewed")
    logger.info(f"Excluding {len(exclusion_ids)} MARTS-DB training sequences...")
    
    fasta_entries = download_uniprot_stream(family_url, "terpene synthase family")
    
    if not fasta_entries:
        logger.error("Failed to download sequences")
        return []
    
    logger.info(f"Downloaded {len(fasta_entries)} total sequences from terpene synthase family")
    
    # Parse entries and filter out MARTS-DB sequences, until we have max_results
    for fasta_entry in fasta_entries:
        if len(all_sequences) >= max_results:
            break
            
        try:
            parsed = parse_fasta_entry(fasta_entry)
            
            # Check if in exclusion list
            if parsed['uniprot_id'] in exclusion_ids:
                excluded_count += 1
                continue
            
            all_sequences.append(parsed)
            
        except Exception as e:
            logger.warning(f"Failed to parse FASTA entry: {e}")
    
    logger.info(f"Excluded {excluded_count} MARTS-DB sequences during download")
    logger.info(f"Parsed {len(all_sequences)} novel sequences (reviewed prioritized)")
    logger.info(f"Total sequences available in family: {len(fasta_entries)}")
    
    if len(fasta_entries) > max_results:
        logger.info(f"Note: {len(fasta_entries) - max_results - excluded_count} additional sequences available")
    
    return all_sequences


def filter_by_exclusion_list(results: List[Dict], exclusion_ids: Set[str]) -> List[Dict]:
    """Filter out sequences that are in MARTS-DB training set."""
    
    logger.info("Filtering sequences by MARTS-DB exclusion list...")
    
    filtered = []
    excluded_count = 0
    
    for entry in results:
        uniprot_id = entry.get('primaryAccession', '')
        
        # Check if this ID is in our exclusion list
        if uniprot_id in exclusion_ids:
            excluded_count += 1
            continue
        
        filtered.append(entry)
    
    logger.info(f"Excluded {excluded_count} sequences (already in MARTS-DB)")
    logger.info(f"Retained {len(filtered)} novel sequences")
    
    return filtered


def save_fasta(sequences: List[Dict], output_path: str):
    """Save sequences in FASTA format for Colab embedding generation."""
    
    logger.info(f"Saving FASTA to {output_path}")
    
    with open(output_path, 'w') as f:
        for seq in sequences:
            uniprot_id = seq['uniprot_id']
            protein_name = seq['protein_name']
            organism = seq['organism']
            
            # Write FASTA header
            f.write(f">{uniprot_id}|{organism}|{protein_name}\n")
            
            # Write sequence (wrapped at 80 characters)
            sequence = seq['sequence']
            for i in range(0, len(sequence), 80):
                f.write(sequence[i:i+80] + '\n')
    
    logger.info(f"Saved {len(sequences)} sequences to FASTA")


def save_metadata(sequences: List[Dict], output_path: str):
    """Save metadata as CSV for later analysis."""
    
    logger.info(f"Saving metadata to {output_path}")
    
    # Create DataFrame
    df = pd.DataFrame(sequences)
    
    # Select columns for metadata
    metadata_cols = ['uniprot_id', 'protein_name', 'organism', 'organism_id', 
                     'gene_name', 'length', 'function']
    
    df[metadata_cols].to_csv(output_path, index=False)
    logger.info(f"Saved metadata for {len(df)} sequences")


def main():
    parser = argparse.ArgumentParser(
        description='Download terpene synthase sequences from UniProt'
    )
    parser.add_argument(
        '--max_sequences',
        type=int,
        default=5000,
        help='Maximum number of sequences to download (default: 5000)'
    )
    parser.add_argument(
        '--exclusion_list',
        type=str,
        default='colab_upload/marts_db_source_ids.txt',
        help='Path to MARTS-DB exclusion list'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='data',
        help='Output directory for FASTA and metadata files'
    )
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load exclusion list
    exclusion_ids = load_marts_exclusion_list(args.exclusion_list)
    
    # Query UniProt (with exclusion filtering built-in)
    logger.info(f"Downloading up to {args.max_sequences} NOVEL sequences from UniProt...")
    results = query_uniprot_tps(max_results=args.max_sequences, exclusion_ids=exclusion_ids)
    
    if not results:
        logger.error("No sequences downloaded. Exiting.")
        return
    
    # Results are already filtered, just use them directly
    filtered_results = results
    
    # Save FASTA for Colab
    fasta_path = output_dir / 'uniprot_tps_sequences.fasta'
    save_fasta(filtered_results, str(fasta_path))
    
    # Save metadata CSV
    metadata_path = output_dir / 'uniprot_tps_metadata.csv'
    save_metadata(filtered_results, str(metadata_path))
    
    # Print summary
    logger.info("\n" + "="*60)
    logger.info("DOWNLOAD COMPLETE!")
    logger.info("="*60)
    logger.info(f"Total sequences: {len(filtered_results)}")
    logger.info(f"FASTA file: {fasta_path}")
    logger.info(f"Metadata file: {metadata_path}")
    logger.info("\nNext steps:")
    logger.info("1. Upload FASTA file to Google Colab")
    logger.info("2. Run ESM-2 embedding generation notebook")
    logger.info("3. Download embeddings and run step3_predict_germacrene.py locally")
    logger.info("="*60)


if __name__ == '__main__':
    main()

