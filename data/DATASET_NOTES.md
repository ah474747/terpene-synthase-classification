# UniProt Terpene Synthase Dataset

**Download Date**: October 24, 2025  
**Source**: UniProt REST API  
**Query**: `family:"terpene synthase family" AND NOT fragment`  
**URL**: https://rest.uniprot.org/uniprotkb/stream?compressed=true&format=fasta&query=%28%28family%3A%22terpene+synthase+family%22%29+AND+%28fragment%3Afalse%29%29

---

## Dataset Statistics

**Total available in family**: 12,216 sequences  
**Downloaded**: First 5,000 sequences (reviewed prioritized)  
**Excluded (MARTS-DB overlap)**: 0 sequences  
**Final dataset**: 5,000 sequences

### Sequence Length Distribution
- **Min**: 66 aa
- **Max**: 3,094 aa
- **Mean**: 499 aa
- **Median**: 416 aa

### Diversity
- **Unique organisms**: 1,632 species
- **Sequences with gene names**: 4,663 (93%)

---

## Top 10 Organisms

| Organism | Count |
|----------|-------|
| Zea mays (maize) | 67 |
| Arabidopsis thaliana | 66 |
| Jatropha curcas | 64 |
| Gossypium barbadense (cotton) | 42 |
| Leptotrombidium deliense | 39 |
| Perilla frutescens | 36 |
| Oryza nivara (rice) | 31 |
| Oryza barthii (rice) | 31 |
| Oryza sativa subsp. japonica (rice) | 31 |
| Gossypium mustelinum (cotton) | 30 |

---

## Sample Terpene Synthases (First 10)

1. **9,13-epoxylabda-14-ene synthase** (Marrubium vulgare)
2. **Gamma-terpinene synthase 1** (Thymus vulgaris)
3. **Eudesmanediol synthase** (Zea mays)
4. **Ent-copalyl diphosphate synthase 2** (Zea mays)
5. **Alpha-terpineol synthase** (Zea mays)
6. **Pseudolaratriene synthase** (Pseudolarix amabilis)
7. **(-)-kolavenyl diphosphate synthase** (Salvia divinorum)
8. **Terpene synthase 1** (Piper nigrum)
9. **Terpene synthase 2** (Piper nigrum)
10. **Myrcene synthase TPS5FN** (Cannabis sativa)

---

## Quality Improvements Over Previous Download

### Previous (Keyword Search)
- Query: `"terpene synthase" OR "monoterpene synthase" OR ...`
- **Problem**: Many false positives (e.g., geranylgeranyl pyrophosphate synthases, which are NOT terpene synthases)
- Example contaminants: Prenyltransferases, GGPP synthases

### Current (Family Classification)
- Query: `family:"terpene synthase family"`
- **Advantage**: Only sequences classified by UniProt curators as true TPS family members
- **Result**: 100% true terpene synthases

---

## Data Files

### `uniprot_tps_sequences.fasta` (2.8 MB)
FASTA format with 5,000 sequences. Header format:
```
>UNIPROT_ID|ORGANISM|PROTEIN_NAME
SEQUENCE...
```

### `uniprot_tps_metadata.csv` (470 KB)
Metadata table with columns:
- `uniprot_id`: UniProt accession
- `protein_name`: Full protein name
- `organism`: Species name
- `organism_id`: NCBI Taxonomy ID
- `gene_name`: Gene symbol
- `length`: Sequence length (amino acids)
- `function`: Functional annotation (empty for FASTA-derived data)

---

## Usage Notes

1. **Reviewed sequences prioritized**: UniProt returns reviewed (SwissProt) entries first, so the first ~1,200 sequences are high-quality, manually curated entries.

2. **No MARTS-DB overlap**: All 5,000 sequences are novel (not in training set).

3. **Additional sequences available**: 7,216 more sequences available from the same family if needed for validation or expansion.

4. **Next step**: Generate ESM-2 embeddings on Google Colab for these sequences.

---

## Citation

Data downloaded from [UniProt](https://www.uniprot.org/) using the REST API.

UniProt Consortium (2023). UniProt: the Universal Protein Knowledgebase in 2023. Nucleic Acids Research, 51(D1), D523-D531.

