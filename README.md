# A Staged LLM-Assisted Framework for Sensitivity Analysis Interpretation in Integrated Water System Models

**Imperial College London — MEng Civil Engineering Final Year Project**  
**Author:** Melina Lee   
**Submitted:** June 2026

---

## Overview

This repository contains the code for the FYP titled *"A Staged LLM-Assisted Framework for Sensitivity Analysis Interpretation in Integrated Water System Models."* The project develops a staged LLM-assisted sensitivity analysis (SA) interpretation pipeline as a foundational component for a future automated SA agent applied to integrated water system models.

Three contributions are implemented:

- **C1:** A four-stage interpretive framework that decomposes Sobol SA interpretation into discrete reasoning stages (Numerical Reasoning, Physical Reasoning, Literature Contextualisation, Action Recommendations).
- **C2:** A qualitative trustworthiness rubric derived from observed LLM failure modes, comprising four dimensions: Reasoning Consistency (D1), Source Grounding (D2), Contextual Specificity (D3), and Instruction Adherence (D4).
- **C3:** A RAG ablation study evaluating whether retrieval-augmented generation provides measurable added value over a non-RAG baseline across all four stages.

The model used is WSIMOD applied to the Barnoldswick catchment. Sobol sensitivity analysis is performed via SALib using Saltelli sampling (N=300, k=2). The LLM is Qwen-VL-8B, served locally via LM Studio.

---

## Repository Structure

```
/SA/
    experimentor.py              # Samples parameters and runs WSIMOD in parallel
    run_sa.py                    # Computes Sobol indices from per-iteration results
    sa_overrides.py              # Applies parameter overrides to WSIMOD config
    parameters/
        sa_parameters_bwick.py   # Parameter definitions and bounds

/LLM/
    stage1_no_rag.py             # Stage 1: Numerical Reasoning, no RAG
    stage1_rag.py                # Stage 1: Numerical Reasoning, with RAG
    stage2_no_rag.py             # Stage 2: Physical Reasoning, no RAG
    stage2_rag.py                # Stage 2: Physical Reasoning, with RAG
    stage3_no_rag.py             # Stage 3: Literature Contextualisation, no RAG
    stage3_rag.py                # Stage 3: Literature Contextualisation, with RAG
    stage4_no_rag.py             # Stage 4: Action Recommendations, no RAG
    stage4_rag.py                # Stage 4: Action Recommendations, with RAG

/v2/
    model/
        config.yml               # WSIMOD model configuration for Barnoldswick
```

---

## Requirements

Install dependencies with:

```bash
pip install -r requirements.txt
```

The LLM and embedding model are served locally via [LM Studio](https://lmstudio.ai/). Before running any stage script, ensure LM Studio is running with the following models loaded:

- **LLM:** `qwen/qwen3-vl-8b` 
- **Embedding model:** `text-embedding-nomic-embed-text-v1.5`

Both are served at `http://localhost:1234/v1`.

---

## Running the Pipeline

### Step 1 — Sensitivity Analysis

```bash
cd SA
python experimentor.py --jobid 0 --nproc 1
python run_sa.py
```

`experimentor.py` runs WSIMOD for each parameter sample and writes per-iteration CSVs to `SA/results/`. `run_sa.py` loads these results, computes Sobol indices, and writes `SA/results/sa_results_formatted.txt`.

### Step 2 — LLM-assisted Interpretation

Stages must be run in order (each stage loads the prior stage's JSON output). Run each script with a version tag:

```bash
cd LLM

# No-RAG condition
python stage1_no_rag.py --version v2
python stage2_no_rag.py --version v2
python stage3_no_rag.py --version v2
python stage4_no_rag.py --version v2

# RAG condition
python stage1_rag.py --version v2
python stage2_rag.py --version v2
python stage3_rag.py --version v2
python stage4_rag.py --version v2
```

Each script saves a JSON file containing the prompt, retrieved chunks (RAG only), and LLM response.

### RAG corpus

The RAG stages require the following papers to be placed in the `LLM/` directory. These are not included in this repository due to copyright:

- Nossent, J., Elsen, P., & Bauwens, W. (2011). Sobol' sensitivity analysis of a complex environmental model. *Environmental Modelling & Software*, 26(12), 1515–1525.  
  Used in: Stages 1 and 2 (stored in `chroma_db/`)

- Wagener, T., et al. (2022). On doing hydrology with dragons: Realizing the value of perceptual models and knowledge transfer. *WIREs Water*, 9(2), e1550.  
  Used in: Stage 3 (stored in `chroma_db_stage3/`)

The ChromaDB vector stores (`chroma_db/`, `chroma_db_stage3/`) are built automatically on first run if the PDFs are present.

---

## Key Configuration

| Parameter | Value |
|---|---|
| WSIMOD catchment | Barnoldswick |
| Sampled node | `3607-land` |
| Parameters | `surface_coefficient`, `percolation_coefficient` |
| Sampling method | Saltelli |
| N | 300 |
| Sobol indices | S1, ST, S2 via SALib |
| LLM | Qwen3-VL-8B |
| Embedding model | nomic-embed-text-v1.5 |
| Vector store | ChromaDB |
| Chunk size | 512 tokens |
| Chunk overlap | 50 tokens |
| Top-k retrieval | 6 |
| Similarity metric | Cosine similarity |

---

## Citation

If referencing this repository, please cite the associated dissertation:

> Lee, M., 2026. *A Staged LLM-Assisted Framework for Sensitivity Analysis Interpretation in Integrated Water System Models.* MEng Final Year Project, Imperial College London.
