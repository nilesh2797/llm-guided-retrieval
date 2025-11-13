# LATTICE: LLM-guided Hierarchical Retrieval

[![arXiv](https://img.shields.io/badge/arXiv-2510.13217-b31b1b.svg)](https://arxiv.org/abs/2510.13217)
[![Colab](https://img.shields.io/badge/Colab-Notebook-blue.svg)](https://colab.research.google.com/drive/1AwDrHzipFVe-9kNAUJjCC4CNL4soRsqb?usp=sharing)
[![GitHub license](https://img.shields.io/badge/license-MIT-black.svg)](LICENSE)
[![Blog](https://img.shields.io/badge/Blog-Read%20More-yellow.svg)](https://nilesh2797.github.io/publications/lattice)

<!-- [![GitHub stars](https://img.shields.io/github/stars/nilesh2797/lattice.svg?style=social&label=Star)](https://github.com/nilesh2797/lattice) -->

<p align="center">
  <img src="assets/lattice-overview.png" width="800">
</p>

Read more in the [blog](https://nilesh2797.github.io/publications/lattice) / [paper](https://arxiv.org/abs/2510.13217) or try it out in the [colab notebook](https://colab.research.google.com/drive/1AwDrHzipFVe-9kNAUJjCC4CNL4soRsqb?usp=sharing)


## Overview

LATTICE proposes an *LLM-native retrieval* paradigm that combines the efficiency of hierarchical search with the reasoning power of modern large language models. Instead of relying on a static retriever + reranker pipeline or attempting to place a large corpus directly in an LLM context, LATTICE organizes the corpus into a semantic tree and uses an LLM as an *active search agent* that navigates that tree. This design yields logarithmic search complexity while preserving the LLM’s ability to perform nuanced, multi-step relevance judgments for complex, reasoning-heavy queries.

## Usage

### Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/nilesh2797/lattice
   cd lattice
   mkdir results trees
   ```

2. **Install dependencies:**
   ```bash
   pip install -r src/requirements.txt
   ```

3. **Download pre-built semantic trees:**
   ```bash
   git clone https://huggingface.co/datasets/quicktensor/lattice-bright-trees ./trees/BRIGHT
   ```

4. **Set up API credentials:**
   ```bash
   export GOOGLE_API_KEY=your_api_key_here
   ```

### Quick Start
Run a single experiment:
```bash
cd src; python run.py --subset biology --tree_version bottom-up --num_iters 20
```

Batch Experiments
```bash
cd src; bash run.sh
```

### Configuration

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--subset` | Dataset subset (biology, economics, etc.) | Required |
| `--tree_version` | Tree construction method (bottom-up/top-down) | Required |
| `--num_iters` | Number of retrieval iterations | 20 |
| `--max_beam_size` | Beam size during traversal | 2 |
| `--relevance_chain_factor` | Weight for current score in path relevance | 0.5 |
| `--reasoning_in_traversal_prompt` | Enable reasoning (thinking budget) | -1 (enabled) |
| `--rerank` | Additional reranking on final results | False |
| `--load_existing` | Resume from checkpoint defined by hyperparams | False |
| `--suffix` | Experiment name suffix | - |

For a complete list, see [`src/hyperparams.py`](src/hyperparams.py).

### Project Structure

```
lattice/release/
├── src/
│   ├── run.py              # Main execution script
│   ├── run.sh              # Batch execution wrapper
|   ├── run.ipynb           # Jupyter notebook for running / debugging experiments
│   ├── hyperparams.py      # Hyperparameter definitions
│   ├── tree_objects.py     # Semantic tree and sample objects
│   ├── llm_apis.py         # LLM API wrappers
│   ├── prompts.py          # Prompt templates
│   ├── utils.py            # Utility functions
│   └── calib_utils.py      # Calibration utilities
├── trees/
│   └── BRIGHT/             # Pre-built semantic trees
├── results/
│   └── BRIGHT/             # Experiment results
└── logs/                   # Execution logs
```

## Results
### Ranking results on BRIGHT
<p align="center">
  <img src="assets/lattice-bright-ndcg.png" width="600">
</p>

### Retrieval results & cost analysis on Stackexchange datasets from BRIGHT
<p align="center">
  <img src="assets/lattice-retrieval-plots.png" width="600">
</p>


## Cite

If you find this work helpful, please cite:

```bibtex
@article{gupta2025lattice,
  title={LLM-Guided Hierarchical Retrieval},
  author={Gupta, Nilesh and Chang, Wei-Cheng and Bui, Ngot and Hsieh, Cho-Jui and Dhillon, Inderjit S.},
  journal={arXiv preprint arXiv:2510.13217},
  year={2025}
}
```
