# LATTICE: LLM-guided Hierarchical Retrieval

[![arXiv](https://img.shields.io/badge/arXiv-2510.13217-b31b1b.svg)](https://arxiv.org/abs/2510.13217)
[![GitHub license](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Blog](https://img.shields.io/badge/Blog-Read%20More-yellow.svg)](https://nilesh2797.github.io/publications/lattice)
<!-- [![GitHub stars](https://img.shields.io/github/stars/nilesh2797/lattice.svg?style=social&label=Star)](https://github.com/nilesh2797/lattice) -->

**Official implementation of [LLM-Guided Hierarchical Retrieval](https://arxiv.org/abs/2510.13217)**  
*Nilesh Gupta, Wei-Cheng Chang, Ngot Bui, Cho-Jui Hsieh, Inderjit S. Dhillon*  
*UT Austin · UCLA · Google*


> LATTICE turns retrieval into an LLM-driven navigation problem over a semantic scaffold for computational tractability needed for large corpora.


## 📄 Overview

LATTICE proposes an *LLM-native retrieval* paradigm that combines the efficiency of hierarchical search with the reasoning power of modern large language models. Instead of relying on a static retriever + reranker pipeline or attempting to place a large corpus directly in an LLM context, LATTICE organizes the corpus into a semantic tree and uses an LLM as an *active search agent* that navigates that tree. This design yields logarithmic search complexity while preserving the LLM’s ability to perform nuanced, multi-step relevance judgments for complex, reasoning-heavy queries.

<p align="center">
  <img src="assets/lattice-overview.png" width="800">
</p>

Read more in the [blog](https://nilesh2797.github.io/publications/lattice) / [paper](https://arxiv.org/abs/2510.13217).

## 🚀 Usage

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

   For Google GenAI (default):
   ```bash
   export GOOGLE_API_KEY=your_api_key_here
   ```

   For vLLM (see [vLLM Setup](#running-with-vllm) below):
   ```bash
   # Start vLLM cluster first, then run experiments
   ```

### Quick Start

#### Running with Google GenAI (default)
Run a single experiment:
```bash
cd src
python run.py --subset biology --tree_version bottom-up --num_iters 20
```

Batch experiments:
```bash
cd src
bash run.sh
```

#### Running with vLLM

**1. Start the vLLM cluster:**

For data parallel mode (recommended for throughput):
```bash
cd scripts
./start_vllm_cluster.sh "Qwen/Qwen3-VL-8B-Instruct" data
```

For tensor parallel mode (for larger models):
```bash
cd scripts
./start_vllm_cluster.sh "meta-llama/Llama-2-70b-hf" tensor
```

The cluster will start 4 vLLM servers on ports 8000-8003 (data parallel) or a single server on port 8000 (tensor parallel).

**2. Check cluster status:**
```bash
cd scripts
./check_vllm_cluster.sh
```

**3. Run experiments with vLLM backend:**

Single experiment:
```bash
cd src
python run.py \
  --subset biology \
  --tree_version bottom-up \
  --num_iters 20 \
  --llm_api_backend vllm \
  --llm Qwen/Qwen3-VL-8B-Instruct \
  --llm_api_timeout 60 \
  --llm_api_max_retries 3
```

Edit [src/run.sh](src/run.sh) to enable vLLM for batch experiments:
```bash
# Uncomment these lines in COMMON_PARAMS:
--llm_api_backend vllm
--llm_api_staggering_delay 0.02
--llm_api_timeout 60
--llm_api_max_retries 3
--llm Qwen/Qwen3-VL-8B-Instruct
```

Then run:
```bash
cd src
bash run.sh
```

**4. Stop the vLLM cluster when done:**
```bash
cd scripts
./stop_vllm_cluster.sh
```

**vLLM Configuration:**
- **Data Parallel Mode**: Runs 4 independent servers (one per GPU) on ports 8000-8003. Best for maximum throughput with smaller models.
- **Tensor Parallel Mode**: Splits model across 4 GPUs on a single server (port 8000). Necessary for models that don't fit on a single GPU.
- Default GPU memory utilization: 95%
- Default max model length: 128K tokens
- GPU IDs used: 0, 1, 3, 4 (configurable in [scripts/start_vllm_cluster.sh](scripts/start_vllm_cluster.sh))

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
| **vLLM-specific parameters** | | |
| `--llm_api_backend` | LLM API backend (genai or vllm) | genai |
| `--llm` | Model name to use | gemini-2.0-flash-thinking-exp-01-21 |
| `--llm_api_timeout` | API call timeout in seconds | 180 |
| `--llm_api_max_retries` | Maximum number of API retries | 5 |
| `--llm_max_concurrent_calls` | Maximum concurrent API calls | 50 |
| `--llm_api_staggering_delay` | Delay between staggered API calls (seconds) | 0.1 |

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
│   ├── llm_apis.py         # LLM API wrappers (GenAI and vLLM)
│   ├── prompts.py          # Prompt templates
│   ├── utils.py            # Utility functions
│   └── calib_utils.py      # Calibration utilities
├── scripts/
│   ├── start_vllm_cluster.sh   # Start vLLM servers (data/tensor parallel)
│   ├── stop_vllm_cluster.sh    # Stop vLLM servers
│   └── check_vllm_cluster.sh   # Check vLLM server status
├── trees/
│   └── BRIGHT/             # Pre-built semantic trees
├── results/
│   └── BRIGHT/             # Experiment results
└── logs/                   # Execution logs (including vLLM server logs)
```

## 📈 Results
### Ranking results on BRIGHT
<p align="center">
  <img src="assets/lattice-bright-ndcg.png" width="600">
</p>

### Retrieval results & cost analysis on Stackexchange datasets from BRIGHT
<p align="center">
  <img src="assets/lattice-retrieval-plots.png" width="600">
</p>


## 📜 Citation

If you find this work helpful, please cite:

```bibtex
@article{gupta2025lattice,
  title={LLM-Guided Hierarchical Retrieval},
  author={Gupta, Nilesh and Chang, Wei-Cheng and Bui, Ngot and Hsieh, Cho-Jui and Dhillon, Inderjit S.},
  journal={arXiv preprint arXiv:2510.13217},
  year={2025}
}
```
