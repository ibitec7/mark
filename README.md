# MaRK Adapters: Markov-adapted Recurrent Kernels for Hydra SSM

## Installation

### Using Dev Containers (Recommended)

If you have Docker and VS Code installed:

1. Clone the repository:
```bash
git clone https://github.com/ibitec7/experiment.git
cd experiment
```

2. Open in VS Code and select "Reopen in Container" when prompted, or run:
```bash
code . --remote-containers
```

The dev container includes all dependencies and CUDA support pre-configured.

### Local Development with uv

For local installation using `uv` (fast Python package manager):

```bash
git clone https://github.com/ibitec7/experiment.git
cd experiment
uv sync
```

### Traditional pip Installation

Alternatively, install dependencies with pip:

```bash
git clone https://github.com/ibitec7/experiment.git
cd experiment
pip install -r requirements.txt
```

Key dependencies:
- PyTorch (>=2.6.0) with CUDA support
- mamba-ssm (>=2.2.4) for Mamba-2 models
- transformers (>=4.48.3) for tokenizers
- einops for tensor operations
- wandb for experiment tracking
- nemo-toolkit for distributed training
