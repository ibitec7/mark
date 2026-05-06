# MaRK Adapters: Markov-adapted Recurrent Kernels for Hydra SSM

## Overview

**MaRK (Markov-adapted Recurrent Kernels)** introduces a novel third conditioning paradigm for State Space Models: **Dynamic Operator Modulation**. Rather than relying on external interventions like input-stream injection or Adaptive Layer Normalization (AdaLN), MaRK pushes the control plane directly into the continuous-time dynamics of the Hydra BERT SSM.

This repository contains the experimental implementation of MaRK, grounded in Cox's theory of Linear Parameter Varying (LPV) State Space Models. By dynamically modulating the A, B, C, D, and discretization (Δt) parameters of a frozen 111M-parameter Hydra backbone, MaRK achieves fine-grained conditioned generation with only 7.5–13M trainable parameters. The framework inherently guarantees Affine Quadratic Stability (AQS) through bounded parameter modulations, ensuring numerical stability and theoretical soundness. Combined with a learned Mamba-2 confidence scorer for intelligent token-level remasking, MaRK enables efficient iterative refinement in discrete diffusion tasks.

## Problem Statement: Limitations of Existing Conditioning Approaches

Adapting pre-trained State Space Models for conditional generation typically relies on two established paradigms:

1. **Input-Stream Injection**: Concatenate conditioning vectors directly into the sequence. While computationally cheap, it demands that the state-space dynamics route context across the entire sequence length via learned hidden states, often diluting the prompt signal and inducing temporal lag.

2. **Adaptive Layer Normalization (AdaLN)**: Dynamically modulate layer activations based on conditioning signals. While effective for Transformers, when applied to SSMs, AdaLN operates strictly externally to the core continuous-time sequence operator—it scales layer inputs/outputs but leaves the fundamental mathematically-defined temporal dynamics (the A, B, C, and Δ matrices) completely static regardless of context.

**Fundamental limitation**: Both paradigms leave the Markov parameter sequence—the complete characterization of input-output temporal behavior—entirely context-independent. This misses the opportunity to directly reshape the recurrent operator itself.

## Key Innovation: Dynamic Operator Modulation

MaRK solves this by introducing **Dynamic Operator Modulation**: a structurally principled approach that directly conditions the temporal recurrence engine itself.

- **Core mechanism**: Project the conditioning context (diffusion timestep + mask signal) into bounded, low-rank shifts and scales of the continuous-time SSM matrices (A, B, C, D, Δt).
- **Theoretical grounding**: Rooted in Linear Parameter Varying (LPV) theory, where each matrix becomes a function of the conditioning signal: A(c_t), B(c_t), C(c_t), D(c_t), Δ(c_t).
- **Behavioral invariant**: The Markov parameter sequence now becomes condition-dependent, meaning the system's input-output behavior fully adapts to context at each generation step—not just its surface activations.
- **Stability guarantee**: Bounded modulations via tanh saturation ensure Affine Quadratic Stability (AQS), mathematically guaranteeing bounded evolution over infinite context sequences.
- **Efficiency as emergent property**: Because stability mandates strict boundedness, parameter efficiency (7.5–13M adapters for a 111M base) emerges naturally rather than as an afterthought.

## Architecture

### Core Components (Implementation)

#### 1. Hydra SSM Kernel (Frozen Base)
The pre-trained Hydra BERT SSM with 23 layers serving as the frozen foundation. Contains the base A, B, C, D, and dt parameters learned during pre-training. These parameters are preserved during MaRK adaptation, enabling transfer learning while minimizing catastrophic forgetting.

#### 2. MaRK Adapters (Trainable, 7.5–13M parameters)
A family of LPV (Linear Parameter Varying) kernel adaptations that dynamically reconstruct Hydra's SSM parameters based on diffusion timesteps and mask embeddings. All adapters apply the same bounded modulation structure to preserve frozen Hydra weights while adapting recurrence (A), read-in (B), read-out (C), skip connections (D), and discretization (Δt):

**Three interchangeable kernel variants**:
- **Hypernet** (7.5M params): Direct linear projections from context to parameter shifts. Most parameter-efficient baseline.
- **Chebyshev Polynomial** (8M params, **default**): Orthogonal polynomial basis expansion (degree 5) with learnable latent coordinate. Provides smoother parameter trajectories and numerically stable gradients.
- **DCT Kernel** (13M params): Discrete Cosine Transform basis (8 frequency components) with decay regularization. Suppresses high-frequency noise accumulation across diffusion steps; most expressive but highest parameter cost.

**Modulation strategy**: All kernels use bounded shifts and scales to ensure Affine Quadratic Stability. Scalar parameters (A, Δt) are bounded via `tanh`, while matrix parameters (B, C) use low-rank (rank-2) factorization to avoid curse of dimensionality. Conservative initialization preserves pre-trained Hydra dynamics from the first training step.

#### 3. Mamba-2 Confidence Scorer (Fine-tuned, 130M parameters)
A Mamba-2 model fine-tuned to produce token-level confidence scores that guide iterative refinement—*not* core generation. The scorer operates on Hydra's intermediate token predictions and learns which tokens are most reliable at each diffusion step. Two scoring modes:
- **Softmax mode**: Emit raw token probabilities from Mamba-2 output layer
- **Pooling mode**: Single aggregated confidence per token via learned pooling layer

These scores drive top-K remasking: tokens with lowest confidence are marked for re-corruption in the next diffusion iteration, accelerating convergence to final predictions compared to uniform masking heuristics.

#### 4. Iterative Diffusion Process

Combines MaRK-conditioned Hydra predictions with Mamba-2 confidence scoring:
1. Initialize with prompt + fully masked response
2. Sample diffusion timestep and encode as conditioning signal
3. Forward through MaRK-conditioned Hydra to generate token logits
4. Score tokens via Mamba-2 guider (softmax or pooled confidence)
5. Apply top-K selection: remask lowest-confidence tokens for next iteration
6. Repeat until convergence (no tokens change their predictions)

### LPV-SSM Theoretical Foundation and Stability

#### Why Markov Parameters Matter

MaRK is grounded in Cox's theory of Linear Parameter Varying State Space Models. The key insight: two SSMs with identical *Markov parameter sequences* are behaviorally indistinguishable, regardless of their internal state coordinates. The Markov sequence fully characterizes input-output dynamics:

$$\mathcal{H}(c_t) = \left\{ h_k(c_t) = C'(c_t)\bar{A}'(c_t)^k B'(c_t) \right\}_{k=0}^{\infty}$$

By making each matrix condition-dependent—A(c_t), B(c_t), C(c_t), D(c_t), Δ(c_t)—MaRK directly adapts the system's input-output behavior at each generation step. This is fundamentally different from external layer normalization, which leaves these matrices static.

#### Affine Quadratic Stability (AQS) Guarantees

MaRK ensures numerical stability through rigorous control theory. All parameter modulations are bounded:
- **Scalar shifts (A, Δt)**: Constrained via `tanh` to $[-\beta, \beta]$ where $\beta$ is learnable but small
- **Matrix scales (B, C)**: Bounded via low-rank factorization + `tanh` to preserve spectral properties
- **Log-space A**: Prevents NaN gradients and ensures eigenvalues stay near unit circle

This boundedness guarantees Affine Quadratic Stability: there exists a common Lyapunov function P ≻ 0 such that the system's "energy" (measured by x^T P x) never grows unboundedly, even over infinite context sequences. Mathematically:
$$A'(v)^\top P A'(v) - P \preceq -\varepsilon I \quad \forall v \in \Psi$$

where Ψ is the hyper-rectangle of parameter variations.

#### Identifiability and Convergence

Despite internal parameter redundancy (low-rank factorizations, scaling ambiguities), the system's input-output behavior is identifiable at the optimal parametric rate O(1/√N). This is validated via **Markov Operator Error (MOE)**—the coordinate-invariant measure of how well learned operators match ground truth temporal dynamics. The paper includes synthetic LPV identification experiments demonstrating this convergence property across dataset sizes.

### Model Structure Recap

The standard SSM (frozen Hydra):
$$x_{t+1} = Ax_t + Bu_t, \quad y_t = Cx_t + Du_t$$

Becomes LPV-SSM with MaRK conditioning:
$$x_{t+1} = \bar{A}'(c_t)x_t + B'(c_t)u_t, \quad y_t = C'(c_t)x_t + D'(c_t)u_t$$

where all matrices are functions of the conditioning signal (timestep + mask embedding).

## Training Methodology

### Three-Stage Progressive Training Pipeline

MaRK adapters use a flexible three-stage training approach to progressively fine-tune the model. Each stage has different frozen components, learning rates, and dataset sizes:

#### Stage 1 - Adapter Pretraining
- **Frozen**: Hydra base parameters (all 23 layers)
- **Trainable**: MaRK adapter kernels only
- **Data**: ~60% of training dataset (3-5 epochs recommended)
- **Learning Rate**: High (2e-4 to 6e-4 for adapters)
- **Goal**: Initialize adapter weights while preserving pre-trained Hydra dynamics
- **Config**: `training_config_{kernel}_stage1.yaml`

#### Stage 2 - Joint Fine-tuning
- **Frozen**: Early layers of Hydra SSM (first 70% of base layers) and output prediction head
- **Trainable**: All MaRK adapter weights and final 30% of Hydra SSM base layers
- **Learning Rates**: High for adapters (2e-4 to 6e-4), low for trainable Hydra layers (~10x lower, e.g., 2e-5)
- **Data**: ~35% of training dataset (3-5 epochs)
- **Goal**: Co-optimize adapter and trainable base model layers while maintaining semantic coherence
- **Config**: `training_config_{kernel}_stage2.yaml`

#### Stage 3 - Head Fine-tuning
- **Frozen**: All except final prediction head
- **Trainable**: Prediction head only
- **Data**: ~5% of training dataset (3-5 epochs)
- **Learning Rate**: Low (typically 1e-4)
- **Goal**: Task-specific calibration of output layer
- **Config**: `training_config_{kernel}_stage3.yaml`

**Note**: These data splits (60/35/5) and epoch counts (3-5) are guidelines and can be adjusted per experiment. Careful monitoring is recommended to avoid overfitting to specific data distributions.

#### CART Loss Objective (Context-Adaptive Token-Level Noise Rescheduling)

Replaces standard masked language modeling loss with spatially-weighted position penalties to prevent mode collapse:

- **Mechanism**: Applies symmetric-geometric distance weighting based on token positions. Masked positions receive higher weight if they are far from unmasked anchor positions, encouraging contextual coherence rather than predicting the global marginal distribution.
- **Why it matters**: In early diffusion steps with heavy masking, standard MLM often produces mode collapse where all masked tokens predict identical tokens (the most frequent token in training data). CART breaks this symmetry.
- **Result**: Faster convergence, higher prediction entropy, reduced distributional bias compared to uniform MLM weighting.

For reference: Dream 1B model baseline achieves loss of 3.0-3.1 on similar tasks.

#### Mamba-2 Guider Training

Guider is fine-tuned as a supervised classifier on training data:
- **Input**: Hydra token predictions (with some positions potentially corrupted)
- **Output**: Confidence scores for each token position (softmax option) or single aggregate score (pooling option)
- **Loss**: Cross-entropy comparing predicted confidence with ground truth labels
- **Integration**: Scores direct top-K remasking decisions in the diffusion process

#### Performance Profiling

- **Comprehensive Analysis**: Enhanced PyTorch profiling capabilities for analyzing model performance, identifying bottlenecks, and optimizing computational efficiency across all components
- **Multi-Component Profiling**: Specialized profiling for Hydra SSM with MaRK adapters and Mamba-2 guider interactions
- **Trace Generation**: Support for Chrome and TensorBoard visualization formats
- **Experiment Tracking**: Uses Weights and Biases to log metrics, parameters, gradients, and GPU resource utilization throughout training stages

## Experimental Validation

The rigorous validation of the MaRK framework, include the following experiments:

- **Affine Quadratic Stability (AQS) Certificate**: Formal mathematical verification that bounded modulations guarantee stable system evolution over infinite sequences.

- **Synthetic LPV Identification**: Empirical demonstration that despite internal parameter redundancy (low-rank factorizations, scaling ambiguities), the learned input-output operators converge to equivalent temporal dynamics at the optimal parametric rate O(1/√N). Measured via Markov Operator Error (MOE)—the coordinate-invariant behavioral distance between learned and ground-truth systems.

- **Dynamics Analysis**: Investigation of eigenvalue distributions, discretization step trajectories, and rank trajectories across diffusion steps to understand how MaRK adapters modulate system behavior.

- **Sampling Efficiency**: Metrics including Expected Calibration Error (ECE) and median token argmax commitment (timestep where each token's prediction stabilizes) demonstrating accelerated convergence with Mamba-2 guider compared to uniform masking heuristics.

- **Ablation Studies**: Comparative analysis against existing paradigms (AdaLN-based systems, input-stream injection approaches) isolating the impact of continuous parameter reconstruction on generation quality and convergence speed.

## Research Context and Positioning

MaRK represents a fundamental departure from prior conditioning approaches for SSMs. The framework directly compares against two established baselines:

1. **AdaLN-based conditioning**: Scales/normalizes layer outputs externally; leaves core SSM dynamics static.
2. **Input-stream injection**: Concatenates conditioning into the token sequence; relies on learned state dynamics to propagate context.

MaRK's **Dynamic Operator Modulation** directly adapts the Markov parameter sequence, offering tighter control with fewer parameters (7.5–13M vs. typical adapter-baseline overhead). Empirical performance gains and sampling efficiency improvements are documented in the research paper.

**Scope and limitations**: Current experimental validation is at 111M parameter scale (Hydra base). Scaling to larger models (7B+ parameter regimes) is an open direction requiring further investigation.

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

## Project Structure

- **src/**: Core implementation directory
  - `hydra.py`: Base Hydra SSM kernel with 23 layers (frozen during MaRK adaptation)
  - `hydra_modules.py`: HydraEncoder orchestrating MaRK adapter dispatching via MarkEnsemble
  - `hydra_model.py`: HydraForMaskedLM wrapper with MaRK configuration (kernel type, ensemble settings)
  - `mark.py`: **MaRK adapters implementation** containing:
    - `HypernettedKernel`: Parameter-efficient MLPs for low-rank B/C modulation (7.5M params)
    - `ChebyshevPolynomial`: Stable Chebyshev basis expansion kernel (8M params, default)
    - `DCTKernel`: Frequency-domain DCT kernel for spectral dynamics (13M params)
    - `MarkEnsemble`: Orchestrates 23 adapters across Hydra layers in parallel
  - `guider.py`: Mamba-2 fine-tuned confidence scorer with softmax and pooling ablation options
  - `train.py`: Three-stage training loops with CART loss weighting
  - `nemo.py`: Nemo ModelPT wrapper for multi-GPU DDP training
  - `main.py`: Main entry point and training orchestrator
  - `data.py`: Dataset loading and preprocessing utilities
  - `performance.py`: PyTorch profiling and performance analysis
  - `inference.py`: Inference pipeline for diffusion process
  - `utils.py`: Training and data utilities
  - `ops.py`: Custom efficient operations
  - `padding.py`: Sequence padding logic
  - `test_*.py`: Unit tests for components
 
- **checkpoints/**: Saved model checkpoints for each kernel type and stage
  - `hydra_mark/`: Pre-training baseline
  - `hydra_mark_hypernet_stage{1,2,3}/`: Hypernet kernel checkpoints
  - `hydra_mark_chebyshev_stage{1,2,3}/`: Chebyshev kernel checkpoints
  - `hydra_mark_dct_stage{1,2,3}/`: DCT kernel checkpoints

- **configs/**: Stage-specific training configurations
  - `training_config_{kernel}_stage{1,2,3}.yaml`: 9 total (3 kernels × 3 stages)
  - Includes: model architecture, adapter config, learning rates, batch sizes, data paths
  - `hydra.yaml`: Model architecture defaults
  - `guider_config.yaml`: Mamba-2 guider configuration

- **data/**: Training and validation datasets
  - `train_shards{1,2,3}/`: Stage-specific training shards (60%, 35%, 5% splits)
  - `val_shards/`: Validation dataset (consistent across stages)
  - `test_shards/`: Test dataset
  - Various data formats: unpacked, shuffled, arrow format
  
- **models/**: Pre-trained model weights
  - `hydra_bert_23layers.pt`: Base Hydra SSM (111M params)
  - `hydra_chebyshev_mark.pt`, `hydra_dct_mark.pt`, `hydra_hypernet_mark.pt`: Trained MaRK adapters
  - `mamba-130m-hf.pt`: Mamba-2 base model for guider fine-tuning
  - `hydra_bert_metadata/`: Tokenizer and hyperparameter metadata

- **logs/**: Training logs and metrics
  
- **profiler_traces/**: Chrome and TensorBoard trace files

- **MaRK_adapters.md**: Detailed theoretical documentation of LPV-SSM and MaRK design
  
- **PROFILING.md**: Comprehensive profiling documentation

- **drafts/**: Research paper and theoretical materials (LaTeX, PDFs, proofs)
  - Not included in production; reference for understanding formal validation and theoretical grounding

## Usage

### Quick Start

#### Single-Kernel Training (Complete 3-Stage Pipeline)
Train a specific kernel type through all three stages:

```bash
# For Chebyshev kernel (default, recommended)
bash train_chebyshev_all_stages.sh

# For Hypernet kernel (most parameter efficient)
bash train_hypernet_all_stages.sh

# For DCT kernel (most expressive)
bash train_dct_all_stages.sh
```

Or run all kernels sequentially:
```bash
bash train.sh
```

#### Individual Stage Training
For manual control over training stages:

```bash
# Stage 1: Adapter pretraining
cp ./configs/training_config_chebyshev_stage1.yaml ./configs/training_config.yaml && python -m src.main

# Stage 2: Joint fine-tuning
cp ./configs/training_config_chebyshev_stage2.yaml ./configs/training_config.yaml && python -m src.main

# Stage 3: Head fine-tuning
cp ./configs/training_config_chebyshev_stage3.yaml ./configs/training_config.yaml && python -m src.main
```

## Experimental Exploration

### Kernel Type Investigation

MaRK supports three kernel implementations that can be explored and compared:

- **Hypernet**: Parameter-efficient MLP-based kernel
- **Chebyshev Polynomial**: Default kernel using polynomial basis expansion
- **DCT Kernel**: Frequency-domain kernel using discrete cosine transform

Training all three allows empirical comparison:
```bash
bash train.sh  # Runs all 3 kernels × 3 stages = 9 training runs
```

### Guider Confidence Scoring

MaRK supports two approaches for guider confidence scoring:

- **Option 1**: Raw softmax token probabilities from Mamba-2 output layer
- **Option 2**: Single aggregated probability per token via learned pooling layer

## Development Notes

- **Novel Conditioning Paradigm**: MaRK introduces Dynamic Operator Modulation, a fundamentally different approach compared to external layer normalization (AdaLN) or token-level conditioning (input-stream injection).

- **Parameter Efficiency**: Reduces trainable parameters to 7.5–13M (compared to 111M full Hydra) while preserving pre-trained knowledge through frozen base networks. Parameter reduction emerges naturally from stability constraints, not architectural tricks.

- **Grounded in LPV Theory**: Rooted in Cox's Linear Parameter Varying systems theory with rigorous Affine Quadratic Stability guarantees. Formal proofs included in research paper.

- **Numerical Safety**: All modulations are strictly bounded—scalar parameters via `tanh`, matrix parameters via low-rank factorization + softplus/tanh. A computed in log-space to prevent NaN gradients. Conservative initialization preserves pre-trained dynamics.

- **Loss Objective**: CART (Context-Adaptive Token-Level Noise Rescheduling) with symmetric-geometric position weighting prevents mode collapse faster than standard MLM, especially in heavy masking regimes.

- **Sampling Efficiency**: Mamba-2 confidence scorer enables intelligent iterative refinement, reducing required diffusion steps compared to uniform masking heuristics. Gains quantified via expected calibration error and median token argmax commitment.

- **Tokenizer**: Uses `bert-base-uncased` from HuggingFace.

- **Experiment Tracking**: Weights & Biases integration logs metrics, gradients, model structure, and GPU utilization per training stage.

- **Research Status**: Core MaRK framework is stable. Ongoing exploration includes scaling to larger models (7B+ regime), advanced kernel designs, and interaction with other diffusion objectives.

- **Reproducibility**: All experimental configurations, checkpoints, and training scripts included. Research paper in `drafts/` provides complete theoretical and empirical validation.
