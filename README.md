# micrograd — Notes & Implementation

My personal implementation **and study notes** for Andrej Karpathy’s **micrograd**.
This code was rebuilt step-by-step while following the videos — **not copy-pasted**.

- Original repo (forked for traceability): https://github.com/banatehrani/micrograd
- My implementation lives here:
  - code in `src/`
  - experiments in `experiments/`
  - notes in `notes/`

## What I’m building
- A tiny autograd engine (reverse-mode autodiff)
- A small neural network library on top (MLP)
- Training on a toy dataset to verify gradients + learning

The goal is learning, clarity, and traceable progress — not performance.

## Repo structure
- `src/` — implementation code (clean modules)
- `experiments/` — quick runs / notebooks / plots
- `notes/` — explanations + insights from the videos

## How this differs from Karpathy’s implementation

This repo follows the same core ideas as Karpathy’s micrograd, but differs in **style, tooling, and intent**:

- Written incrementally as a learning exercise (not a minimal reference)
- Uses modern Python features:
  - `dataclasses`
  - type hints
  - `from __future__ import annotations`
- Organized as an installable Python package (`src/` layout)
- Experiments are run as standalone scripts importing the package
- Deterministic training runs for reproducibility

The goal is clarity, structure, and traceable learning progress.

## Progress

- [x] Value class + basic ops
- [x] Backprop via topological sort
- [x] Autograd visualization (Graphviz)
- [x] MLP module (Neuron / Layer / MLP)
- [x] Training loop + sanity checks

## Run a tiny training demo

```bash
python experiments/train_mlp_tiny.py