# micrograd — Notes & Implementation

My personal implementation + study notes for Andrej Karpathy’s **micrograd**.

- Original repo (forked for traceability): https://github.com/banatehrani/micrograd
- My work lives here: code in `src/`, notes in `notes/`, experiments in `experiments/`

## What I’m building
- A tiny autograd engine (reverse-mode autodiff)
- A small neural network library on top (MLP)
- Training on a toy dataset to verify gradients + learning

## Repo structure
- `src/` — implementation code (clean modules)
- `experiments/` — quick runs / notebooks / plots
- `notes/` — explanations + insights from the videos

## Progress
- [ ] Value class + basic ops
- [ ] Backprop via topological sort
- [ ] MLP module
- [ ] Training loop + sanity checks