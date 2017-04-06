# Sequence to Sequence models with PyTorch

This repository contains implementation of (Attention based Sequence to Sequence models)[https://arxiv.org/abs/1508.04025] in PyTorch


## Sequence to Sequence models

An extension of sequence to sequence models that incorporate an attention mechanism was presented in https://arxiv.org/abs/1409.0473 that uses information from the RNN hidden states in the source language at each time step in the deocder RNN. This attention mechanism significantly improves performance on tasks like machine translation. A few variants of the attention model for the task of machine translation have been presented in https://arxiv.org/abs/1508.04025.


## Running

`python nmt.py --config <your_config_file>` ; only works on GPU
