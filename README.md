# Graph_convolutional_neural_network

<img width="1560" height="888" alt="gnn" src="https://github.com/user-attachments/assets/31ed7a0e-009f-4833-ab0b-c065242949aa" />


## Requirements

To run the code, please install the following dependencies:

```bash
pip install networkx
pip install torch-geometric
```
Torch Geometric is an extension library for PyTorch designed specifically for building Graph Neural Networks (GNNs). It follows the same class structure and data handling conventions as PyTorch, making it easy to integrate with existing PyTorch workflows.

## Project Overview

This project aims to develop a potential biomarker for Autism Spectrum Disorder (ASD) using fMRI data and brain network configurations derived via BOLD signals. We construct whole-brain causal networks where:

- **Nodes** represent salient brain regions (defined by the EZ (Eickhoff-Zilles) atlas).
- **Edges** represent directed causal influences between regions.

We apply **Graph Neural Networks (GNNs)** to classify these brain networks. GNNs are designed to operate on data with an inherent graph structure, such as:

- fMRI signals aligned with anatomical atlases,
- social networks,
- and other non-Euclidean domains.

Unlike traditional CNNs, which rely on fixed grid-based kernels, GNNs aggregate features from neighboring nodes based on graph topology. This enables more flexible and biologically meaningful modeling of brain connectivity.

## Methodology

### Graph Construction

- **Nodes**: Brain regions based on the EZ atlas  
- **Edges**: Top *K* nearest neighbors from the Pearson correlation matrix  
- **Node Features**: fMRI neural activation signals

### Models Used

- **Graph Convolutional Network (GCN) Classifier**
- **Global Attention Network**

These models were trained and evaluated using the **ABIDE** dataset.

## References

- [fMRI-based networks defines the performance of a graph neural network for the classification](https://www.sciencedirect.com/science/article/abs/pii/S0960077922012206)  
- [Benchmarking Graph Neural Networks for FMRI analysis](https://arxiv.org/abs/2211.08927)
