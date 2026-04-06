# CHEXA-GNN

A graph neural network surrogate for predicting disequilibrium chemistry in exoplanet atmospheres.

This repository contains the code accompanying the paper **“Graph neural networks for exoplanet atmospheres”**. The model represents chemical species as nodes and reaction rates as edges, allowing information to propagate along chemically meaningful pathways rather than through the spatial neighbourhood bias of convolutional networks.

The network is trained on atmospheres generated with the **Venot 2020 chemical network** and **Guillot temperature-pressure profiles**, and is evaluated on both abundance reconstruction and transmission spectra. Compared to the previous U-Net approach, the GNN improves reconstruction accuracy and better reflects the topology of the underlying chemical network.

## Motivation

Disequilibrium chemistry is important for exoplanet atmospheric retrievals, but full kinetic calculations are computationally expensive. This project explores a physically informed graph neural network surrogate that accelerates these calculations while preserving chemically meaningful information flow.

## Main features

- graph-based representation of exoplanet chemistry
- chemical species treated as nodes and reaction rates as directed edges
- message passing guided by chemical connectivity
- trained on synthetic atmospheres with Guillot temperature-pressure profiles
- improved performance relative to the previous U-Net surrogate
- evaluated in abundance space and through transmission spectra


## Repository structure

```text
CHEXA-GNN/
├── data/                       # Split CSVs and training data locations
│   ├── train_split.csv
│   ├── val_split.csv
│   └── test_split.csv
├── models/                     # Core architecture
│   ├── __init__.py
│   ├── encoders.py             # EdgeMLP, GlobalEncoder, NodeUpdateMLP
│   ├── layers.py               # CustomGNNLayer
│   └── gnn_model.py            # Main GNNModel
├── run_samples/                # Preprocessed samples for inference
│   └── .gitkeep
├── scripts/                    # Entry points
│   ├── predict.py              # Run inference on preprocessed samples
│   └── train.py                # Train the final architecture
├── src/                        # Source logic
│   ├── __init__.py
│   ├── config.py               # Paths and hyperparameters
│   ├── constants.py            # Molecules and fixed metadata
│   ├── dataset.py              # AtmosphereGraphDataset
│   ├── inference.py            # Prediction logic
│   ├── training.py             # Training loop
│   └── utils.py                # Reproducibility and helper functions
├── trained_network/            # Saved trained weights
│   ├── .gitkeep
│   └── trained_chexa-gnn.pt
├── requirements.txt
└── README.md

## Requirements
numpy==2.2.5
pandas==2.3.3
scipy==1.15.3
torch>=2.5,<2.6
torch-geometric==2.7.0


## Paper

**Graph neural networks for exoplanet atmospheres**  
Antonia Vojtekova, Kai Hou Yip, Ingo P. Waldmann, Nikolaos Nikolaou, Olivia Venot, Ahmed Faris Al-Refaie, Bruno Merín

**INFORMATION NEEDED**  
- arXiv link  
- journal link / DOI  
- publication status

## Data

The model is trained on atmospheres generated using equilibrium chemistry, disequilibrium kinetic calculations, and extracted reaction rates.

