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

## Repository contents

**INFORMATION NEEDED**  
Describe briefly what each folder contains once uploaded. For example:

- `src/` — model architecture, training, and inference
- `data/` — processed metadata or example inputs
- `notebooks/` — exploratory analysis and visualisation
- `scripts/` — training and evaluation scripts
- `figures/` — figures from the paper
- `requirements.txt` or `environment.yml` — dependencies

## Status

This repository is currently being prepared for public release. Code, documentation, and reproducibility instructions will be added progressively.

## Paper

**Graph neural networks for exoplanet atmospheres**  
Antonia Vojtekova, Kai Hou Yip, Ingo P. Waldmann, Nikolaos Nikolaou, Olivia Venot, Ahmed Faris Al-Refaie, Bruno Merín

**INFORMATION NEEDED**  
- arXiv link  
- journal link / DOI  
- publication status

## Data

The model is trained on atmospheres generated using equilibrium chemistry, disequilibrium kinetic calculations, and extracted reaction rates.

**INFORMATION NEEDED**  
- whether data will be uploaded to this repository
- whether only processed data will be shared
- external download link, DOI, or contact method
- licence or access restrictions

## Installation

```bash
git clone https://github.com/Sponka/CHEXA-GNN.git
cd CHEXA-GNN
