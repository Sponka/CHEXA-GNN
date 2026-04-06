# CHEXA-GNN

Graph neural network surrogate for predicting disequilibrium chemistry in exoplanet atmospheres.

This project replaces the spatial bias of a U-Net with a graph-based representation in which chemical species are nodes and reaction rates are edges. The model is trained on atmospheres generated with the Venot 2020 chemical network and Guillot temperature-pressure profiles, and it improves abundance reconstruction compared to the previous U-Net baseline. :contentReference[oaicite:0]{index=0}

## Highlights

- graph neural network for exoplanet atmospheric chemistry
- physically meaningful message passing through chemical connectivity
- trained on Venot 2020 chemistry and Guillot T-P profiles
- improved performance over the previous U-Net model
- transmission spectra generally within JWST / Ariel precision range :contentReference[oaicite:1]{index=1} :contentReference[oaicite:2]{index=2}

## Repository status

This repository is being prepared for public release. Code, training scripts, and usage instructions will be added soon.

## Paper

**Graph neural networks for exoplanet atmospheres**  
Antonia Vojtekova et al.

## Citation

If you use this work, please cite the associated paper.
