# Data format

This file defines the current input and output structure expected by CHEXA-GNN.

It serves as a data contract for:
- training
- validation
- testing
- inference on preprocessed samples

---

## Training data structure

```text
data/
├── train_split.csv
├── val_split.csv
├── test_split.csv
├── mix_profiles_pytorch/
│   ├── <System_ID>.pt
├── ode_results_time_steps_pytorch/
│   ├── <System_ID>.pt
├── temperature_pt/
│   ├── <System_ID>.pt
├── edges/
│   ├── edges_<System_ID>.pt
└── configuration_normalised/
    └── system_configuration.csv
```

---

## Inference data structure

```text
run_samples/
├── sample_ids.csv
├── mix_profiles_pytorch/
│   ├── <System_ID>.pt
├── temperature_pt/
│   ├── <System_ID>.pt
├── edges/
│   ├── edges_<System_ID>.pt
└── system_configuration.csv
```

Note:
- `sample_ids.csv` is simply a list of sample names to run.
- It is not a train/test/validation split.

---

## Required files

### 1. Split or sample-ID CSV

Each split or inference CSV must contain a column named:

```text
System_ID
```

Example:

```csv
System_ID
12345
12346
12347
```

---

### 2. Node tensor

Path:

```text
mix_profiles_pytorch/<System_ID>.pt
```

Meaning:
- input equilibrium abundances

Expected shape:

```text
[40, 128]
```

---

### 3. Target tensor

Path:

```text
ode_results_time_steps_pytorch/<System_ID>.pt
```

Meaning:
- target disequilibrium abundances

Expected shape:

```text
[40, 128]
```

Required only for:
- training
- validation
- testing

Not required for:
- inference

---

### 4. Temperature tensor

Path:

```text
temperature_pt/<System_ID>.pt
```

Meaning:
- temperature profile

Expected shape:

```text
[128]
```

---

### 5. Edge file

Path:

```text
edges/edges_<System_ID>.pt
```

Content:

```python
(edge_index, edge_attr)
```

Expected shapes:

```text
edge_index: [2, num_edges]
edge_attr:  [num_edges, 128]
```

Notes:
- `edge_index` must be integer-valued
- `edge_attr` must be floating-point

---

### 6. Configuration CSV

Training path:

```text
data/configuration_normalised/system_configuration.csv
```

Inference path:

```text
run_samples/system_configuration.csv
```

Must contain a `System_ID` column and the following feature columns:

- `co_ratio`
- `metalicity`
- `gravity`
- `planet_radius`
- `planet_mass`
- `T_irr`
- `T_int`
- `alpha`
- `kappa_irr`
- `kappa_v1`
- `kappa_v2`
- `gamma_1`
- `gamma_2`

Notes:
- column names must match exactly
- the current spelling `metalicity` is preserved because this is what the code expects

---

## Global input vector

The model constructs the global input vector as:

```text
u = [temperature profile ; selected configuration values]
```

Expected shape before batching:

```text
[1, 141]
```

because:
- temperature profile = 128 values
- selected configuration values = 13 values

---

## Molecule order

The node and target tensors must use this exact molecule order:

```text
CH3CO, NCO, CH3O, O2, CO2H, HOCN, C2H5, CH2OH,
C2H6, C2H3, CH2CO, NNH, H2CN, CH3OH, CN, 1CH2,
HNCO, O3P, C2H4, 3CH2, HCO, C2H2, H2CO, NH2, CO2,
OH, CH3, HCN, NH3, CH4, N2, CO, H2O, H, He,
H2, N2H2, N2H3, HCNH, HOCHO
```

This order must remain consistent across:
- input node tensors
- target tensors
- saved predictions
- any postprocessing

---

## Current model dimensions

The current final model expects:

- number of molecules: `40`
- node feature dimension: `128`
- edge feature dimension: `128`
- global input dimension: `141`

These values are tied directly to the current preprocessing output and model architecture.

---

## Training requirements

Training requires:

- `train_split.csv`
- `val_split.csv`
- node tensors
- target tensors
- temperature tensors
- edge files
- configuration CSV

---

## Inference requirements

Inference requires:

- `sample_ids.csv`
- node tensors
- temperature tensors
- edge files
- configuration CSV
- trained checkpoint

Target tensors are optional and are not required for generating predictions.

---

## Prediction outputs

Predictions are saved as:

```text
run_samples/predicted/<System_ID>_pred.pt
```

Expected shape:

```text
[40, 128]
```

These files contain the model output in the same internal representation used by the model.

If inverse normalisation or scientific postprocessing is needed, that should be handled separately.

---

## Recommended validation checks for any future preprocessing pipeline

Any future preprocessing pipeline should verify that:

- every `System_ID` listed in the CSV exists in all required locations
- node tensor shape is `[40, 128]`
- target tensor shape is `[40, 128]` when targets are required
- temperature tensor shape is `[128]`
- `edge_index.shape[0] == 2`
- `edge_attr.shape[1] == 128`
- configuration CSV contains all required columns
- global input vector length is exactly `141`
- molecule ordering matches the expected ordering

---

## Future pipeline note

A future end-to-end pipeline may automate:

1. parameter input
2. equilibrium chemistry generation
3. reaction-rate extraction
4. graph construction
5. normalisation
6. CHEXA-GNN inference
7. optional postprocessing back to physical units

When that pipeline is added, this document should be updated to distinguish clearly between:

- raw physical inputs
- intermediate preprocessing products
- final model-ready tensors
