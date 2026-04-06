REFERENCE_EXPERIMENT_ID = "GNN_13_12_2025_version_7"

REFERENCE_EXPERIMENT = {
    "experiment_id": "GNN_13_12_2025_version_7",
    "seed": 2,
    "node_dim": 128,
    "edge_dim": 128,
    "global_dim": 141,
    "hidden_dim_global": 256,
    "hidden_dim_edge": 256,
    "hidden_dim_node": 512,
    "hidden_dim_decoder": 1024,
    "global_encoding_dim": 128,
    "num_layers": 4,
    "param_count": 4484736,
}

MOLECULES = [
    'CH3CO', 'NCO', 'CH3O', 'O2', 'CO2H', 'HOCN', 'C2H5', 'CH2OH',
    'C2H6', 'C2H3', 'CH2CO', 'NNH', 'H2CN', 'CH3OH', 'CN', '1CH2',
    'HNCO', 'O3P', 'C2H4', '3CH2', 'HCO', 'C2H2', 'H2CO', 'NH2', 'CO2',
    'OH', 'CH3', 'HCN', 'NH3', 'CH4', 'N2', 'CO', 'H2O', 'H', 'He',
    'H2', 'N2H2', 'N2H3', 'HCNH', 'HOCHO'
]

ATMOSPHERE_INFO_COLUMNS = [
    'co_ratio', 'metalicity', 'gravity', 'planet_radius',
    'planet_mass', 'T_irr', 'T_int', 'alpha', 'kappa_irr',
    'kappa_v1', 'kappa_v2', 'gamma_1', 'gamma_2'
]
