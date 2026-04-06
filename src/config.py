from dataclasses import dataclass


@dataclass
class DataPaths:
    train_split_csv: str = "data/train_split.csv"
    val_split_csv: str = "data/val_split.csv"
    test_split_csv: str = "data/test_split.csv"
    node_dir: str = "data/mix_profiles_pytorch/"
    target_dir: str = "data/ode_results_time_steps_pytorch/"
    temp_dir: str = "data/temperature_pt/"
    config_csv: str = "data/configuration_normalised/system_configuration.csv"
    edge_dir: str = "data/edges/"


@dataclass
class RunSamplesPaths:
    base_dir: str = "run_samples/"
    split_csv: str = "run_samples/run_list.csv"
    node_dir: str = "run_samples/mix_profiles_pytorch/"
    target_dir: str = "run_samples/ode_results_time_steps_pytorch/"
    temp_dir: str = "run_samples/temperature_pt/"
    config_csv: str = "run_samples/system_configuration.csv"
    edge_dir: str = "run_samples/edges/"
    checkpoint_path: str = "trained_network/trained_chexa-gnn.pt"
    output_dir: str = "run_samples/predicted/"


@dataclass
class ModelConfig:
    node_dim: int = 128
    edge_dim: int = 128
    global_dim: int = 141
    h_dim_global: int = 256
    h_dim_edge: int = 256
    h_dim_node: int = 512
    h_dim_decoder: int = 1024
    node_num_layers: int = 4
    global_encoding_dim: int = 128


@dataclass
class TrainConfig:
    seed: int = 2
    batch_size: int = 8
    learning_rate: float = 1e-4
    dropout: float = 0.0
    num_epochs: int = 150
    max_steps_per_epoch: int = 2000
    max_val_steps: int = 100
    early_stopping_patience: int = 10
    scheduler_factor: float = 0.5
    scheduler_patience: int = 4
    scheduler_min_lr: float = 1e-8
    spike_threshold: float = 0.10
    safenet_reset_epochs: int = 12
    num_workers: int = 2
