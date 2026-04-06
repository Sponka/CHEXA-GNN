import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import DataPaths, ModelConfig, TrainConfig
from src.dataset import AtmosphereGraphDataset, load_configuration, load_split_ids
from src.training import train_single_run
from src.utils import configure_determinism


def parse_args():
    parser = argparse.ArgumentParser(description="Train CHEXA-GNN.")

    parser.add_argument("--h_dim_global", type=int, default=256)
    parser.add_argument("--h_dim_edge", type=int, default=256)
    parser.add_argument("--h_dim_node", type=int, default=512)
    parser.add_argument("--h_dim_decoder", type=int, default=1024)
    parser.add_argument("--node_num_layers", type=int, default=4)
    parser.add_argument("--global_encoding_dim", type=int, default=128)
    parser.add_argument("--seed", type=int, default=2)
    parser.add_argument("--code", type=str, default="final_training_architecture.py")

    parser.add_argument("--train_split_csv", type=str, default="data/train_split.csv")
    parser.add_argument("--val_split_csv", type=str, default="data/val_split.csv")
    parser.add_argument("--node_dir", type=str, default="data/mix_profiles_pytorch/")
    parser.add_argument("--target_dir", type=str, default="data/ode_results_time_steps_pytorch/")
    parser.add_argument("--temp_dir", type=str, default="data/temperature_pt/")
    parser.add_argument("--config_csv", type=str, default="data/configuration_normalised/system_configuration.csv")
    parser.add_argument("--edge_dir", type=str, default="data/edges/")

    return parser.parse_args()


def main():
    configure_determinism()
    args = parse_args()

    data_paths = DataPaths(
        train_split_csv=args.train_split_csv,
        val_split_csv=args.val_split_csv,
        node_dir=args.node_dir,
        target_dir=args.target_dir,
        temp_dir=args.temp_dir,
        config_csv=args.config_csv,
        edge_dir=args.edge_dir,
    )

    model_cfg = ModelConfig(
        h_dim_global=args.h_dim_global,
        h_dim_edge=args.h_dim_edge,
        h_dim_node=args.h_dim_node,
        h_dim_decoder=args.h_dim_decoder,
        node_num_layers=args.node_num_layers,
        global_encoding_dim=args.global_encoding_dim,
    )
    train_cfg = TrainConfig(seed=args.seed)

    train_ids = load_split_ids(data_paths.train_split_csv)
    val_ids = load_split_ids(data_paths.val_split_csv)
    configuration = load_configuration(data_paths.config_csv)

    train_dataset = AtmosphereGraphDataset(
        node_dir=data_paths.node_dir,
        target_dir=data_paths.target_dir,
        temp_dir=data_paths.temp_dir,
        config_df=configuration,
        edge_dir=data_paths.edge_dir,
        filenames=train_ids
    )
    validation_dataset = AtmosphereGraphDataset(
        node_dir=data_paths.node_dir,
        target_dir=data_paths.target_dir,
        temp_dir=data_paths.temp_dir,
        config_df=configuration,
        edge_dir=data_paths.edge_dir,
        filenames=val_ids
    )

    train_single_run(
        model_cfg=model_cfg,
        train_cfg=train_cfg,
        train_dataset=train_dataset,
        validation_dataset=validation_dataset,
        code_name=args.code
    )


if __name__ == "__main__":
    main()
