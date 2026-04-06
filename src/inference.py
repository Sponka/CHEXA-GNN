from pathlib import Path

import torch
from torch_geometric.loader import DataLoader

from models.gnn_model import GNNModel
from .dataset import AtmosphereGraphDataset, load_configuration, load_split_ids


def load_trained_model(checkpoint_path, model_cfg):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = GNNModel(
        node_dim=model_cfg.node_dim,
        edge_dim=model_cfg.edge_dim,
        global_dim=model_cfg.global_dim,
        global_hidden_dim=model_cfg.h_dim_global,
        edge_hidden_dim=model_cfg.h_dim_edge,
        node_hidden_dim=model_cfg.h_dim_node,
        decoder_hidden_dim=model_cfg.h_dim_decoder,
        global_encoding_dim=model_cfg.global_encoding_dim,
        num_layers=model_cfg.node_num_layers
    ).to(device)

    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()
    return model, device


def run_preprocessed_samples(run_cfg, model_cfg, batch_size=8):
    sample_ids = load_split_ids(run_cfg.split_csv)
    configuration = load_configuration(run_cfg.config_csv)

    target_dir = run_cfg.target_dir if Path(run_cfg.target_dir).exists() else None

    dataset = AtmosphereGraphDataset(
        node_dir=run_cfg.node_dir,
        target_dir=target_dir,
        temp_dir=run_cfg.temp_dir,
        config_df=configuration,
        edge_dir=run_cfg.edge_dir,
        filenames=sample_ids
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
        persistent_workers=True
    )

    model, device = load_trained_model(run_cfg.checkpoint_path, model_cfg)

    output_dir = Path(run_cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model.eval()
    with torch.no_grad():
        for batch_idx, batch in enumerate(loader, start=1):
            batch = batch.to(device)

            pred = model(batch.x, batch.edge_index, batch.edge_attr, batch.u, batch.batch)
            pred_cpu = pred.detach().cpu()

            batch_mapping = batch.batch.cpu().numpy()
            for i in range(batch.num_graphs):
                global_idx = (batch_idx - 1) * batch_size + i
                if global_idx >= len(sample_ids):
                    break

                filename = sample_ids[global_idx]
                node_mask = (batch_mapping == i)
                pred_single = pred_cpu[node_mask]

                save_path = output_dir / f"{filename}_pred.pt"
                torch.save(pred_single, save_path)

    return output_dir
