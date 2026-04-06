import os
from datetime import datetime

import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.loader import DataLoader

from models.gnn_model import GNNModel
from .utils import generate_version_name, make_generator, seed_worker, set_seed


def create_dataloaders(train_dataset, validation_dataset, cfg):
    generator = make_generator(cfg.seed)

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        persistent_workers=cfg.num_workers > 0,
        pin_memory=True,
        worker_init_fn=seed_worker,
        generator=generator
    )

    validation_loader = DataLoader(
        validation_dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        pin_memory=True,
        worker_init_fn=seed_worker,
        generator=generator
    )

    return train_loader, validation_loader


def train_single_run(model_cfg, train_cfg, train_dataset, validation_dataset, code_name=None):
    set_seed(train_cfg.seed)

    version_name = generate_version_name()
    model_version = f"GNN_{version_name}"
    os.makedirs(version_name, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = GNNModel(
        node_dim=model_cfg.node_dim,
        edge_dim=model_cfg.edge_dim,
        global_dim=model_cfg.global_dim,
        global_hidden_dim=model_cfg.h_dim_global,
        global_encoding_dim=model_cfg.global_encoding_dim,
        edge_hidden_dim=model_cfg.h_dim_edge,
        node_hidden_dim=model_cfg.h_dim_node,
        decoder_hidden_dim=model_cfg.h_dim_decoder,
        num_layers=model_cfg.node_num_layers,
        dropout_p=train_cfg.dropout
    ).to(device)

    param_count = sum(p.numel() for p in model.parameters())

    criterion = nn.L1Loss()
    optimizer = optim.Adam(model.parameters(), lr=train_cfg.learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=train_cfg.scheduler_factor,
        patience=train_cfg.scheduler_patience, min_lr=train_cfg.scheduler_min_lr
    )

    train_loader, validation_loader = create_dataloaders(train_dataset, validation_dataset, train_cfg)

    csv_log_path = f"{version_name}/training_log.csv"
    log_df = pd.DataFrame(columns=["epoch", "train_loss", "val_loss", "lr"])
    best_val_loss = float("inf")
    checkpoint_path = f"{version_name}/{model_version}.pt"

    early_stopping_patience = train_cfg.early_stopping_patience
    epochs_no_improve = 0
    safenet_reset_counter = 0
    avg_train_loss = float('nan')
    avg_val_loss = float('nan')
    epoch = 0

    for epoch in range(train_cfg.num_epochs):
        model.train()
        total_train_loss = 0.0
        num_batches = len(train_loader)

        for batch_idx, batch in enumerate(train_loader, start=1):
            if batch_idx > train_cfg.max_steps_per_epoch:
                break

            batch = batch.to(device)
            optimizer.zero_grad()

            pred = model(batch.x, batch.edge_index, batch.edge_attr, batch.u, batch.batch)
            loss = criterion(pred, batch.y)
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()

        avg_train_loss = total_train_loss / min(num_batches, train_cfg.max_steps_per_epoch)

        model.eval()
        total_val_loss = 0.0
        num_val_batches_run = 0

        with torch.no_grad():
            for batch_idx_val, batch in enumerate(validation_loader, start=1):
                if batch_idx_val > train_cfg.max_val_steps:
                    break
                if batch is None:
                    continue

                batch = batch.to(device)
                pred = model(batch.x, batch.edge_index, batch.edge_attr, batch.u, batch.batch)
                val_loss = criterion(pred, batch.y)

                if not torch.isnan(val_loss):
                    total_val_loss += val_loss.item()
                    num_val_batches_run += 1

        if num_val_batches_run == 0:
            avg_val_loss = float('inf')
        else:
            avg_val_loss = total_val_loss / num_val_batches_run

        current_lr = optimizer.param_groups[0]["lr"]

        spike_detected = (
            avg_train_loss > train_cfg.spike_threshold or
            avg_val_loss > train_cfg.spike_threshold
        )

        if spike_detected:
            epochs_no_improve = 0
            safenet_reset_counter = train_cfg.safenet_reset_epochs

        scheduler.step(avg_val_loss)

        if avg_val_loss < best_val_loss - 1e-5:
            best_val_loss = avg_val_loss
            epochs_no_improve = 0
            safenet_reset_counter = 0
            torch.save(model.state_dict(), checkpoint_path)
        else:
            if safenet_reset_counter > 0:
                safenet_reset_counter -= 1
            else:
                epochs_no_improve += 1

        log_df.loc[len(log_df)] = [epoch + 1, avg_train_loss, avg_val_loss, current_lr]
        log_df.to_csv(csv_log_path, index=False)

        if epoch >= 95 and epochs_no_improve >= early_stopping_patience:
            break

    experiment_row = {
        "experiment_id": model_version,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "seed": train_cfg.seed,
        "node_dim": model_cfg.node_dim,
        "edge_dim": model_cfg.edge_dim,
        "global_dim": model_cfg.global_dim,
        "hidden_dim_global": model_cfg.h_dim_global,
        "hidden_dim_edge": model_cfg.h_dim_edge,
        "hidden_dim_node": model_cfg.h_dim_node,
        "hidden_dim_decoder": model_cfg.h_dim_decoder,
        "global_encoding_dim": model_cfg.global_encoding_dim,
        "num_layers": model_cfg.node_num_layers,
        "train_loss_final": avg_train_loss,
        "val_loss_final": avg_val_loss,
        "best_val_loss": best_val_loss,
        "param_count": param_count,
        "epochs": epoch + 1,
        "code": code_name
    }

    summary_csv = "experiments_summary_final_node.csv"
    if not os.path.exists(summary_csv):
        pd.DataFrame(columns=experiment_row.keys()).to_csv(summary_csv, index=False)

    df_summary = pd.read_csv(summary_csv)
    df_summary.loc[len(df_summary)] = experiment_row
    df_summary.to_csv(summary_csv, index=False)

    return {
        "version_name": version_name,
        "model_version": model_version,
        "checkpoint_path": checkpoint_path,
        "best_val_loss": best_val_loss,
    }
