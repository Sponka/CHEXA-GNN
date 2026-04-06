import os
from typing import Optional

import pandas as pd
import torch
from torch_geometric.data import Data, Dataset

from .constants import ATMOSPHERE_INFO_COLUMNS


class AtmosphereGraphDataset(Dataset):
    """PyTorch Geometric dataset for atmospheric GNN training and prediction."""
    def __init__(self, node_dir, temp_dir, config_df, edge_dir, filenames, target_dir: Optional[str] = None):
        super().__init__()
        self.node_dir = node_dir
        self.target_dir = target_dir
        self.temp_dir = temp_dir
        self.edge_dir = edge_dir
        self.filenames = filenames
        self.config_df = config_df
        self.atmosphere_info = ATMOSPHERE_INFO_COLUMNS

    def len(self):
        return len(self.filenames)

    def get(self, idx):
        filename = self.filenames[idx]
        file_id = filename.split('.')[0]

        x = torch.load(os.path.join(self.node_dir, file_id + '.pt'), weights_only=True).float()

        y = None
        if self.target_dir is not None:
            y = torch.load(os.path.join(self.target_dir, file_id + '.pt'), weights_only=True).float()

        temp = torch.load(os.path.join(self.temp_dir, file_id + '.pt'), weights_only=True).float()

        edge_path = os.path.join(self.edge_dir, 'edges_' + file_id + '.pt')
        if not os.path.exists(edge_path):
            raise FileNotFoundError(f"Edge file not found for sample {file_id}")
        edge_index, edge_attr = torch.load(edge_path, weights_only=True)
        edge_index = edge_index.long()
        edge_attr = edge_attr.float()

        try:
            row = self.config_df.loc[file_id]
        except KeyError:
            raise ValueError(f"Missing configuration for sample {file_id} (KeyError)")

        globals_vec = row[self.atmosphere_info].to_numpy().flatten()
        globals_tensor = torch.tensor(globals_vec, dtype=torch.float32)

        u = torch.cat([temp.view(-1), globals_tensor.view(-1)], dim=0).float().unsqueeze(0)

        data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, u=u)
        if y is not None:
            data.y = y
        return data


def load_split_ids(csv_path: str, column: str = "System_ID") -> list[str]:
    df = pd.read_csv(csv_path)
    return df[column].astype(str).tolist()


def load_configuration(config_csv: str) -> pd.DataFrame:
    df = pd.read_csv(config_csv)
    return df.set_index("System_ID")
