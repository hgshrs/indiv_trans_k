import os
import json
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from collections import defaultdict

def prepare_data(dataset_dir, file_name, seq_len, pred_len=0):
    \"\"\"
    Load and preprocess trajectory data from a JSON file.
    Filters sequences based on length and removes redundant (stationary) points.
    \"\"\"
    path = os.path.join(dataset_dir, f\"{file_name}.json\")
    final_data = defaultdict(list)

    if not os.path.exists(path):
        print(f\"Warning: {path} not found.\")
        return final_data
        
    with open(path, 'r') as f:
        data_list = json.load(f)
        
    for entry in data_list:
        steamid = entry.get(\"steamid\", \"unknown\")
        seq_df = pd.DataFrame(entry[\"sequence\"])
        
        # Ensure numeric types
        for col in [\"X\", \"Y\", \"Ally_X\", \"Ally_Y\"]:
            seq_df[col] = pd.to_numeric(seq_df[col], errors='coerce')
        
        seq_df = seq_df.dropna(subset=[\"X\", \"Y\", \"Ally_X\", \"Ally_Y\"])
        features = seq_df[[\"X\", \"Y\", \"Ally_X\", \"Ally_Y\"]].values
        
        if len(features) < seq_len + pred_len:
            continue
        
        for i in range(len(features) - seq_len - pred_len):
            past_traj = features[i : i + seq_len]
            if pred_len > 0:
                future_traj = features[i + seq_len + pred_len][np.newaxis, :]
                # Skip if stationary
                if np.all(past_traj[-1, :2] == future_traj[0, :2]):
                    continue
                seq_part = np.concatenate([past_traj, future_traj], axis=0) 
            else:
                seq_part = past_traj
            final_data[steamid].append(seq_part)

    return final_data

class TrajectoryDataset(Dataset):
    \"\"\"
    PyTorch Dataset for handling trajectory sequences.
    \"\"\"
    def __init__(self, data_dict=None, seq_len=5):
        self.data_dict = data_dict if data_dict is not None else {}
        self.seq_len = seq_len
        self.all_sequences = []
        self.player_indices = defaultdict(list)
        
        current_idx = 0
        for steamid, seqs in self.data_dict.items():
            for seq in seqs:
                self.all_sequences.append(seq)
                self.player_indices[steamid].append(current_idx)
                current_idx += 1
        
        self.pids = list(self.player_indices.keys())

    def __len__(self):
        return len(self.all_sequences)

    def __getitem__(self, idx):
        seq = self.all_sequences[idx]
        x = torch.tensor(seq[:self.seq_len], dtype=torch.float32)
        # y is the first point after seq_len (future position)
        y = torch.tensor(seq[self.seq_len, :2], dtype=torch.float32) if seq.shape[0] > self.seq_len else torch.tensor([0.0, 0.0], dtype=torch.float32)
        return x, y

    def get_player_ids(self):
        return list(self.player_indices.keys())

    def get_player_seqs(self, steamid):
        \"\"\"Get all sequences for a specific player as tensors.\"\"\"
        if steamid not in self.data_dict:
            return None, None
        
        seqs = self.data_dict[steamid]
        if not seqs: return None, None
        
        batch_seqs = torch.tensor(np.array(seqs), dtype=torch.float32)
        x = batch_seqs[:, :self.seq_len, :]
        y = batch_seqs[:, self.seq_len, :2] if batch_seqs.shape[1] > self.seq_len else torch.zeros(batch_seqs.size(0), 2)
        return x, y

def load_checkpoint(encs, combiner, decs, model_path, device=torch.device('cpu'), verbose=False):
    \"\"\"Load trained model weights from a checkpoint file.\"\"\"
    try:
        checkpoint = torch.load(model_path, weights_only=False, map_location=device)
        encs.load_state_dict(checkpoint['encs'])
        combiner.load_state_dict(checkpoint['combiner'])
        decs.load_state_dict(checkpoint['decs'])
        train_losses = checkpoint.get('train_losses', [])
        valid_losses = checkpoint.get('valid_losses', [])
        if verbose:
            print(f\"Loaded {os.path.basename(model_path)}.\")
        return train_losses, valid_losses
    except Exception as e:
        if verbose:
            print(f\"No checkpoint found at {model_path} or error: {e}\")
        return [], []

def save_checkpoint(encs, combiner, decs, train_losses, valid_losses, model_path):
    \"\"\"Save current model weights and training history.\"\"\"
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    torch.save({
        'encs': encs.state_dict(),
        'combiner': combiner.state_dict(),
        'decs': decs.state_dict(),
        'train_losses': train_losses,
        'valid_losses': valid_losses
    }, model_path)

def extract_indiv_z(z_seqs, mode='mean'):
    \"\"\"
    Aggregate latent vectors from multiple segments of the same individual.
    Default mode is 'mean' (centroid of the individual's styles).
    \"\"\"
    if mode == 'mean':
        return z_seqs.mean(0)
    elif mode == 'last':
        return z_seqs[-1]
    elif mode == 'random':
        return z_seqs[np.random.randint(0, z_seqs.size(0))]
