import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm
import argparse

# Local imports
from models import init_models, put_w2net
from utils import prepare_data, TrajectoryDataset, save_checkpoint, load_checkpoint, extract_indiv_z

# Constants for sequence lengths
SEQ_LEN1 = 60 # Encoder input (long history)
SEQ_LEN2 = 5  # Task solver input (short history)
PRED_LEN = 1  # Prediction horizon

def parse_args():
    parser = argparse.ArgumentParser(description=\"Train Multi-Source Multi-Target Transfer Model\")
    parser.add_argument('--data_dir', type=str, default='data', help='Directory containing the split JSON files')
    parser.add_argument('--model_dir', type=str, default='models', help='Directory to save trained models')
    parser.add_argument('--epochs', type=int, default=1000, help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=1e-5, help='Learning rate')
    parser.add_argument('--z_dim', type=int, default=8, help='Dimensionality of the latent representation')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    
    # Domain configuration
    parser.add_argument('--sources', type=str, nargs='+', default=['DUT', 'DUC', 'INT', 'INC', 'MIT', 'MIC'])
    parser.add_argument('--targets', type=str, nargs='+', default=['DUT', 'DUC', 'INT', 'INC', 'MIT', 'MIC'])
    
    return parser.parse_args()

def load_all_datasets(args):
    \"\"\"Load datasets for all domains and splits.\"\"\"
    datasets = {}
    for split in ['train', 'valid', 'test']:
        prefix = {'train': 'T', 'valid': 'V', 'test': 'E'}[split]
        for i, src in enumerate(args.sources):
            data = prepare_data(args.data_dir, f\"{prefix}{src}M1\", SEQ_LEN1)
            datasets[f'src{i}_{split}'] = TrajectoryDataset(data, SEQ_LEN1)
        for j, tgt in enumerate(args.targets):
            data = prepare_data(args.data_dir, f\"{prefix}{tgt}M2\", SEQ_LEN2, PRED_LEN)
            datasets[f'tgt{j}_{split}'] = TrajectoryDataset(data, SEQ_LEN2)
    return datasets

def get_common_players(datasets, args, split):
    \"\"\"Find players present across all specified domains in a given split.\"\"\"
    common = None
    for i in range(len(args.sources)):
        pids = set(datasets[f'src{i}_{split}'].get_player_ids())
        common = pids if common is None else common & pids
    for j in range(len(args.targets)):
        pids = set(datasets[f'tgt{j}_{split}'].get_player_ids())
        common &= pids
    return list(common) if common else []

def main():
    args = parse_args()
    device = torch.device(args.device)
    os.makedirs(args.model_dir, exist_ok=True)
    
    # 1. Initialize models
    encs, combiner, decs, ts = init_models(
        len(args.sources), len(args.targets), z_dim=args.z_dim, device=device
    )
    
    # 2. Load data
    datasets = load_all_datasets(args)
    pids_train = get_common_players(datasets, args, 'train')
    pids_valid = get_common_players(datasets, args, 'valid')
    print(f\"Common players - Train: {len(pids_train)}, Valid: {len(pids_valid)}\")
    
    # 3. Setup training
    params = list(encs.parameters()) + list(combiner.parameters()) + list(decs.parameters())
    optimizer = optim.Adam(params, lr=args.lr)
    criterion = nn.MSELoss()
    
    best_val_loss = float('inf')
    train_losses, valid_losses = [], []
    
    # 4. Training loop
    for epoch in range(args.epochs):
        encs.train(); combiner.train(); decs.train()
        epoch_loss, total_samples = 0, 0
        
        np.random.shuffle(pids_train)
        for pid in tqdm(pids_train, desc=f\"Epoch {epoch+1}/{args.epochs}\", leave=False):
            # Extract individual signatures from sources
            z_raw_list = []
            valid_p = True
            for i in range(len(args.sources)):
                x, _ = datasets[f'src{i}_train'].get_player_seqs(pid)
                if x is None: {valid_p := False}; break
                # Randomly pick a segment for training diversity
                idx = np.random.randint(0, x.size(0))
                z_raw_list.append(encs[i](x[idx:idx+1].to(device))[0])
            
            if not valid_p: continue
            
            optimizer.zero_grad()
            p_loss, p_samples = 0, 0
            
            # Transfer to each target domain
            for j in range(len(args.targets)):
                target_domain = args.targets[j]
                # Exclude target from sources during transfer
                indices = [idx for idx, name in enumerate(args.sources) if name != target_domain]
                # Randomly drop sources for robustness
                n_drop = np.random.randint(len(indices) + 1)
                for _ in range(n_drop): indices.pop(np.random.randint(len(indices)))
                
                z = combiner([z_raw_list[idx] for idx in indices]) if indices else torch.ones_like(z_raw_list[0])
                
                tx, ty = datasets[f'tgt{j}_train'].get_player_seqs(pid)
                if tx is None: continue
                tx, ty = tx.to(device), ty.to(device)
                
                # Dynamic weight assignment via HyperNetwork
                params_dict = put_w2net(ts, decs[j](z))
                pred_norm = torch.func.functional_call(ts, params_dict, (tx,))
                loss = criterion(pred_norm, ts.normalize(ty))
                
                p_loss += loss * ty.size(0)
                p_samples += ty.size(0)
                
            if p_samples > 0:
                (p_loss / p_samples).backward()
                optimizer.step()
                epoch_loss += p_loss.item()
                total_samples += p_samples
        
        # 5. Validation
        encs.eval(); combiner.eval(); decs.eval()
        v_loss, v_samples = 0, 0
        with torch.no_grad():
            for pid in pids_valid:
                z_list = []
                for i in range(len(args.sources)):
                    x, _ = datasets[f'src{i}_valid'].get_player_seqs(pid)
                    z_list.append(extract_indiv_z(encs[i](x.to(device))))
                
                for j in range(len(args.targets)):
                    indices = [idx for idx, name in enumerate(args.sources) if name != args.targets[j]]
                    z = combiner([z_list[idx] for idx in indices])
                    tx, ty = datasets[f'tgt{j}_valid'].get_player_seqs(pid)
                    if tx is None: continue
                    pred_norm = torch.func.functional_call(ts, put_w2net(ts, decs[j](z)), (tx.to(device),))
                    v_loss += criterion(pred_norm, ts.normalize(ty.to(device))).item() * ty.size(0)
                    v_samples += ty.size(0)
        
        avg_train = epoch_loss / total_samples if total_samples > 0 else 0
        avg_valid = v_loss / v_samples if v_samples > 0 else 0
        train_losses.append(avg_train)
        valid_losses.append(avg_valid)
        
        print(f\"Epoch {epoch+1}: Train Loss={avg_train:.6f}, Valid Loss={avg_valid:.6f}\")
        
        if avg_valid < best_val_loss:
            best_val_loss = avg_valid
            save_checkpoint(encs, combiner, decs, train_losses, valid_losses, os.path.join(args.model_dir, \"best_model.pth\"))

if __name__ == \"__main__\":
    main()
