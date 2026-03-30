import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import argparse
import itertools
from tqdm import tqdm

# Local imports
from models import init_models
from utils import prepare_data, TrajectoryDataset, load_checkpoint, extract_indiv_z

# Constants
SEQ_LEN1 = 60

def parse_args():
    parser = argparse.ArgumentParser(description=\"Analyze Latent Space Dimensionality\")
    parser.add_argument('--data_dir', type=str, default='data')
    parser.add_argument('--model_dir', type=str, required=True, help='Directory containing checkpoints for different z_dim')
    parser.add_argument('--z_dims', type=int, nargs='+', default=[2, 4, 8])
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--sources', type=str, nargs='+', default=['DUT', 'DUC', 'INT', 'INC', 'MIT', 'MIC'])
    
    return parser.parse_args()

def main():
    args = parse_args()
    device = torch.device(args.device)
    
    # Load validation datasets for sources to extract z
    datasets = {}
    for i, src in enumerate(args.sources):
        data = prepare_data(args.data_dir, f\"V{src}M1\", SEQ_LEN1)
        datasets[f'src{i}'] = TrajectoryDataset(data, SEQ_LEN1)
        
    common_pids = None
    for i in range(len(args.sources)):
        pids = set(datasets[f'src{i}'].get_player_ids())
        common_pids = pids if common_pids is None else common_pids & pids
    common_pids = list(common_pids)
    
    for z_dim in args.z_dims:
        print(f\"\\nAnalyzing z_dim = {z_dim}\")
        # Initialize model components (only encs and combiner needed for z extraction)
        encs, combiner, _, _ = init_models(len(args.sources), z_dim=z_dim, device=device)
        
        # Note: Expects model filenames to contain z_dim info if following the training script conventions
        # For simplicity in this public script, we assume a path is provided or searchable
        model_path = os.path.join(args.model_dir, f\"model_z{z_dim}.pth\") 
        if not os.path.exists(model_path):
            print(f\"Skipping z_dim={z_dim}, model not found at {model_path}\")
            continue
            
        load_checkpoint(encs, combiner, torch.nn.ModuleList([]), model_path, device=device)
        encs.eval(); combiner.eval()
        
        all_z = []
        with torch.no_grad():
            for pid in common_pids:
                z_list = []
                for i in range(len(args.sources)):
                    x, _ = datasets[f'src{i}'].get_player_seqs(pid)
                    z_list.append(extract_indiv_z(encs[i](x.to(device))))
                z_p = combiner(z_list).cpu().numpy()
                all_z.append(z_p)
        
        all_z = np.array(all_z) # (N, z_dim)
        
        # Plot first two dimensions
        if z_dim >= 2:
            plt.figure(figsize=(6, 6))
            plt.scatter(all_z[:, 0], all_z[:, 1], alpha=0.6)
            plt.xlabel('z[0]'); plt.ylabel('z[1]')
            plt.title(f'Latent Distribution (z_dim={z_dim})')
            plt.grid(True)
            plt.savefig(f'latent_z{z_dim}.png')
            print(f\"Latent plot saved to latent_z{z_dim}.png\")

if __name__ == \"__main__\":
    main()
