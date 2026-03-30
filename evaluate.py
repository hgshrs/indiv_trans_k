import os
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import argparse
from scipy import stats
import pandas as pd
from tqdm import tqdm

# Local imports
from models import init_models, put_w2net
from utils import prepare_data, TrajectoryDataset, load_checkpoint, extract_indiv_z

# Constants
SEQ_LEN1 = 60
SEQ_LEN2 = 5
PRED_LEN = 1

def parse_args():
    parser = argparse.ArgumentParser(description=\"Evaluate Multi-Source Multi-Target Transfer Model\")
    parser.add_argument('--data_dir', type=str, default='data', help='Directory containing the split JSON files')
    parser.add_argument('--model_path', type=str, required=True, help='Path to the best_model.pth')
    parser.add_argument('--z_dim', type=int, default=8)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    
    # Domain configuration
    parser.add_argument('--sources', type=str, nargs='+', default=['DUT', 'DUC', 'INT', 'INC', 'MIT', 'MIC'])
    parser.add_argument('--targets', type=str, nargs='+', default=['DUT', 'DUC', 'INT', 'INC', 'MIT', 'MIC'])
    
    return parser.parse_args()

def main():
    args = parse_args()
    device = torch.device(args.device)
    
    # 1. Initialize and Load Model
    encs, combiner, decs, ts = init_models(
        len(args.sources), len(args.targets), z_dim=args.z_dim, device=device
    )
    load_checkpoint(encs, combiner, decs, args.model_path, device=device, verbose=True)
    encs.eval(); combiner.eval(); decs.eval()
    
    # 2. Load Test Data
    datasets = {}
    for i, src in enumerate(args.sources):
        data = prepare_data(args.data_dir, f\"E{src}M1\", SEQ_LEN1)
        datasets[f'src{i}'] = TrajectoryDataset(data, SEQ_LEN1)
    for j, tgt in enumerate(args.targets):
        data = prepare_data(args.data_dir, f\"E{tgt}M2\", SEQ_LEN2, PRED_LEN)
        datasets[f'tgt{j}'] = TrajectoryDataset(data, SEQ_LEN2)
        
    # Find common players in test set
    common_pids = None
    for i in range(len(args.sources)):
        pids = set(datasets[f'src{i}'].get_player_ids())
        common_pids = pids if common_pids is None else common_pids & pids
    for j in range(len(args.targets)):
        pids = set(datasets[f'tgt{j}'].get_player_ids())
        common_pids &= pids
    common_pids = list(common_pids) if common_pids else []
    print(f\"Common players in Test set: {len(common_pids)}\")
    
    # 3. Evaluation
    criterion = nn.MSELoss(reduction='none')
    results = []
    
    with torch.no_grad():
        for pid in tqdm(common_pids, desc=\"Evaluating players\"):
            # Extract z from sources
            z_raw_list = []
            for i in range(len(args.sources)):
                x, _ = datasets[f'src{i}'].get_player_seqs(pid)
                z_raw_list.append(extract_indiv_z(encs[i](x.to(device))))
            
            for j in range(len(args.targets)):
                target_domain = args.targets[j]
                
                # Baseline: Stay (Persistence)
                tx, ty = datasets[f'tgt{j}'].get_player_seqs(pid)
                if tx is None: continue
                tx, ty = tx.to(device), ty.to(device)
                
                loss_stay = criterion(tx[:, -1, :2], ty).mean().item()
                
                # Proposed: Transfer
                indices = [idx for idx, name in enumerate(args.sources) if name != target_domain]
                z = combiner([z_raw_list[idx] for idx in indices])
                
                params_dict = put_w2net(ts, decs[j](z))
                pred_norm = torch.func.functional_call(ts, params_dict, (tx,))
                pred_raw = ts.denormalize(pred_norm)
                loss_transfer = criterion(pred_raw, ty).mean().item()
                
                results.append({
                    'Player': pid, 'Domain': target_domain,
                    'Stay_RMSE': np.sqrt(loss_stay),
                    'Transfer_RMSE': np.sqrt(loss_transfer)
                })
                
    # 4. Statistical Analysis
    df = pd.DataFrame(results)
    summary = df.groupby('Domain')[['Stay_RMSE', 'Transfer_RMSE']].mean()
    print(\"\\nEvaluation Summary (Mean RMSE):\")
    print(summary)
    
    # Paired t-test
    t_stat, p_val = stats.ttest_rel(df['Stay_RMSE'], df['Transfer_RMSE'])
    print(f\"\\nGlobal Paired t-test (Stay vs Transfer): t={t_stat:.4f}, p={p_val:.4e}\")
    
    # 5. Visualization (Simple scatter)
    plt.figure(figsize=(6, 6))
    plt.scatter(df['Stay_RMSE'], df['Transfer_RMSE'], alpha=0.5)
    max_val = max(df['Stay_RMSE'].max(), df['Transfer_RMSE'].max())
    plt.plot([0, max_val], [0, max_val], 'k--', alpha=0.7)
    plt.xlabel('Stay RMSE'); plt.ylabel('Transfer RMSE')
    plt.title('Prediction Error Comparison')
    plt.grid(True)
    plt.savefig('evaluation_scatter.png')
    print(\"Visualization saved to evaluation_scatter.png\")

if __name__ == \"__main__\":
    main()
