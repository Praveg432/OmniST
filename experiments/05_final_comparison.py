#!/usr/bin/env python3
"""Final 5-seed comparison: OmniST(POT) vs OmniST(MSE) vs E-STGCN."""
import sys, os, torch, numpy as np, time, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scipy import stats
from src.data import DataConfig, prepare_delhi_data
from src.omnist_model import OmniST, ModelConfig
from src.baselines import ESTGCN
from src.trainer import TrainConfig, train_model, evaluate_model

N_SEEDS = 5
data_config = DataConfig(window_size=60, horizon=30, n_features=3)
cfg_pot = TrainConfig(lr=1e-3, n_epochs=35, patience=10, batch_size=8, loss_type='pot')
cfg_mse = TrainConfig(lr=1e-3, n_epochs=35, patience=10, batch_size=8, loss_type='mse', use_evt_loss=False)

all_results = []
for seed in range(N_SEEDS):
    print(f'Seed {seed+1}/{N_SEEDS}...', end='', flush=True)
    t0 = time.time()
    train_ds, val_ds, test_ds, meta = prepare_delhi_data(data_config, seed=seed)
    res = {}
    
    torch.manual_seed(seed)
    m1 = OmniST(ModelConfig(n_features=3, horizon=30, d_model=128, k_neighbors=8))
    train_model(m1, train_ds, val_ds, cfg_pot, meta)
    res['OmniST_POT'] = evaluate_model(m1, test_ds, meta)

    torch.manual_seed(seed)
    m2 = OmniST(ModelConfig(n_features=3, horizon=30, d_model=128, k_neighbors=8))
    train_model(m2, train_ds, val_ds, cfg_mse, meta)
    res['OmniST_MSE'] = evaluate_model(m2, test_ds, meta)

    torch.manual_seed(seed)
    sf, sc, _, _ = train_ds[0]
    e = ESTGCN(n_features=3, d_model=128, horizon=30, n_nodes=37, station_coords=sc)
    train_model(e, train_ds, val_ds, cfg_pot, meta)
    res['ESTGCN'] = evaluate_model(e, test_ds, meta)

    all_results.append(res)
    print(f' {time.time()-t0:.0f}s')

# Print results
print('\n' + '='*60)
models = ['OmniST_POT', 'OmniST_MSE', 'ESTGCN']
print(f'{"Model":<16} {"MAE":<16} {"RMSE":<16} {"ExtRecall":<14}')
for m in models:
    maes = [r[m]['mae'] for r in all_results]
    rmses = [r[m]['rmse'] for r in all_results]
    recs = [r[m]['extreme_recall'] for r in all_results]
    print(f'{m:<16} {np.mean(maes):.2f}+/-{np.std(maes):.2f}   {np.mean(rmses):.2f}+/-{np.std(rmses):.2f}   {np.mean(recs):.3f}+/-{np.std(recs):.3f}')

# Paired tests
for metric in ['mae', 'rmse', 'extreme_recall']:
    o = [r['OmniST_POT'][metric] for r in all_results]
    e = [r['ESTGCN'][metric] for r in all_results]
    _, p = stats.ttest_rel(o, e)
    d = (np.mean(o)-np.mean(e))/(np.std(np.array(o)-np.array(e))+1e-8)
    print(f'POT vs ESTGCN [{metric}]: p={p:.4f} d={d:.2f}')

with open(os.path.join(os.path.dirname(__file__), '..', 'results', 'exp_final_5seed.json'), 'w') as f:
    json.dump(all_results, f, indent=2, default=float)
print('Saved.')
