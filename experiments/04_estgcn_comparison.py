#!/usr/bin/env python3
"""
Experiment 04: E-STGCN Comparison — Core Paper Result

10-seed paired comparison between OmniST and E-STGCN baseline.
Statistical testing: paired t-test, Cohen's d, win count.

This produces the paper's main results table.
"""
import sys, os, time, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
from scipy import stats

from src.data import DataConfig, prepare_delhi_data
from src.omnist_model import OmniST, ModelConfig
from src.baselines import ESTGCN, LSTMOnlyBaseline
from src.trainer import TrainConfig, train_model, evaluate_model, evaluate_topology_robustness


N_SEEDS = 5
DATA_CONFIG = DataConfig(window_size=60, horizon=30, n_features=3, normalize=True)
TRAIN_CONFIG = TrainConfig(lr=1e-3, n_epochs=40, patience=10, batch_size=8)


def run_single_seed(seed: int, verbose: bool = False):
    """Run all models on same data split for one seed."""
    # Prepare data (same split for all models at this seed)
    train_ds, val_ds, test_ds, metadata = prepare_delhi_data(DATA_CONFIG, seed=seed)

    results = {}

    # --- OmniST ---
    torch.manual_seed(seed)
    model_config = ModelConfig(n_features=3, horizon=30, d_model=128, k_neighbors=8)
    omnist = OmniST(model_config)
    history = train_model(omnist, train_ds, val_ds, TRAIN_CONFIG, metadata, verbose=verbose)
    metrics = evaluate_model(omnist, test_ds, metadata)
    topo = evaluate_topology_robustness(omnist, test_ds, metadata)
    results["OmniST"] = {**metrics, "epochs": history["epochs_trained"]}
    results["OmniST"]["topo"] = {
        k: np.mean(v) for k, v in topo.items()
    }

    # --- E-STGCN ---
    torch.manual_seed(seed)
    estgcn = ESTGCN(n_features=3, d_model=128, horizon=30, n_nodes=37)
    history = train_model(estgcn, train_ds, val_ds, TRAIN_CONFIG, metadata, verbose=verbose)
    metrics = evaluate_model(estgcn, test_ds, metadata)
    # E-STGCN topology test: it CANNOT handle variable N — document this
    try:
        topo_e = evaluate_topology_robustness(estgcn, test_ds, metadata)
        results["E-STGCN"] = {**metrics, "epochs": history["epochs_trained"]}
        results["E-STGCN"]["topo"] = {k: np.mean(v) for k, v in topo_e.items()}
    except Exception as e:
        results["E-STGCN"] = {**metrics, "epochs": history["epochs_trained"]}
        results["E-STGCN"]["topo"] = {"error": str(e)}

    # --- LSTM-only ---
    torch.manual_seed(seed)
    lstm = LSTMOnlyBaseline(n_features=3, d_model=128, horizon=30)
    history = train_model(lstm, train_ds, val_ds, TRAIN_CONFIG, metadata, verbose=verbose)
    metrics = evaluate_model(lstm, test_ds, metadata)
    results["LSTM-only"] = {**metrics, "epochs": history["epochs_trained"]}

    return results


def main():
    print("=" * 70)
    print("  Experiment 04: E-STGCN Comparison (10-seed paired)")
    print("=" * 70)
    t_start = time.time()

    all_results = []
    for seed in range(N_SEEDS):
        print(f"\n  Seed {seed+1}/{N_SEEDS}...", end="", flush=True)
        t0 = time.time()
        res = run_single_seed(seed, verbose=False)
        elapsed = time.time() - t0
        print(f" done ({elapsed:.0f}s)")
        all_results.append(res)

    total_time = time.time() - t_start
    print(f"\n  Total time: {total_time:.0f}s = {total_time/60:.1f}min")

    # Aggregate results
    models = ["OmniST", "E-STGCN", "LSTM-only"]
    metrics = ["mae", "rmse", "smape", "extreme_recall", "mae_10d", "mae_30d"]

    print("\n" + "=" * 70)
    print("  RESULTS")
    print("=" * 70)

    # Summary table
    print(f"\n{'Model':<16}", end="")
    for m in ["MAE", "RMSE", "SMAPE(%)", "ExtRecall"]:
        print(f"{m:<14}", end="")
    print()
    print("-" * 70)

    model_metrics = {}
    for model_name in models:
        values = {m: [] for m in metrics}
        for res in all_results:
            if model_name in res:
                for m in metrics:
                    if m in res[model_name]:
                        values[m].append(res[model_name][m])
        model_metrics[model_name] = values

        mae_str = f"{np.mean(values['mae']):.2f}±{np.std(values['mae']):.2f}"
        rmse_str = f"{np.mean(values['rmse']):.2f}±{np.std(values['rmse']):.2f}"
        smape_str = f"{np.mean(values['smape']):.1f}±{np.std(values['smape']):.1f}"
        recall_str = f"{np.mean(values['extreme_recall']):.3f}±{np.std(values['extreme_recall']):.3f}"
        print(f"{model_name:<16}{mae_str:<14}{rmse_str:<14}{smape_str:<14}{recall_str:<14}")

    # Statistical tests: OmniST vs E-STGCN
    print("\n" + "-" * 70)
    print("  Statistical Comparison: OmniST vs E-STGCN (paired)")
    print("-" * 70)

    for metric in ["mae", "rmse", "extreme_recall"]:
        omnist_vals = model_metrics["OmniST"][metric]
        estgcn_vals = model_metrics["E-STGCN"][metric]

        if len(omnist_vals) == len(estgcn_vals) and len(omnist_vals) > 1:
            t_stat, p_val = stats.ttest_rel(omnist_vals, estgcn_vals)
            # Cohen's d for paired samples
            diffs = np.array(omnist_vals) - np.array(estgcn_vals)
            cohens_d = diffs.mean() / (diffs.std() + 1e-8)
            # Win count
            if metric == "extreme_recall":
                wins = sum(1 for o, e in zip(omnist_vals, estgcn_vals) if o > e)
            else:
                wins = sum(1 for o, e in zip(omnist_vals, estgcn_vals) if o < e)

            direction = "lower is better" if metric != "extreme_recall" else "higher is better"
            print(f"  {metric:<18} p={p_val:.4f}  Cohen's d={cohens_d:.3f}  Wins={wins}/{N_SEEDS}  ({direction})")

    # Topology robustness comparison
    print("\n" + "-" * 70)
    print("  Topology Robustness (30% node dropout)")
    print("-" * 70)

    for model_name in models:
        topo_results = []
        full_results = []
        for res in all_results:
            if model_name in res and "topo" in res[model_name]:
                topo = res[model_name]["topo"]
                if isinstance(topo, dict) and "error" not in topo:
                    if "drop_30pct" in topo and "full" in topo:
                        full_results.append(topo["full"])
                        topo_results.append(topo["drop_30pct"])

        if full_results and topo_results:
            full_mean = np.mean(full_results)
            drop_mean = np.mean(topo_results)
            degradation = (drop_mean - full_mean) / full_mean * 100
            print(f"  {model_name:<16} Full={full_mean:.4f}  Drop30%={drop_mean:.4f}  Degradation={degradation:+.1f}%")
        else:
            print(f"  {model_name:<16} CANNOT handle topology change (structural limitation)")

    # Save raw results
    output_path = os.path.join(os.path.dirname(__file__), "..", "results", "exp04_comparison.json")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Convert numpy values for JSON serialization
    serializable = []
    for res in all_results:
        s = {}
        for model_name, metrics_dict in res.items():
            s[model_name] = {k: (float(v) if isinstance(v, (np.floating, float)) else v)
                            for k, v in metrics_dict.items()}
        serializable.append(s)

    with open(output_path, "w") as f:
        json.dump(serializable, f, indent=2, default=str)
    print(f"\n  Results saved to: {output_path}")


if __name__ == "__main__":
    main()
