#!/usr/bin/env python3
"""OmniST — one-command reproduction manifest.

Lists every experiment that produces a paper result, in dependency order, with est. runtime. Run the
whole pipeline or a single stage. Batches are resumable (re-running skips completed cells).

Usage:
    python run_all.py --list                 # show the manifest
    python run_all.py --stage B1             # run one stage
    python run_all.py --all                  # run everything in order (long)
    python run_all.py --figures              # just regenerate figures from existing JSONs

Hardware: developed on Apple M4 Pro (MPS), Python 3.14; see requirements.txt (pinned).
Low-load: training batches cap threads to 2 (set OMP_NUM_THREADS/MKL_NUM_THREADS).
"""
import argparse, subprocess, sys, os

HERE = os.path.dirname(os.path.abspath(__file__))

# (stage, script, ~runtime, depends_on, note)
MANIFEST = [
    ("conformal", "experiments/06_inductive_conformal.py", "~5 min", None,
     "headline inductive split-conformal coverage at unseen stations (real CPCB) -> tab:conformal."),
    ("B1", "experiments/B1_multidomain_v2.py", "~30 min", None,
     "3-type cross-domain coverage + dropout on 4 real datasets; writes residual+model cache."),
    ("B2", "experiments/B2_baselines.py", "~10 min", "B1",
     "NN/IDW/meanK/GP interpolation baselines + conformal (reuses B1 cache)."),
    ("B4", "experiments/B4_calibration.py", "~1 min", "B1",
     "calibration curves + sharpness + size-stratified coverage (post-hoc on B1 cache)."),
    ("B3", "experiments/B3_ablations.py", "~40 min", None,
     "ablations: full / no-GAT / k=4 / k=16 on coverage + inductive-MAE (trains variants)."),
    ("B5", "experiments/B5_theorem_check.py", "~3 min", None,
     "empirical verification of the locality Proposition (exact far-change invariance + coverage band)."),
    ("B6", "experiments/B6_estgcn_controlled.py", "~25 min", None,
     "controlled OmniST vs faithful E-STGCN reimpl, identical conditions -> tab:controlled."),
    ("B7", "experiments/B7_largeN_pemsbay.py", "~10 min", None,
     "large-N real inductive coverage at 325 sensors (full PeMS-BAY); needs pemsbay_vel.csv (see data/README)."),
    ("B8", "experiments/B8_fixedtopo_crash_demo.py", "~20 min", None,
     "fixed-topology crash-under-dropout head-to-head (demonstrated runtime failure)."),
    ("B9", "experiments/B9_ignnk_inductive.py", "~25 min", None,
     "OmniST vs learned IGNNK inductive-kriging baseline at unseen nodes -> tab:inductive row."),
    ("temporal", "experiments/10_temporal_conformal.py", "~5 min", None,
     "online-conformal temporal-drift fix (rolling-ACI) on real Delhi -> tab:temporal."),
    ("rigor", "experiments/11_realmultidomain_rigor.py", "~20 min", None,
     "multi-seed / multi-target + conditional coverage on 3 real datasets."),
    ("scalability", "experiments/12_scalability_stress.py", "~30 min", None,
     "cost-scaling ladder toward large N (runtime; honest coverage caveat inside)."),
    ("mondrian", "experiments/13_mondrian_conformal.py", "~35 min", None,
     "Mondrian group-conditional conformal vs marginal on 3 real datasets -> fig:mondrian."),
    ("knn", "experiments/14_approx_knn_test.py", "~1 min", None,
     "scalable cKDTree k-NN equivalence + speed vs exact."),
    ("figures", "experiments/make_figures.py", "<1 min", "B1,B2,B4,temporal,mondrian,knn",
     "regenerate all paper figures from the result JSONs."),
]


def show():
    print(f"{'stage':10s} {'runtime':9s} {'depends':14s} script")
    print("-" * 90)
    for s, script, rt, dep, note in MANIFEST:
        print(f"{s:10s} {rt:9s} {str(dep or '-'):14s} {script}")
        print(f"{'':10s} {'':9s} {'':14s} -> {note}")


def run_stage(stage):
    for s, script, rt, dep, note in MANIFEST:
        if s == stage:
            print(f"[run_all] {stage}: {script} ({rt})", flush=True)
            return subprocess.call([sys.executable, os.path.join(HERE, script)])
    print(f"unknown stage '{stage}'. use --list."); return 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--stage")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--figures", action="store_true")
    a = ap.parse_args()
    if a.list or (not a.stage and not a.all and not a.figures):
        show(); return
    if a.figures:
        run_stage("figures"); return
    if a.stage:
        run_stage(a.stage); return
    if a.all:
        for s, *_ in MANIFEST:
            rc = run_stage(s)
            if rc != 0:
                print(f"[run_all] stage {s} failed (rc={rc}); stopping."); return


if __name__ == "__main__":
    main()
