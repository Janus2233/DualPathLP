# DualPathLP

Code and datasets for the paper *"Dynamic Network Link Prediction Based on a Dual-Route Graph Neural Network"* (基于双路线图神经网络的动态网络链路预测).

This repository contains the model, the data layer (preprocessing / temporal splitting / negative sampling / metrics), and the raw files of four public datasets — everything needed to reproduce the main results (Table 3) and the ablation studies.

## Repository structure

```
├── src/
│   ├── models/          # node route, subgraph route, gated fusion, time encoding, route calibration
│   │   ├── dual_route.py        # dual-route gated fusion
│   │   ├── node_route.py        # node route (GRU memory + temporal graph attention)
│   │   ├── subgraph_route.py    # subgraph route (DRNL labeling + weighted GCN + multiplicative readout)
│   │   ├── time_encoding.py     # cosine time encoding
│   │   └── route_calibration.py # optional route-score calibration
│   ├── graph/           # causal enclosing-subgraph construction, candidate-pair causal features
│   └── train/
│       ├── m1_dual_route.py     # main training / evaluation entry (fusion and single-route ablations)
│       └── m0_subgraph_only.py  # subgraph-route-only entry
├── data/                # data layer: loading, preprocessing, temporal split, inductive holdout, negatives, AP/AUC
│   └── checks/          # data-layer self-checks (integrity / splits / negatives / metrics / preprocessing)
├── dataset/             # raw files of the four datasets (public SNAP data)
└── pyproject.toml       # data-layer package config (dglp-data)
```

## Installation

Python ≥ 3.11.

```bash
pip install -r requirements.txt
```

- GPU is optional: the paper experiments ran on a single 6 GB GPU; CPU works but is slower (`--device cpu`).
- For GPU use, install a CUDA build of torch matching your driver first ([pytorch.org](https://pytorch.org)), then `torch_geometric` (≥ 2.6; GCNConv uses native torch scatter, so `torch_scatter` / `torch_sparse` are not required).
- Tested environment: Python 3.13, torch 2.9.0+cu128, torch_geometric 2.7.0, numpy 2.4.4, pandas 3.0.2, scikit-learn 1.9.0.

## Datasets

`dataset/` ships the raw files of four public continuous-time dynamic graphs (source: SNAP):

| Registry name | File | Description |
|---|---|---|
| `college_msg` | CollegeMsg.txt | online messages between UC Irvine students |
| `email_eu_core` | email-Eu-core-temporal.txt | internal emails of a European research institution |
| `reddit_hyperlinks` | soc-redditHyperlinks-body.txt | cross-subreddit hyperlinks |
| `sx_mathoverflow` | sx-mathoverflow.txt | Stack Exchange math Q&A |

SHA256 checksums and row counts are registered in `data/config.py`. Evaluation protocol: chronological 70%/15%/15% split (time quantiles 0.70 / 0.85); an additional 10% of test-period nodes are held out as inductive new nodes (seed 2020) and all their edges are removed from training; 1:1 endpoint-replacement negative sampling for training, fixed full-graph negatives for evaluation; batches of 200 positive + 200 negative candidates; AP/AUC computed per batch and averaged across batches. Event and node features are 172-dimensional zero vectors on all four datasets.

## Data preprocessing (required on a fresh clone)

```bash
python -m data.build --all                    # all four datasets
python -m data.build --dataset college_msg    # or a single dataset
```

The build performs SHA256 verification, self-loop removal, timestamp sorting, splitting and inductive holdout. Outputs are written to `runs/data/` (not tracked by git).

Data-layer self-checks:

```bash
python -m data.checks.run_all
```

## Reproducing the main table (Table 3)

Flagship configuration (CollegeMsg, seed 42, deterministic mode):

```bash
python -m src.train.m1_dual_route \
  --dataset college_msg --route fused --pair-features --input struct \
  --hop 3 --window-frac 0 --sub-layers 3 --lr 0.005 --epochs 20 \
  --seed 42 --deterministic --tag repro_college_s42
```

The other three datasets differ only in subgraph neighborhood and training schedule (all other hyperparameters are defaults, as above):

```bash
python -m src.train.m1_dual_route --dataset email_eu_core     --route fused --pair-features --hop 2 --window-frac 0.05 --sub-layers 3 --lr 0.003 --epochs 15 --seed 42 --deterministic --tag repro_email_s42
python -m src.train.m1_dual_route --dataset reddit_hyperlinks --route fused --pair-features --hop 2 --window-frac 0.05 --sub-layers 3 --lr 0.003 --epochs 15 --seed 42 --deterministic --tag repro_reddit_s42
python -m src.train.m1_dual_route --dataset sx_mathoverflow   --route fused --pair-features --hop 2 --window-frac 0.05 --sub-layers 3 --lr 0.003 --epochs 15 --seed 42 --deterministic --tag repro_sx_s42
```

- Eight-seed results: run with `--seed 42` through `49` and report mean ± sample standard deviation of `test.fused.average_precision`.
- Single-route controls (Table 4): keep all other flags, use `--route node` or `--route sub`.
- Results are written to `runs/<tag>/summary.json` (`test.fused / test.node / test.sub`, each with `average_precision` and `roc_auc`); add `--gate-stats` to also emit gate statistics (`gate_stats.json`).
- For CPU runs add `--device cpu`.

Expected main-table AP (eight-seed mean ± sample std):

| Dataset | DualPathLP (gated fusion) |
|---|---|
| CollegeMsg | 0.9657 ± 0.0006 |
| email-Eu-core | 0.9858 ± 0.0001 |
| RedditHyperlinks | 0.9753 ± 0.0087 |
| sx-mathoverflow | 0.9813 ± 0.0046 |

## Ablations and controls

Add the following flags to the flagship command:

| Experiment | Flag |
|---|---|
| Causal statistics → zero input (Table 5) | `--input zeros` |
| Remove candidate-pair causal features (Table 5) | drop `--pair-features` |
| Per-dimension ablation of pair features | `--pair-dims aa,jac` (any subset) |
| Remove staleness from gate context (Table 5) | `--ctx-no-dt` |
| Node decoder without staleness (Table 5 control) | `--no-decoder-dt` |
| Constant-gate control (Table 5 / Fig. 4b) | `--const-gate` |
| Disable subgraph edge-time gating (§4.4.2) | `--no-edge-gate` |
| Cross-route consistency regularizer (§4.4.2) | `--lambda-cons 0.05` (default 0 = off) |
| Strict negative-sampling audit (Table 6) | `--strict-negatives` |

## Implementation notes (frozen caliber reported in the paper — keep defaults)

The following three points are the frozen implementation caliber reported in Section 5 of the paper. Default flags match it; changing them changes all frozen results:

1. Node-route GRU memory updates run under a no-grad path (318,888 of the 606,316 reported parameters are not trained); the fix flag `--memory-grad onestep` is off by default.
2. The first two dimensions of the gate context (subgraph size, edge-to-node ratio) are filled with batch-level quantities; the fix flag `--ctx-per-candidate` is off by default.
3. The subgraph edge table registers one directed edge per interaction; graph convolution propagates along that direction only.

## Citation

*"Dynamic Network Link Prediction Based on a Dual-Route Graph Neural Network"* (DualPathLP).