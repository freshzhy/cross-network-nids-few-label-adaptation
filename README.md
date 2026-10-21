# Attack-Specific Feature Signatures for Cross-Network Intrusion Detection: From Model Transfer to Coverage-Aware Few-Label Adaptation

Code and reference outputs for the paper of the same title (manuscript under review).

The study compares four kinds of knowledge that can be carried from a source network to a new target network — attack-specific SHAP signatures (RQ1), source-selected feature subsets retrained on the target (RQ2), source-trained models deployed unchanged (RQ3) and few-label target adaptation with a variant-held-out stress test (RQ4) — on four NetFlow v2 (NFv2) datasets that share a 40-feature schema. A deployment-cost analysis compares Full-40 with ten-feature models.

## Repository layout

```
.
├── run.sh                       single entry point for the whole pipeline
├── requirements.txt
├── code/
│   ├── preprocess.py            NFv2 preprocessing (Section 3.1.1)
│   ├── rq1_signatures.py        RQ1: one-vs-rest XGBoost + TreeSHAP signatures
│   ├── rq1_distinctiveness.py   RQ1: intra-dataset distinctiveness (Table III, Fig. S4)
│   ├── rq1_stability.py         RQ1: signature stability across 30 seeds (Section 4.1.3)
│   ├── rq1_dedup_robustness.py  RQ1 signatures on exact-deduplicated data (Table S6)
│   ├── rq2_similarity.py        RQ1: cross-network Jaccard / Spearman (Table IV, Fig. 3)
│   ├── rq3_diagnostics.py       raw class counts, duplicate and IP-group diagnostics
│   ├── rq3_transfer.py          RQ2/RQ3: feature-set and model portability, statistics
│   ├── rq3_jsd.py               Jensen–Shannon divergence as a predictor (Fig. 5)
│   ├── rq4_label_budget.py      RQ4: target-label budget, deduplicated replication, post hoc budgets
│   ├── rq4_variant_holdout.py   RQ4: leave-one-tool-out and attack-family swap stress test
│   ├── deployment_cost.py       deployment cost and NetFlow field requirements (Tables S10–S11)
│   ├── plot_framework.py        Fig. 1 and the graphical abstract
│   ├── plot_figures_rq1_rq2.py  Figs. 2, 3, S1, S3, S5
│   ├── plot_figures_rq3.py      Figs. 4–6, S2, S6
│   └── plot_figures_rq4.py      Figs. 7 and 8
├── data/
│   ├── raw/                     place the four NFv2 CSV files here (not redistributed)
│   └── processed/               written by the preprocessing step
└── results/                     reference outputs behind every table and figure
```

Script names keep the original analysis numbering: `rq2_similarity.py` produces the cross-network signature comparison that the paper reports under RQ1, and `rq3_transfer.py` produces both the retrained feature-transfer conditions (RQ2) and the zero-shot conditions (RQ3).

## Setup

The results were produced with **Python 3.12 and XGBoost 3.2.0** (`tree_method = hist`) on macOS (Apple Silicon).

```bash
./run.sh setup                   # creates .venv and installs requirements.txt
```

Download NF-UNSW-NB15-v2, NF-CSE-CIC-IDS2018-v2, NF-ToN-IoT-v2 and NF-BoT-IoT-v2 from https://staff.itee.uq.edu.au/marius/NIDS_datasets/ and save them as `data/raw/<dataset>.csv`.

## Reproducing the results

```bash
./run.sh preprocess              # data/raw  -> data/processed
./run.sh rq1                     # signatures, distinctiveness, stability
./run.sh rq2                     # cross-network signature similarity
./run.sh rq3                     # diagnostics, feature-set / zero-shot transfer (30 seeds), JSD
./run.sh robustness              # deduplication, selection protocol, scenario-feature ablation
./run.sh rq4                     # label budget (original, deduplicated, post hoc), variant hold-out
./run.sh cost                    # deployment cost and field requirements
./run.sh figures                 # results/figures/
```

`./run.sh all` runs every step in order. The RQ3, robustness and RQ4 steps take several hours; run them in the background with `./run.sh bg <step>`, check progress with `./run.sh status`, and stop with `kill $(cat results/logs/<step>.pid)`. Restarting a step resumes from the completed tasks. On macOS the background job keeps the machine awake with `caffeinate`.

Parallelism is set with the `WORKERS` and `THREADS` environment variables (default: `(cores − 2)/4` parallel tasks with 4 XGBoost threads each), e.g. `WORKERS=4 THREADS=4 ./run.sh bg rq4`. The cost step times every model single-threaded and runs one task at a time by default (`WORKERS=1`). Every script documents its options under `python code/<script>.py --help`.

`rq4_label_budget.py` has the subcommands `run`, `status`, `aggregate`, `dupdiag` and `compare`; `--dedup` selects the exact-deduplicated data and `run --posthoc` adds the post hoc budgets of one, two and five labeled flows.

## Where each reported result comes from

| Paper item | File(s) in `results/` |
|---|---|
| Fig. 1, graphical abstract | `figures/Fig1_framework.*`, `figures/Graphical_abstract.*` |
| Table I (dataset characteristics) | `preprocess_logs/*.log`, `rq3_diagnostics/raw_class_counts.json` |
| Table S1, Fig. S3 (per-class performance) | `rq1_signatures/*_rq1_model_metrics.csv` |
| Fig. 2, Fig. S1 (signatures) | `rq1_signatures/*_rq1_shap_matrix.csv`, `*_rq1_top10_signatures.csv` |
| Table III, Fig. S4 (intra-dataset distinctiveness) | `rq1_distinctiveness/summary.csv`, `*_distinctiveness_matrix.csv` |
| Section 4.1.3 (stability across seeds) | `rq1_stability/signature_stability_summary.csv` |
| Table IV, Fig. 3, Fig. S5 (cross-network similarity) | `rq2_similarity/rq2_transferability_report.csv`, `rq2_jaccard_*.csv`, `rq2_spearman_*.csv` |
| Table V-a (within-dataset compression) | `rq3_main/config_summary.csv` (conditions Full-40, Target-10) |
| Table V-b, Section 4.2.2 (configuration-level tests) | `rq3_main/config_level_tests.csv` |
| Table S2, Figs. 4 and S2 (per-configuration results) | `rq3_main/config_summary.csv`, `per_config_tests.csv`, `raw.csv` |
| Fig. S6 (environment), Fig. 5 (predictors) | `rq3_main/per_config_tests.csv`, `rq3_jsd/jsd_matrix.csv`, `rq2_similarity/` |
| Table S7, Fig. 6 (zero-shot transfer) | `rq3_main/config_summary.csv` (`*-Direct` conditions), `per_config_tests.csv` |
| Table VI, Fig. 7, Tables S8a–c (label budget) | `rq4_budget/curve_by_config.csv`, `rq4_budget_dedup/curve_by_config.csv`, `*/zero_shot_baseline.csv`, `*/posthoc_vs_zero_shot.csv` |
| Table S8d (H1–H3), Table S8e (crossover budget) | `rq4_budget*/hypotheses.csv`, `rq4_budget*/crossover.csv`, `rq4_budget*/summary.txt` |
| Table S8f (duplicate diagnostic) | `rq4_budget/dup_diag.csv` |
| Fig. S7 (original vs deduplicated), Fig. S8 (per configuration) | `rq4_budget_dedup/compare_orig_vs_dedup.*`, `rq4_budget/fig_budget_per_config_f1.*` |
| Fig. 8, Table S9 (variant-held-out stress test) | `rq4_variant/summary_by_test.csv`, `rq4_variant/verdicts.csv`, `rq4_variant/summary.txt` |
| Tables S10–S11, Section 5.4 (deployment cost, field requirements) | `deployment_cost/summary_by_config.csv`, `fields_by_config.csv`, `field_levels.csv`, `summary.txt` |
| Table S3 (deduplicated data) | `rq3_dedup/config_level_tests.csv` |
| Table S4 (rows retained after deduplication) | `rq3_dedup/dedup_report.csv` |
| Table S5 (selection protocol, scenario features) | `rq3_trainshap/`, `rq3_dedup_reselect/`, `rq3_noscenario/` |
| Table S6, Section 4.5 (signatures on deduplicated data) | `rq1_dedup_robustness/signature_dedup.csv`, `intra_distinctiveness.csv`, `cross_dataset.csv` |
| All figures | `figures/Fig*.pdf` / `.png` |

Each `summary.txt` is a human-readable digest of the corresponding experiment. In the result tables, `std` is the 1:10 test set and `nat` the prevalence-matched test set.

## Notes on reproducibility

- Seeds 42–71 are used for the RQ2/RQ3 repeated splits; robustness analyses and RQ4 use seeds 42–51; the deployment-cost analysis uses seeds 42–46.
- RQ4 reuses the data caches, splits and test sets of `rq3_transfer.py`; at the full label budget it reproduces the RQ2 results.
- The one-, two- and five-label budgets of RQ4 were added after the main results and are post hoc exploratory analyses.
- XGBoost fits use row and column subsampling, so a different thread configuration can change individual fits slightly; aggregate conclusions are unaffected. Timings in `deployment_cost/` depend on the machine.
- Large caches (`results/**/cache/`, per-task `parts/`) and the processed data are excluded from the repository and are regenerated by the pipeline.
