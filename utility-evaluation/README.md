# Utility Evaluation

Cleaned, runnable implementation of the fidelity and downstream-task
utility metrics used to check whether NetShare/NetSSM-style synthetic
network traffic is still *useful*: does it statistically resemble the real
data it was trained on, and can a model trained on it substitute for real
training data on a downstream classification task?

This is the counterpart to
[`../privacy-evaluation`](../privacy-evaluation): a generator can score well
on privacy (little leakage) and poorly on utility (unusable synthetic data),
or vice versa. In particular,
[`../privacy-evaluation/network-attack/sensitive-network-properties`](../privacy-evaluation/network-attack/sensitive-network-properties)
scores the *same kind* of distributional-closeness idea as an attack (lower
divergence = more leakage), while this folder scores it as a benefit (lower
divergence = more fidelity). Don't compare numbers across the two folders
without relabeling.

## Attribution



- The repeated JSD/EMD-over-categorical-fields pattern → "Fidelity: shared
  distribution-divergence core" section (`jensen_shannon_divergence()`,
  `normalized_earth_movers_distance()`, `compare_categorical_distributions()`)
- `fidelity_NetShare()` → "Fidelity: NetShare" section (`fidelity-netshare` subcommand)
- `fidelity_NetSSM()` → "Fidelity: NetSSM" section (`fidelity-netssm` subcommand)
- The live (uncommented) nprint-flattening path in `downstream_task_performance.py`
  (`prepare_data()`/`prepare_data_real()`) and the unused flow-statistics
  helpers it defined but never called (`prepare_real_stats()`/
  `prepare_synthetic_stats()`) → "Downstream task" section, exposed as two
  selectable `--feature_mode` options (`nprint`, `flow_stats`) rather than
  one live path plus dead code
- The commented-out CatBoost block and the actively-used XGBoost classifier
  → `make_classifier()`'s `--classifier` choices (`xgboost`, `catboost`,
  plus `random_forest` for a scikit-learn-only fallback that needs no extra
  packages)

No functional change to the fidelity or downstream-task math itself.

## Fidelity: categorical-field distribution divergence

For a set of categorical fields (source/destination IP, source/destination
port, protocol, ...), compares the real and generated value distributions
with two divergence measures:

- **Jensen-Shannon divergence** (`scipy.spatial.distance.jensenshannon`): a
  bounded, symmetric divergence between two probability distributions.
- **Normalized Earth Mover's Distance**: `sum(|cumsum(p) - cumsum(q)|) / (n - 1)`,
  computed over the two value-count distributions after aligning them to a
  shared, lexicographically-sorted category index. This treats sorted
  category *rank* as if it were a numeric axis — a different (and
  order-dependent) computation from the `scipy.stats.wasserstein_distance`
  used in
  [`../privacy-evaluation/.../sensitive-network-properties`](../privacy-evaluation/network-attack/sensitive-network-properties),
  which operates on genuinely numeric fields. Kept faithful to the original
  for continuity with prior results.

**⚠️ Lower is better here** (0 = identical distributions, i.e. maximal
fidelity) — the opposite direction from the same-shaped metric used as an
attack score in
[`../privacy-evaluation/.../sensitive-network-properties`](../privacy-evaluation/network-attack/sensitive-network-properties).

## Downstream task: TRTR vs. TSTR classification accuracy

Trains a classifier on **real** training data, evaluates on real test data
(TRTR: Train-Real-Test-Real, an upper-bound reference), then trains an
identical classifier on **synthetic** data and evaluates on the same real
test data (TSTR: Train-Synthetic-Test-Real). TSTR accuracy close to TRTR
means the synthetic data preserves enough task-relevant signal to
substitute for real training data.

Two input feature representations are supported via `--feature_mode`:

| Mode | Features | Source |
|---|---|---|
| `nprint` (default, matches the original's live code path) | First 3 packets' nprint bit-columns per flow, flattened into one row | `.nprint` files |
| `flow_stats` | Per-flow summary stats (packet/byte counts, IAT stats, rates) | `_flows.csv` files |

## Usage

```bash
# Fidelity: NetShare (tabular flow CSVs, columns srcip/dstip/srcport/dstport/proto)
python utility_evaluation.py fidelity-netshare \
  --original_glob "/path/to/<dataset>/train/*_merged/pre_processed_data/*.csv" \
  --generated_glob "/path/to/<dataset>/train/*_merged/generated_data/*/*.csv" \
  --dataset <dataset>

# Fidelity: NetSSM (per-packet CSVs, columns Src/Dst IP/Port, Protocol)
python utility_evaluation.py fidelity-netssm \
  --original_packets_glob "/path/to/data_MIA_new/single_<dataset>/new_data/train/*_packets.csv" \
  --generated_packets_glob "/path/to/inference/<dataset>_singleflow_.../*/singleprompt_*_packets.csv" \
  --dataset <dataset>

# Downstream task: TRTR vs. TSTR classification accuracy
python utility_evaluation.py downstream-task \
  --train_real_folder "/path/to/data_MIA_new/single_<dataset>/new_data/train/" \
  --test_real_folder "/path/to/data_MIA_new/single_<dataset>/new_data/test/" \
  --synthetic_glob "/path/to/inference/<dataset>_singleflow_.../*/singleprompt_*.nprint" \
  --feature_mode nprint --classifier xgboost --model_name NetSSM --dataset <dataset>
```

Run `python utility_evaluation.py <command> --help` for the full flag list
of any subcommand.

## Dependencies

`pandas`, `numpy`, `scipy`, `scikit-learn` are required for all subcommands.
`downstream-task` additionally needs `xgboost` or `catboost` installed if
you pass `--classifier xgboost` (the default) or `--classifier catboost`;
`--classifier random_forest` needs only scikit-learn.
