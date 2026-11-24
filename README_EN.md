# EEG Cognitive State Decoding for Industrial BCI — 2026 Competition

[中文](README.md) · [Reproduction guide](docs/reproducibility.md) · [Release report](reports/github_release_report.md)

Single-channel EEG decoding at 250 Hz: relax=0, task=1, microvolt inputs and continuous `P(task)` outputs. The repository centers on the final **H1x5** submission and preserves the preliminary R0 LAG-EnergyProto model, adaptive D2, and earlier strict-window and transition studies.

**Local release candidate; not yet published to GitHub.** Owner approval of the code license and contributor attribution is pending. Raw EEG, private labels, formal weights and internal conversations are excluded. Fourteen CPU tests passed in a new isolated environment. Small training/export/inference workflows were executed; full competition experiments were not rerun.

## Methods and support boundaries

H1x5 applies a causal 100 Hz low-pass filter, then combines a six-second query with ten fixed 30-second reference segments. Quality-controlled windows produce 252 features. Four probability branches have weights **0.20 rank tree + 0.10 bandpass/notch tree/LR + 0.35 base HybridNet + 0.35 bandpass/notch HybridNet**. Each tree member has three seeds and each neural member has five. External pretrained weights are not required. See the [architecture and diagram](docs/model_architecture.md).

| Route | Method | Temporal support |
| --- | --- | --- |
| H1x5, final submission | Reference-conditioned tree/LR and HybridNet ensemble | Six-second query; fixed reference and global causal IIR extend support beyond six seconds; prediction order independent |
| R0, preliminary submission | Energy CNN, LightGBM and adaptive prototypes | At most 1500 raw buffered samples; earlier unlabeled normalization/prototype history |
| D2, historical route | Online rank tree and strict tree/LR | At most 1500 raw feature samples; longer adaptation and held-score history |
| legacy | Classical models, CNN/TCN, transfer and transition studies | Strict end-to-end six-second implementations and tests; check each configuration |

Causality and bounded support are separate properties. Record IDs are not verified participant IDs; electrode, device and reliable identity metadata are unavailable.

## Quick start

The target repository URL is pending. After publication use `git clone <confirmed-repository-url>`; for now run from the prepared local checkout. PowerShell:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
$env:PYTHONUTF8 = '1'
$env:OMP_NUM_THREADS = '1'
python -m pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements-dev.txt
python -m pip check
python -m pytest -q
python scripts/h1_synthetic_smoke.py --out cache/h1_demo
python scripts/synthetic_smoke.py --out cache/r0_demo
python scripts/evaluate_predictions.py examples/sample_output/metric_fixture.csv
```

Inspect `cache/h1_demo/report.json` and `cache/r0_demo/predictions.csv`. Synthetic models are interface fixtures, not scientifically evaluated classifiers. Use a new output directory for another training run. Windows CPU execution was validated. Installation succeeded using locally cached wheels after online downloads stalled; a fresh online installation, Linux execution and GitHub Actions remain unverified. See [installation](docs/installation.md) for Linux, GPU and optional transfer dependencies.

## Data, training, evaluation and inference

Obtain authorized data separately; no public download URL is asserted. Layout: `<data_root>/<record_id>/*.bdf` with `*_events.csv`. MNE volts are converted to microvolts; alignment uses the filename's millisecond UTC+08:00 timestamp. Pair complete trial start/end events and retain BAD gaps on the timeline. Route-specific evaluation rules differ; see [dataset](docs/dataset.md).

```bash
python scripts/validate_data.py --data-root /path/to/authorized_data --output cache/data_manifest.json
python scripts/run_route.py --data-root /path/to/authorized_data h1 scripts.finals.build_cache
python scripts/run_route.py --data-root /path/to/authorized_data h1 scripts.finals.build_cache --bpn
```

Follow [training](docs/training.md) for both five-seed, 30-epoch H1x5 neural branches and final export. [Evaluation](docs/evaluation.md) gives five-fold H1 real-interface development commands. The final model fits all 57 records; none of those records is an independent holdout for that fitted model.

H1 exposes `filter_eeg(bdf_path)` and `predict(window, reference)`; set `FIN_MODEL_DIR` to a complete bundle. R0 supports continuous chunks:

```bash
python routes/r0/run_inference.py --input cache/r0_demo/raw_eeg_uv.npy --model cache/r0_demo/model --output cache/r0_demo/replayed.csv --device cpu
```

Honor validity flags: neutral 0.5 is not reliable evidence of task. [Inference](docs/inference.md) documents loading, reference input, missing-data fallback, reset behavior and state management.

## Historical results, not rerun for this release

| Model and protocol | AUC | bACC | ACC |
| --- | ---: | ---: | ---: |
| D2, 46 records, five folds, mean of 20 orders, pooled | 0.700424 | 0.647805 | — |
| H1, 46 records, five folds, real interface, record mean | 0.710315 | 0.652764 | 0.652445 |
| H2, same development protocol as H1 | 0.709913 | 0.652543 | 0.652303 |

Do not rank different protocols against one another. H1/H2 average reference seeds 14/15/16 within each record. Final-stage score is `(AUC+ACC)/2`, whereas the preliminary score uses `(AUC+bACC)/2`. H1 development models are distinct from the all-record H1x5 weights. No official hidden-test score is claimed. [Experiments](docs/experiments.md) retains baselines, failed promotion checks, transition ablations, D2's worst order and the limitations of already-exposed historical lockbox results.

## Layout and evidence

`routes/h1`, `routes/r0`, `routes/d2` and `routes/legacy` preserve isolated research namespaces. `scripts` provides launchers and synthetic/data/model/safety checks; `tests` contains CPU acceptance tests; `docs` contains detailed guides; `results` stores historical aggregates and sanitized execution evidence; `checkpoints` stores hashes only; `reports` stores the audit, source mapping and release report. `.github/workflows/ci.yml` defines Windows/Linux synthetic CPU checks.

Executed: independent venv installation and `pip check`, 14 tests, synthetic-input checks of actual H1x5 weights, read-only format/hash checks of 57 H1 records, and R0 training/export/reload on one real record. A fresh local Git clone without runtime caches also passed all 14 tests in 35.61 seconds. The R0 real-data run used the existing project environment; independent-environment tests used synthetic CPU workflows. Not executed: complete CV/search, new lockbox evaluation, hidden testing, new-environment GPU training, Linux or Actions. See [reproducibility](docs/reproducibility.md) for code/workflow/result distinctions.

## Weights, citation and licensing

The [weight inventory](checkpoints/README.md) records size and SHA-256 for 36 model/config artifacts without distributing them. Train from authorized data or obtain permission from the rights holder; verify the complete bundle.

[LICENSE](LICENSE) is explicitly pending and grants no project-wide open-source license yet. MIT is proposed subject to owner and team rights confirmation. Original BIOT/SingLEM MIT notices are retained; weight permissions are separate. See [third-party notices](THIRD_PARTY_NOTICES.md). Competition data and organizer documents are not relicensed here.

[CITATION.cff](CITATION.cff) is a software citation draft with contributor names, URL and formal version awaiting confirmation. No paper, DOI or competition ranking is invented. We acknowledge the organizers, team contributors and upstream open-source projects.
