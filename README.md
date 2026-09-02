# FYPothesis

FYPothesis is an autonomous machine-learning research agent for recommender
systems, built for TikTok TechJam 2026 Track 2. Given the KuaiRand-Pure dataset,
it forms hypotheses, writes complete experiment scripts, trains and evaluates
them, reflects on the evidence, recovers from failures, and decides whether to
confirm, ensemble, pivot, or stop.

The project is designed around one principle: a higher validation score is not
enough by itself. FYPothesis records how each result was obtained and prevents a
single lucky seed, validation-selected ensemble, or test-label leak from becoming
the submitted result.

## Result

The final model was evaluated once on the hidden test split after the
configuration and 16 ensemble seeds were fixed.

| Split | System | GAUC | nDCG@5 | Primary |
|---|---|---:|---:|---:|
| Validation | Official FM baseline | 0.6674 | 0.5357 | 0.6016 |
| Validation | FYPothesis | **0.67212** | **0.53870** | **0.60541** |
| Validation | Absolute improvement | **+0.00472** | **+0.00300** | **+0.00381** |
| Hidden test | Official FM baseline | 0.6610 | 0.5282 | 0.5946 |
| Hidden test | FYPothesis | **0.66510** | **0.53110** | **0.59810** |
| Hidden test | Absolute improvement | **+0.0041** | **+0.0029** | **+0.0035** |

The submitted model is a rank-normalized mean of all 16 predeclared seeds of one
configuration. No seed, subset, or blend weight was selected using validation
performance. The complete result record is in
[`results/final_results.json`](results/final_results.json), and the one-time
evaluation proof is in
[`results/final_evaluation.lock`](results/final_evaluation.lock).

## Problem definition

FYPothesis follows the task definition pinned by the organizer's Starter Kit:

- Benchmark: KuaiRand-Pure.
- Task: rank each user's logged impressions.
- Positive label: `long_view`.
- Metrics: GAUC and nDCG@5.
- Primary score: `mean(GAUC, nDCG@5)`.
- Training data: the fixed 8–21 April 2022 train window only.
- Development feedback: the fixed 22–28 April validation window.
- Hidden test: the fixed 29 April–8 May window, evaluated once for submission.
- Hard limits: 50 iterations and six hours per benchmark run.
- Convergence: the organizers' published rule, `epsilon=0.002` and `N=3`.

The older challenge-summary line mentioning click, NDCG@10, and Recall@50
conflicts with the detailed benchmark definition and executable Starter Kit.
FYPothesis uses the Starter Kit's authoritative `long_view`, GAUC, and nDCG@5
implementation without changing it.

## How FYPothesis works

Each outer-loop iteration follows the same evidence-producing workflow:

1. Read the append-only research history and current evidence state.
2. Inspect train/validation data through bounded, read-only tools.
3. Propose a measurable hypothesis and expected effect before execution.
4. Generate a complete Python experiment rather than an untracked code fragment.
5. Run syntax, capability, configuration, and leakage checks.
6. Execute the experiment in a subprocess whose test outcome columns are
   mechanically removed.
7. Score validation predictions with the unchanged organizer evaluator.
8. Record the hypothesis, code diff, metrics, resource use, and recovery events.
9. Repair, pivot, confirm across paired seeds, ensemble, or stop.

The search space covers ranking losses, negative sampling, user-history models,
multi-task learning, model families, temporal features, training schedules,
sample weighting, and regularization. Leakage-sensitive data options are hidden
from the model and rejected by the validator unless a human deliberately enables
and records an override.

### Evidence policy

- A single-seed gain is `PRELIMINARY` and cannot change the submission.
- Promotion requires paired multi-seed confirmation with both arms fixed first.
- The final ensemble keeps every predeclared seed from one configuration.
- Failed or malformed runs are journaled but cannot become evidence.
- The validation-best eligible checkpoint is selected when the official
  convergence rule fires.

### Final configuration

| Component | Choice |
|---|---|
| Model | NumPy factorization machine |
| Loss | BPR pairwise ranking loss |
| Negative sampling | Uniform, one negative per positive |
| User history | Recency-weighted positive-history pooling |
| Temporal context | Hour of day and day of week |
| Training | Lower learning rate with a longer schedule |
| Ensemble | Rank-normalized mean over seeds 0–15 |

## Autonomy, robustness, and resource use

The recorded competition run converged at node 8 and selected node 4 as the
officially eligible validation-best checkpoint.

| Measure | Recorded value |
|---|---:|
| Outer-loop iterations | 9 of 50 |
| Training executions | 27 |
| Manual interventions | 0 |
| LLM calls | 17 |
| LLM tokens | 203,602 |
| LLM spend | USD 0.718218 |
| Agent wall-clock | 42.8 minutes |
| GPU hours | 0.0 |

FYPothesis classifies syntax, runtime, timeout, artifact, and evaluation failures
by consequence. It can repair a broken implementation, skip an invalid result,
or pivot to a cheaper experiment without promoting a failed candidate. The
fault-injection evidence is summarized in
[`results/JUDGE_PACKET.md`](results/JUDGE_PACKET.md).

## Repository structure

```text
FYPothesis/
├── run_agent.py                 # autonomous-run entry point and configuration
├── app.py                       # live-run and submission-proof dashboard
├── agent/                       # planning, execution, evidence, recovery, reports
├── runtime/                     # model training, data boundaries, research tools
├── config/                      # search space, model defaults, budget settings
├── kuairand-starter-kit/        # organizer data loader, evaluator, and baseline
├── tests/test_harness.py        # deterministic safety and orchestration checks
├── logs/                        # required run journal, diffs, metrics, and scripts
├── results/                     # final result, manifest, lock, and judge packet
└── RESULTS.md                   # generated results and resource summary
```

## Setup and installation

The project was verified with Python 3.12.10. Training is CPU-capable; a GPU is
not required for the submitted NumPy model.

```bash
git clone https://github.com/Khantminn290/FYPothesis.git
cd FYPothesis

python3.12 -m venv .venv
source .venv/bin/activate
python3 -m pip install --upgrade pip
python3 -m pip install -r requirements.txt
```

On Windows PowerShell, activate the environment with `.\.venv\Scripts\Activate.ps1`.

Download KuaiRand-Pure into the Starter Kit directory. The archive is ignored by
Git and is not redistributed in this repository.

```bash
cd kuairand-starter-kit
curl -L -o KuaiRand-Pure.tar.gz \
  https://zenodo.org/records/10439422/files/KuaiRand-Pure.tar.gz
tar -xzf KuaiRand-Pure.tar.gz
cd ..
```

An API key is needed only for a new autonomous research run. It is not needed to
rebuild the fixed final model, verify stored metadata, or use the dashboard.

```bash
cp .env.example .env
# Edit .env and add either OPENAI_API_KEY or ANTHROPIC_API_KEY.
```

`.env`, datasets, caches, generated predictions, and submission CSVs are ignored
and must never be committed.

## Reproduce the results

Run every command below from the repository root after installing the dataset.

### 1. Reproduce the official baseline

```bash
python3 -m agent.baseline_repro
```

The baseline command stores the organizer-script hashes and reproduced metrics
under `logs/baseline/`.

### 2. Rebuild and verify the submitted validation model

Prediction arrays are intentionally not committed. Rebuild all 16 fixed members,
then recompute the ensemble with the unchanged evaluator:

```bash
python3 -m agent.final_ensemble --seeds 16
python3 -m agent.verify_incumbent
```

Expected validation result:

```text
GAUC 0.67212 | nDCG@5 0.53870 | primary 0.60541
```

### 3. Run the verification and recovery suites

```bash
python3 tests/test_harness.py
python3 -m agent.recovery_eval
```

The harness makes no LLM calls. It is run after rebuilding the ensemble because
the generated prediction arrays are intentionally excluded from Git. Some
executor-boundary tests also load the local KuaiRand cache, so the dataset must
be installed first. The deterministic recovery evaluation also makes no API
call; it exercises runtime-error, malformed-artifact, and timeout recovery
through the real loop and executor.

### 4. Rebuild and validate submission files

```bash
python3 -m agent.make_submission \
  --split valid --out submission_valid.csv --score --ensemble

python3 -m agent.make_submission \
  --split test --out submission_test.csv --ensemble

python3 kuairand-starter-kit/submit.py \
  --data_dir kuairand-starter-kit/KuaiRand-Pure/data \
  --check --split test submission_test.csv
```

These commands build the CSV files from the prediction arrays created in step 2
and validate schema/alignment. They do not repeat the hidden-test evaluation.
The one allowed evaluation has already been recorded, and `--final-test-eval`
now refuses to run because the lock exists.

### 5. Regenerate submission documents

```bash
python3 -m agent.iteration_log
python3 -m agent.manifest --run-tests
python3 -m agent.judge_packet
python3 -m agent.results_report --run-tests
```

### 6. Open the proof dashboard

```bash
streamlit run app.py
```

The dashboard renders the committed manifest, journal, recovery evidence, final
result, and downloadable submission documents. It can also launch a new live
research run only when a local operator explicitly enables that capability.
By default it starts in public showcase mode: all paid-run controls are disabled,
while the complete branching **Demo run** remains available.
The one-time hidden-test evaluation is never exposed as a dashboard button.
For recording or judging walkthroughs, the **Demo run** tab replays a preloaded
branching journal after the judge presses **Start demo**. It demonstrates
observation, hypothesis selection, preflight rejection, recovery, evaluation,
paired confirmation, ensembling, and stopping without starting training or
making an API call.

To enable dashboard-launched runs in a trusted local environment only:

```bash
FYPOTHESIS_ENABLE_REAL_RUNS=1 streamlit run app.py
```

For Streamlit Community Cloud, deploy branch `main` with entrypoint
`dashboard/streamlit_app.py` and Python 3.12. Leave the Secrets field empty—the
deployment entrypoint forcibly keeps paid execution disabled and uses the
lightweight dependencies in `dashboard/requirements.txt`.

### 7. Start a new autonomous competition run

This is optional, uses the configured LLM API, and creates a new research run:

```bash
python3 run_agent.py --competition --fresh
```

The resolved model, spend limit, iteration limit, training-run limit, and
wall-clock limit are printed before the first paid call.

## Submission artifacts

| Deliverable | Location |
|---|---|
| Project overview and reproduction guide | [`README.md`](README.md) |
| Required per-iteration log | [`logs/ITERATION_LOG.md`](logs/ITERATION_LOG.md) |
| Machine-readable journal | [`logs/journal.jsonl`](logs/journal.jsonl) |
| Generated experiment scripts | [`logs/solutions/`](logs/solutions/) |
| Per-iteration code diffs | [`logs/diffs/`](logs/diffs/) |
| Results and resource usage | [`RESULTS.md`](RESULTS.md) |
| Judge evidence packet | [`results/JUDGE_PACKET.md`](results/JUDGE_PACKET.md) |
| Machine-readable final result | [`results/final_results.json`](results/final_results.json) |
| Canonical evidence manifest | [`results/manifest.json`](results/manifest.json) |
| One-time evaluation proof | [`results/final_evaluation.lock`](results/final_evaluation.lock) |

## Compliance safeguards

- Only KuaiRand-Pure is used; there is no external training data or pretrained
  weight trained on benchmark labels.
- Generated experiments receive train and validation data plus a test view with
  all outcome columns removed.
- The raw dataset and real cache are inaccessible to the generated subprocess
  during execution.
- Static leakage checks reject target-derived feature construction before
  training.
- Validation is the only feedback used for search, early stopping, and model
  selection.
- The Starter Kit evaluator remains unchanged, and its hash is recorded.
- Every iteration records its hypothesis, full script, code diff, metrics, token
  use, wall-clock time, and error/recovery events.
- The final-evaluation lock records the submission hash, timestamp, and one-shot
  metrics.

## Limitations and future work

- The hidden-test gain is real but modest (`+0.0035` primary), and the benchmark
  has narrow headroom relative to seed noise.
- The final ensemble reduces variance within one FM/BPR configuration; it does
  not obtain additional diversity from multiple competitive model families.
- Autonomous feature discovery did not produce a confirmed improvement in the
  submitted run.
- KuaiRand-1k and KuaiRand-27k bonus benchmarks were not attempted.
- Search is sequential by default. Safe parallel exploration could reduce
  wall-clock time, but would require stricter shared-budget coordination.
- A future version should improve independent hypothesis discovery, add broader
  paired ablations, and scale the data cache for the bonus datasets.

## Team member contributions

The five team members contributed across the research, engineering, evaluation,
and presentation work:

- **Kaung Khant Minn** led the overall architecture and integration of
  FYPothesis. He connected the autonomous agent loop, evidence and convergence
  system, final competition runs, dashboard, documentation, and submission
  verification into one reproducible project, and coordinated the final release.
- **Min Wai Phyo** contributed to experiment design and research policy,
  including hypothesis selection, the modification search space, confirmation
  strategy, and interpretation of recommendation-model results.
- **Samuel Christy George** contributed to the training and data pipeline,
  including model execution, Starter Kit evaluator integration, reproducibility,
  and validation of the final ensemble workflow.
- **Mani Kumar Prateek** contributed to safety and reliability, including data
  boundaries, leakage protection, preflight and output-contract checks, failure
  classification, and recovery testing.
- **Bill Sujith Kumaar** contributed to the Streamlit interface and submission
  presentation, including the experiment tree, run-detail views, result
  summaries, judge-facing usability, and final quality assurance.
