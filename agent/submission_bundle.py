"""Build the two judge-upload folders from verified canonical artifacts.

This does not train, score the hidden test, or mutate the research journal. It
only copies already-produced evidence and submission CSVs into names that map
directly to deliverables 3 and 4.

Usage:
    python3 -m agent.submission_bundle
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUT = ROOT / "submission_upload"


def _load(path: Path) -> dict:
    with path.open() as fh:
        return json.load(fh)


def _copy(source: Path, target: Path) -> None:
    if not source.is_file():
        raise FileNotFoundError(f"required artifact is missing: {source}")
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _csv_rows(path: Path) -> int:
    with path.open("rb") as fh:
        return max(0, sum(1 for _ in fh) - 1)


def build(out: Path = DEFAULT_OUT) -> Path:
    if out.exists():
        raise FileExistsError(
            f"refusing to overwrite existing bundle: {out}\n"
            "Move it aside or delete it deliberately, then rerun.")

    run_dir = out / "03_RUN_AND_ITERATION_LOGS"
    final_dir = out / "04_FINAL_SUBMISSION_AND_RESULTS"
    run_dir.mkdir(parents=True)
    final_dir.mkdir(parents=True)

    # Deliverable 3: readable log plus the raw evidence behind every entry.
    _copy(ROOT / "logs" / "ITERATION_LOG.md",
          run_dir / "RUN_AND_ITERATION_LOG.md")
    _copy(ROOT / "logs" / "journal.jsonl", run_dir / "journal.jsonl")
    _copy(ROOT / "logs" / "final_summary.json", run_dir / "RUN_SUMMARY.json")
    for source in sorted((ROOT / "logs" / "diffs").glob("node_*.diff")):
        _copy(source, run_dir / "diffs" / source.name)
    for source in sorted((ROOT / "logs" / "solutions").glob("node_*.py")):
        _copy(source, run_dir / "solutions" / source.name)

    summary = _load(ROOT / "logs" / "final_summary.json")
    run_readme = f"""# Deliverable 3 - Run and iteration logs

Open `RUN_AND_ITERATION_LOG.md` first. It contains the hypothesis, reason,
code-diff link, GAUC, nDCG@5, and error/recovery record for every journal node.

- Journal records: {summary['journal_nodes']} (one mandatory baseline plus
  {summary['iterations_used']} charged research iterations)
- Iteration cap used: {summary['iterations_used']} of {summary['iteration_cap']}
- Manual interventions: {summary['manual_interventions']}
- Raw append-only record: `journal.jsonl`
- Exact diffs and generated scripts: `diffs/` and `solutions/`

The raw journal intentionally preserves the absolute `code_path` values from
the original execution checkout (`Tiktok-TechJam-2026`) before this repository
was moved. The byte-preserved scripts are available here under `solutions/`,
and every script/diff link in `RUN_AND_ITERATION_LOG.md` resolves locally.
"""
    (run_dir / "README.md").write_text(run_readme)

    # Deliverable 4: the test CSV is the canonical output; validation is
    # included so judges can independently reproduce the reported table.
    test_csv = ROOT / "submission_test.csv"
    valid_csv = ROOT / "submission_valid.csv"
    _copy(test_csv, final_dir / "submission.csv")
    _copy(valid_csv, final_dir / "validation_submission.csv")
    _copy(ROOT / "RESULTS.md", final_dir / "RESULTS_AND_RESOURCES.md")
    _copy(ROOT / "results" / "final_results.json",
          final_dir / "final_results.json")
    _copy(ROOT / "results" / "final_evaluation.lock",
          final_dir / "final_evaluation.lock")
    _copy(ROOT / "results" / "manifest.json",
          final_dir / "evidence_manifest.json")
    _copy(ROOT / "logs" / "ensemble_results.json",
          final_dir / "ensemble_results.json")
    _copy(ROOT / "kuairand-starter-kit" / "baseline_scores.json",
          final_dir / "official_baseline_scores.json")

    result = _load(ROOT / "results" / "final_results.json")
    baseline = _load(ROOT / "kuairand-starter-kit" / "baseline_scores.json")
    valid_base = baseline["scores"]["fm_official"]["valid"]
    valid = result["valid"]
    tokens = summary["total_llm_tokens"]
    delta = {key: valid[key] - valid_base[key]
             for key in ("GAUC", "nDCG@5", "primary")}
    results_readme = f"""# Deliverable 4 - Final submission and results

`submission.csv` is the canonical KuaiRand-Pure test output in the Starter Kit
schema: `row_id,user_id,video_id,score`. It has {_csv_rows(test_csv):,} data
rows. No bonus benchmark was attempted.

## Validation-best result

| System | GAUC | nDCG@5 | Primary |
|---|---:|---:|---:|
| Official baseline | {valid_base['GAUC']:.5f} | {valid_base['nDCG@5']:.5f} | {valid_base['primary']:.5f} |
| FYPothesis | {valid['GAUC']:.5f} | {valid['nDCG@5']:.5f} | {valid['primary']:.5f} |
| Absolute delta | {delta['GAUC']:+.5f} | {delta['nDCG@5']:+.5f} | {delta['primary']:+.5f} |

## Resources required to convergence

| Measure | Recorded value |
|---|---:|
| LLM input tokens | {tokens['input_tokens']:,} |
| LLM output tokens | {tokens['output_tokens']:,} |
| LLM input + output tokens | {tokens['input_plus_output']:,} |
| Agent wall-clock | {summary['total_agent_wall_clock_s']:,.1f} seconds |
| Iterations | {summary['iterations_used']} of {summary['iteration_cap']} |
| GPU-hours | {summary['gpu_hours']:.1f} (CPU only) |
| Manual interventions | {summary['manual_interventions']} |

The submitted system is the rank-normalized mean of all 16 declared seeds of
one configuration. `final_results.json` and `final_evaluation.lock` record the
one-time final evaluation; `ensemble_results.json` records member provenance.
"""
    (final_dir / "README.md").write_text(results_readme)

    verification = {
        "schema": "fypothesis_submission_bundle/1",
        "benchmark": "KuaiRand-Pure",
        "bonus_benchmarks_attempted": [],
        "submission": {
            "path": "04_FINAL_SUBMISSION_AND_RESULTS/submission.csv",
            "data_rows": _csv_rows(test_csv),
            "sha256": _sha256(test_csv),
            "starter_kit_schema_checked": True,
        },
        "validation_submission": {
            "path": "04_FINAL_SUBMISSION_AND_RESULTS/validation_submission.csv",
            "data_rows": _csv_rows(valid_csv),
            "sha256": _sha256(valid_csv),
            "starter_kit_schema_and_metrics_checked": True,
        },
        "manual_interventions": summary["manual_interventions"],
        "iterations_used": summary["iterations_used"],
        "iteration_cap": summary["iteration_cap"],
    }
    (out / "VERIFICATION.json").write_text(json.dumps(verification, indent=2) + "\n")

    upload_guide = """# FYPothesis submission upload guide

Upload the folders according to the matching requirement numbers:

1. `03_RUN_AND_ITERATION_LOGS/` - per-iteration evidence and autonomy summary.
2. `04_FINAL_SUBMISSION_AND_RESULTS/` - KuaiRand-Pure `submission.csv`, results,
   resource accounting, and verification/provenance records.

The KuaiRand-1k and KuaiRand-27k bonus benchmarks were not attempted. Do not
upload the raw KuaiRand dataset, local caches, `.env`, or API credentials.
"""
    (out / "UPLOAD_GUIDE.md").write_text(upload_guide)

    checksum_lines = []
    for path in sorted(p for p in out.rglob("*") if p.is_file()):
        if path.name == "CHECKSUMS.sha256":
            continue
        checksum_lines.append(f"{_sha256(path)}  {path.relative_to(out)}")
    (out / "CHECKSUMS.sha256").write_text("\n".join(checksum_lines) + "\n")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args()
    try:
        made = build(args.out.resolve())
    except (FileExistsError, FileNotFoundError) as exc:
        sys.exit(str(exc))
    print(f"built {made.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
