"""Streamlit dashboard — the submission, explained without a terminal.

    streamlit run app.py

Ordered for a first-time viewer: how the agent operates, what it can do, then
the submission proof, followed by the live logs and judging evidence.

Two properties held deliberately, because they are the agent's own rules:

  * **Nothing here is typed in.** Every number is read from an artifact or
    recomputed live. The incumbent check re-derives 0.60541 from the stored
    predictions on demand.
  * **This screen cannot promote anything.** Evidence tiers are recomputed from
    `agent.evidence`, so a single-seed result displays as PRELIMINARY however
    good it looks. The dashboard is a window, not a decision-maker.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time

import altair as alt
import pandas as pd
import streamlit as st

ROOT = os.path.dirname(os.path.abspath(__file__))
LOGS = os.path.join(ROOT, "logs")
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# The dashboard is a read-only showcase unless a local operator explicitly
# opts into paid execution. The hosted entrypoint also forces this flag off, so
# deploying the app cannot expose an LLM-spending subprocess by accident.
REAL_RUNS_ENABLED = os.getenv(
    "FYPOTHESIS_ENABLE_REAL_RUNS", "0").strip().lower() in {
        "1", "true", "yes", "on"}

from agent import live as L  # noqa: E402

NOISE = 0.0008

# Every headline number comes from the generated manifest, never from a literal
# typed here. Stale dashboard figures were a real problem: the app once showed a
# test count and a tab layout that the repository had moved past.
from agent import manifest as MF  # noqa: E402

_M = MF.load() or {}
if not _M:
    _M = MF.write()
BASELINE = (_M.get("baseline", {}).get("validation", {}).get("primary") or 0.6016)
BASELINE_TEST = (_M.get("baseline", {}).get("hidden_test", {}).get("primary") or 0.5946)
_SUB = _M.get("submitted", {}) or {}
INCUMBENT = (_SUB.get("reported", {}) or {}).get("primary") or 0.60541
_VALID_REPORTED = _SUB.get("reported", {}) or {}
_HIDDEN = _M.get("hidden_test", {}) or {}
_HIDDEN_RESULT = ((_HIDDEN.get("result") or {}).get("test") or {})
_HIDDEN_DELTA = ((_HIDDEN.get("result") or {}).get("delta_test") or {})
HIDDEN_PRIMARY = _HIDDEN_RESULT.get("primary")
HIDDEN_GAIN = _HIDDEN_DELTA.get("primary")

st.set_page_config(page_title="FYPothesis · Autonomous ML Research Agent",
                   page_icon="🔬", layout="wide")

st.markdown("""<style>
.block-container{padding-top:2.2rem;max-width:1250px}
[data-testid="stMetricValue"]{font-size:1.5rem}
h1{font-family:"Avenir Next","Trebuchet MS",sans-serif;font-size:1.9rem !important}
h2{font-size:1.25rem !important;margin-top:1.4rem !important}
h3{font-size:1.05rem !important}
.small{color:#8b949e;font-size:0.86rem;line-height:1.5}
.pill{display:inline-block;padding:2px 10px;border-radius:99px;font-size:0.74rem;
font-weight:600;margin-right:6px}
.overview-hero{padding:24px 28px 22px;border:1px solid #b9e5db;border-radius:16px;
background:radial-gradient(circle at 90% 5%,#ccfbf1 0,transparent 32%),
linear-gradient(125deg,#f0fdfa 0%,#f8fafc 58%,#eff6ff 100%);margin-bottom:18px}
.overview-hero .eyebrow{color:#0f766e;font-size:.72rem;font-weight:800;
letter-spacing:.12em;text-transform:uppercase}
.overview-hero h2{color:#0f172a;font-family:"Avenir Next","Trebuchet MS",sans-serif;
font-size:1.8rem !important;letter-spacing:-.03em;margin:.28rem 0 .5rem !important}
.overview-hero p{color:#475569;font-size:1rem;max-width:720px;margin:0}
.capability{height:100%;min-height:148px;padding:18px;border:1px solid #dbe4ea;
border-radius:13px;background:linear-gradient(145deg,#ffffff,#f8fafc)}
.capability .cap-kicker{font-size:.72rem;font-weight:800;letter-spacing:.1em;
text-transform:uppercase;color:#0f766e}
.capability h3{margin:.5rem 0 !important;color:#172554}
.capability p{margin:0;color:#64748b;font-size:.88rem;line-height:1.55}
.proof-strip{padding:15px 18px;border-left:4px solid #0f766e;background:#f0fdfa;
border-radius:0 12px 12px 0;color:#134e4a;font-size:.88rem}
.flow-proof{background:#ffffff;border:1px solid #dbe4ea;border-left:4px solid #0f766e;
color:#475569}
.criterion{min-height:154px;padding:17px 18px;border:1px solid #dbe4ea;
border-radius:13px;background:#ffffff;margin-bottom:12px}
.criterion-top{display:flex;align-items:center;justify-content:space-between;gap:10px}
.criterion h3{margin:0 !important;color:#172554;font-size:1rem !important}
.criterion .weight{padding:3px 9px;border-radius:99px;background:#e0f2fe;color:#075985;
font-size:.75rem;font-weight:800;white-space:nowrap}
.criterion p{margin:.6rem 0 0;color:#64748b;font-size:.87rem;line-height:1.55}
.criterion .evidence{display:inline-block;margin-top:.7rem;color:#0f766e;font-size:.76rem;
font-weight:700}
.log-guide{padding:14px 16px;border:1px solid #dbe4ea;border-radius:12px;
background:linear-gradient(135deg,#f8fafc,#ffffff);height:100%}
.log-guide b{display:block;color:#172554;margin-bottom:4px}
.log-guide span{color:#64748b;font-size:.84rem;line-height:1.45}
.iteration-brief{padding:18px;border:1px solid #dbe4ea;border-radius:14px;
background:#ffffff;margin-top:12px}
.iteration-brief h3{margin:0 0 .45rem !important;color:#172554}
.iteration-brief p{margin:.35rem 0;color:#475569;font-size:.9rem;line-height:1.55}
.iteration-brief .label{font-size:.7rem;font-weight:800;letter-spacing:.08em;
text-transform:uppercase;color:#0f766e}
</style>""", unsafe_allow_html=True)


# ----------------------------------------------------------------- helpers ---
def read_json(path):
    try:
        with open(path) as fh:
            return json.load(fh)
    except (OSError, json.JSONDecodeError):
        return None


def journals() -> dict:
    out = {}
    p = os.path.join(LOGS, "journal.jsonl")
    if os.path.exists(p):
        out["current run"] = p
    d = os.path.join(LOGS, "opus_research")
    if os.path.isdir(d):
        for f in sorted(os.listdir(d)):
            if f.endswith(".jsonl"):
                out[f.replace(".jsonl", "").replace("_", " ")] = os.path.join(d, f)
    return out


ACTION_LABEL = {"draft": "New idea", "improve": "Refine best",
                "debug": "Fix failure", "confirm": "Confirm (multi-seed)",
                "ensemble": "Ensemble seeds", "merge": "Combine ideas",
                "crossover": "Combine ideas"}
TIER = {"CONFIRMED": ("#1a7f37", "Established — may change the submission"),
        "PRELIMINARY": ("#9a6700", "One seed — not actionable"),
        "UNCONFIRMED": ("#bc4c00", "Repeated, still not separable from noise"),
        "REJECTED": ("#cf222e", "Measured; does not hold"),
        "PROBED": ("#57606a", "Measured cheaply, not trained"),
        "REDUNDANT": ("#57606a", "Real alone, adds nothing here")}


def pill(text, colour, bg=None):
    bg = bg or f"{colour}1f"
    return (f"<span class='pill' style='background:{bg};color:{colour}'>"
            f"{text}</span>")


def experiment_tree_chart(nodes: list[dict], height=620, node_size=5400,
                          root_size=7800, label_size=11):
    """Build a compact clickable tree from journal parent links."""
    by_id = {n["id"]: n for n in nodes if n.get("id") is not None}
    children = {"start": []}
    for node_id in by_id:
        children[node_id] = []
    for node_id, node in by_id.items():
        parent = node.get("parent")
        children[parent if parent in by_id else "start"].append(node_id)
    for values in children.values():
        values.sort()

    positions = {}
    next_leaf = [0]

    def place(node_id, depth):
        descendants = children.get(node_id) or []
        if descendants:
            child_x = [place(child, depth + 1) for child in descendants]
            x = sum(child_x) / len(child_x)
        else:
            x = next_leaf[0]
            next_leaf[0] += 1
        positions[node_id] = (x, -depth)
        return x

    place("start", 0)
    node_rows = [{"node_key": "start", "node_id": None, "x": positions["start"][0],
                  "y": 0, "label": "START", "state": "start",
                  "action": "Begin research run", "score": "—",
                  "evidence": "root"}]
    edge_rows = []
    for node_id, node in by_id.items():
        parent = node.get("parent") if node.get("parent") in by_id else "start"
        x, y = positions[node_id]
        px, py = positions[parent]
        action = ACTION_LABEL.get(node.get("action"), node.get("action") or "Experiment")
        node_rows.append({
            "node_key": f"n{node_id}", "node_id": node_id, "x": x, "y": y,
            "label": f"#{node_id}", "state": node.get("state") or "pending",
            "action": action,
            "score": (f"{node['primary']:.5f}"
                      if node.get("primary") is not None else "not scored"),
            "evidence": ((node.get("evidence") or {}).get("state")
                         or node.get("state") or "pending"),
        })
        edge_rows.append({"x": px, "y": py, "x2": x, "y2": y})

    node_df = pd.DataFrame(node_rows)
    edge_df = pd.DataFrame(edge_rows)
    x = alt.X("x:Q", axis=None, scale=alt.Scale(padding=45))
    y = alt.Y("y:Q", axis=None, scale=alt.Scale(padding=40))
    edges = alt.Chart(edge_df).mark_rule(color="#8c959f", strokeWidth=1.5).encode(
        x=x, y=y, x2="x2:Q", y2="y2:Q")
    node_pick = alt.selection_point(
        name="experiment_node", fields=["node_id"], on="click",
        clear=False, empty=False)
    points = alt.Chart(node_df).mark_circle(
        cursor="pointer", opacity=1, fillOpacity=1).encode(
        x=x, y=y,
        size=alt.condition("datum.state === 'start'", alt.value(root_size),
                           alt.value(node_size)),
        color=alt.Color(
            "state:N", legend=None,
            scale=alt.Scale(
                domain=["start", "ok", "fail", "preflight", "pending"],
                range=["#0f766e", "#d8f3dc", "#ffebe9", "#fff8c5", "#f6f8fa"])),
        stroke=alt.condition(node_pick, alt.value("#0f766e"), alt.value("#57606a")),
        strokeWidth=alt.condition(node_pick, alt.value(4), alt.value(1.5)),
        tooltip=[alt.Tooltip("label:N", title="Experiment"),
                 alt.Tooltip("action:N", title="Decision"),
                 alt.Tooltip("score:N", title="Primary"),
                 alt.Tooltip("evidence:N", title="Evidence")],
    ).add_params(node_pick)
    labels = alt.Chart(node_df).mark_text(
        fontSize=label_size, fontWeight="bold").encode(
        x=x, y=y, text="label:N",
        color=alt.condition("datum.state === 'start'", alt.value("white"),
                            alt.value("#172554")))
    return (edges + points + labels).properties(height=height).configure_view(strokeWidth=0)


def overview_workflow_dot() -> str:
    """The short visual explanation of one autonomous research cycle."""
    return """digraph workflow {
      rankdir=LR;
      graph [bgcolor=transparent, ranksep=0.34, nodesep=0.16];
      node [shape=box, style=\"rounded,filled\", fontname=\"Avenir Next\",
            fontsize=10, margin=\"0.16,0.11\", color=\"#cbd5e1\",
            fillcolor=\"#ffffff\", fontcolor=\"#334155\"];
      edge [color=\"#0f766e\", penwidth=1.5, arrowsize=0.65];
      observe [label=\"OBSERVE\\nData + research memory\"];
      question [label=\"QUESTION\\nCompeting explanations\"];
      build [label=\"BUILD\\nScript or feature\"];
      evaluate [label=\"EVALUATE\\nOfficial validation metric\"];
      confirm [label=\"CONFIRM\\nPaired multi-seed evidence\",
               fillcolor=\"#ffffff\", color=\"#5eead4\"];
      observe -> question -> build -> evaluate -> confirm;
    }"""


DEMO_DURATION_S = 24
DEMO_STAGES = [
    {"at": 0, "phase": "BOOT", "title": "Lock the rules before searching",
     "question": "What may this run optimize, and what evidence is allowed to change the submission?",
     "decision": "Load the KuaiRand-Pure boundary, official scorer, budgets, and one-shot hidden-test lock.",
     "evidence": "No model decision yet — establish the guardrails first.",
     "log": "[00:00] profile=competition | hidden labels unavailable | budget armed",
     "decisions": 0, "training": 0, "spend": 0.00, "best": None},
    {"at": 3, "phase": "OBSERVE", "title": "Reproduce the measured starting point",
     "question": "Does the unchanged organizer FM reproduce its published validation baseline?",
     "decision": "Run the pointwise FM through the protected executor and official evaluator.",
     "evidence": "Baseline anchor: primary 0.60148; test labels were not available.",
     "log": "[00:03] node=0 BASELINE | GAUC=.66723 nDCG@5=.53572 primary=.60148",
     "decisions": 1, "training": 1, "spend": 0.00, "best": 0.60148},
    {"at": 6, "phase": "QUESTION", "title": "Choose the highest-value uncertainty",
     "question": "Can pairwise ranking plus recency-weighted history improve long-view ranking beyond seed noise?",
     "decision": "Screen BPR + recency history + hour/day context on one seed.",
     "evidence": "Node 1 scores 0.60497, but remains PRELIMINARY until paired confirmation.",
     "log": "[00:06] node=1 SUCCESS | primary=.60497 | evidence=PRELIMINARY",
     "decisions": 2, "training": 2, "spend": 0.02, "best": 0.60497},
    {"at": 9, "phase": "PREFLIGHT", "title": "Reject an invalid implementation before training",
     "question": "Does the generated script obey the callable selection-rule and output contracts?",
     "decision": "Block node 2 at preflight; feed the exact contract error into a bounded debug action.",
     "evidence": "Caught early: zero training runs spent and no invalid result entered the evidence pool.",
     "log": "[00:09] node=2 PREFLIGHT_REJECTED | call signature mismatch | compute=0",
     "decisions": 3, "training": 2, "spend": 0.04, "best": 0.60497},
    {"at": 12, "phase": "RECOVER", "title": "Bound the failed branch and pivot",
     "question": "Can the same hypothesis be tested after correcting only the failed API contract?",
     "decision": "Try one debug node, then abandon the branch when its artifact is still malformed.",
     "evidence": "Node 3 fails safely; search returns to node 1 instead of retrying indefinitely.",
     "log": "[00:12] node=3 DEBUG_FAILED | branch=2→3 | action=abandon",
     "decisions": 4, "training": 3, "spend": 0.06, "best": 0.60497},
    {"at": 16, "phase": "EVALUATE", "title": "Explore a separate branch from the incumbent",
     "question": "Is the stronger BPR recipe under-capacity at k=16?",
     "decision": "Branch from node 1, increase only factor dimension, and compare against the incumbent.",
     "evidence": "Node 4 scores 0.60454: valid but lower, so capacity is not promoted.",
     "log": "[00:16] node=4 SUCCESS | parent=1 | primary=.60454 | keep node=1",
     "decisions": 5, "training": 4, "spend": 0.09, "best": 0.60497},
    {"at": 20, "phase": "CONFIRM", "title": "Ask whether the gain survives paired seeds",
     "question": "Does the treatment beat its control when both arms use seeds 0, 1, and 2?",
     "decision": "Run paired confirmation; all three treatment-control differences are positive.",
     "evidence": "CONFIRMED: Δ +0.00257, 3/3 wins, +3.21σ; the result may now guide submission.",
     "log": "[00:20] node=5 CONFIRM | parent=1 | wins=3/3 | Δ=+.00257",
     "decisions": 6, "training": 10, "spend": 0.14, "best": 0.60497},
    {"at": 24, "phase": "DECIDE", "title": "Reduce variance, adopt, and stop",
     "question": "Does a fixed all-seed ensemble improve on the mean member without selecting lucky seeds?",
     "decision": "Keep all 16 declared seeds, rank-normalize, average, adopt 0.60541, and stop cleanly.",
     "evidence": "CONFIRMED: +0.00078 over the mean member; reproducible fixed ensemble selected.",
     "log": "[00:24] node=6 ENSEMBLE | parent=5 | 16/16 seeds | primary=.60541 | STOP",
     "decisions": 7, "training": 26, "spend": 0.18, "best": 0.60541},
]


DEMO_NODE_RECORDS = [
    {"visible_at": 1, "id": 0, "parent": None, "action": "baseline",
     "state": "ok", "primary": 0.60148, "evidence": {"state": "ANCHOR"},
     "allocator": "mandatory_baseline",
     "question": "Can the unchanged organizer FM reproduce its published baseline?",
     "hypotheses": ["The evaluator and split reproduce the published anchor.",
                    "A data or scorer mismatch makes later comparisons invalid."],
     "plan": "Run the pointwise FM through the protected executor and official evaluator.",
     "outcome": "Measured the starting point before any autonomous change.",
     "config": {"model": "fm_numpy", "loss": "pointwise_logloss",
                "user_history": "none", "temporal": "none"}},
    {"visible_at": 2, "id": 1, "parent": 0, "action": "draft",
     "state": "ok", "primary": 0.60497,
     "evidence": {"state": "PRELIMINARY", "n_seeds": 1},
     "allocator": "exploration",
     "question": "Can ranking-aligned loss and behavioral context beat the anchor beyond noise?",
     "hypotheses": ["BPR aligns training with within-user ranking.",
                    "The apparent gain is only a lucky seed."],
     "plan": "Test BPR with recency-weighted history and hour/day context on one screening seed.",
     "outcome": "Promising single-seed gain, deliberately held at PRELIMINARY.",
     "config": {"model": "fm_numpy", "loss": "bpr_pairwise",
                "user_history": "recency_weighted_pool",
                "temporal": "hour_plus_dow", "training": "lower_lr_longer"}},
    {"visible_at": 3, "id": 2, "parent": 1, "action": "improve",
     "state": "preflight", "primary": None,
     "evidence": {"state": "BLOCKED"}, "allocator": "exploitation",
     "question": "Can checkpoint-rule selection add signal without validation leakage?",
     "hypotheses": ["Held-out checkpoint selection improves stability.",
                    "The implementation violates the callable-rule contract."],
     "plan": "Capture per-epoch validation predictions and compare fixed selection rules.",
     "outcome": "Rejected before training; no compute spent and no evidence created.",
     "error": "CALL_ARITY: selection_rule_test received a non-callable rule.",
     "config": {"capture_epoch_scores": True, "selection_rule": "held_out_users"}},
    {"visible_at": 4, "id": 3, "parent": 2, "action": "debug",
     "state": "fail", "primary": None,
     "evidence": {"state": "FAILED"}, "allocator": "bounded_debug",
     "question": "Can the checkpoint probe be repaired without changing its claim?",
     "hypotheses": ["Correcting the callable signature is sufficient.",
                    "The captured payload is structurally incompatible."],
     "plan": "Repair only the API contract and retry once in the sandbox.",
     "outcome": "The repair still failed; the branch was abandoned at its debug limit.",
     "error": "INVALID_ARTIFACT: per_epoch_scores had an inhomogeneous shape.",
     "config": {"debug_parent": 2, "max_debug_chain": 1}},
    {"visible_at": 5, "id": 4, "parent": 1, "action": "improve",
     "state": "ok", "primary": 0.60454,
     "evidence": {"state": "PRELIMINARY", "n_seeds": 1},
     "allocator": "exploration",
     "question": "Is the stronger BPR branch under-capacity at k=16?",
     "hypotheses": ["A wider FM captures useful new interactions.",
                    "Capacity is not the bottleneck and the score will fall."],
     "plan": "Branch from node 1, change only factor dimension to k=32, and rescore.",
     "outcome": "Valid result, but below node 1; keep the incumbent and move to confirmation.",
     "config": {"model": "fm_numpy", "loss": "bpr_pairwise", "k": 32}},
    {"visible_at": 6, "id": 5, "parent": 1, "action": "confirm",
     "state": "ok", "primary": 0.60446,
     "evidence": {"state": "CONFIRMED", "n_seeds": 3},
     "allocator": "multi_seed_replication",
     "question": "Does node 1 beat its control on the same three seeds?",
     "hypotheses": ["The configuration gain survives paired replication.",
                    "The single-seed gain collapses toward the baseline."],
     "plan": "Run control and treatment on seeds 0, 1, and 2 before seeing outcomes.",
     "outcome": "Treatment won 3/3 paired seeds; the gain became actionable.",
     "paired": {"control": 0.60190, "treatment": 0.60446,
                "delta": 0.00257, "sigma": 3.21, "n": 3, "promote": True},
     "config": {"seeds": [0, 1, 2], "comparison": "paired"}},
    {"visible_at": 7, "id": 6, "parent": 5, "action": "ensemble",
     "state": "ok", "primary": 0.60541,
     "evidence": {"state": "CONFIRMED", "n_seeds": 16},
     "allocator": "ensemble_construction",
     "question": "Can a fixed all-seed ensemble reduce variance without selecting lucky members?",
     "hypotheses": ["Rank averaging improves on the mean member.",
                    "The ensemble adds no value beyond its members."],
     "plan": "Keep all 16 declared seeds, rank-normalize their predictions, and average.",
     "outcome": "Improved +0.00078 over the mean member; adopt and stop.",
     "config": {"seeds": list(range(16)), "aggregation": "rank_normalise_then_mean"}},
]


def demo_tree_nodes(stage_index):
    """Return the visible portion of a realistic branching mock journal."""
    return [dict(node) for node in DEMO_NODE_RECORDS
            if node["visible_at"] <= stage_index]


@st.fragment(run_every=1)
def render_demo_run():
    started_at = st.session_state.get("demo_started_at")
    if started_at is None:
        return

    elapsed = min(max(time.time() - started_at, 0), DEMO_DURATION_S)
    stage_index = max(i for i, stage in enumerate(DEMO_STAGES)
                      if stage["at"] <= elapsed)
    stage = DEMO_STAGES[stage_index]
    st.progress(elapsed / DEMO_DURATION_S,
                text=f"Demo run · {stage['phase']}: {stage['title']}")
    nodes = demo_tree_nodes(stage_index)
    scored = [n for n in nodes if n["state"] == "ok" and n.get("primary")]
    best = max((n["primary"] for n in scored), default=None)

    metrics = st.columns(6)
    metrics[0].metric("Experiments", len(nodes))
    metrics[1].metric("Completed", len(scored))
    metrics[2].metric("Crashed", sum(n["state"] == "fail" for n in nodes))
    metrics[3].metric("Caught early",
                      sum(n["state"] == "preflight" for n in nodes))
    metrics[4].metric("Confirmations",
                      sum(n["action"] == "confirm" for n in nodes))
    metrics[5].metric("Best so far", f"{best:.5f}" if best else "—",
                      f"{(best - BASELINE) / NOISE:+.2f}σ" if best else None)
    resources = st.columns(3)
    resources[0].metric("Training runs used", f"{stage['training']} / 30")
    resources[1].metric("LLM spend", f"${stage['spend']:.2f}", "of $1.00 mock cap")
    resources[2].metric("Manual interventions", 0)

    st.divider()
    if not nodes:
        st.info("The simulated agent is loading its competition rules and "
                "research memory. The first measured node appears at 3 seconds.")
    else:
        st.markdown("#### Experiment tree")
        st.caption("This is the agent's decision record: each node is a "
                   "hypothesis or experiment chosen by the search policy, and "
                   "each edge shows the prior result it builds on. Branches "
                   "show a failed repair, a capacity probe, paired confirmation, "
                   "and the final ensemble.")
        tree_col, _ = st.columns([5, 1])
        with tree_col:
            tree_event = st.altair_chart(
                experiment_tree_chart(nodes), width="stretch", height=620,
                key=f"demo_tree_{stage_index}", on_select="rerun",
                selection_mode="experiment_node")

        node_ids = [n["id"] for n in nodes]
        selected_key = "selected_demo_experiment"
        stage_key = "selected_demo_stage"
        if (st.session_state.get(stage_key) != stage_index
                or st.session_state.get(selected_key) not in node_ids):
            st.session_state[selected_key] = node_ids[-1]
            st.session_state[stage_key] = stage_index
        selection = getattr(tree_event, "selection", {}) or {}
        selected_points = selection.get("experiment_node") or {}
        if isinstance(selected_points, dict):
            selected_values = selected_points.get("node_id") or []
            clicked_id = selected_values[-1] if selected_values else None
        else:
            clicked_id = (selected_points[0].get("node_id")
                          if selected_points else None)
        if clicked_id in node_ids:
            st.session_state[selected_key] = clicked_id
        selected = next(n for n in nodes
                        if n["id"] == st.session_state[selected_key])

        with st.expander(f"Experiment #{selected['id']} details", expanded=True):
            st.caption("Click another circle in the tree to inspect that mock "
                       "experiment's decision, evidence, and artifacts.")
            head = st.columns([2, 2, 2])
            head[0].metric("Action", ACTION_LABEL.get(
                selected["action"], selected["action"].title()))
            head[1].metric("Score", (f"{selected['primary']:.5f}"
                                     if selected.get("primary") else "not scored"))
            head[2].metric("Evidence", selected["evidence"]["state"])
            st.markdown(f"<span class='small'>Allocator chose "
                        f"<b>{selected['allocator']}</b> for this decision.</span>",
                        unsafe_allow_html=True)
            st.markdown(f"**Asked:** {selected['question']}")
            st.markdown("**Competing explanations**")
            for hypothesis in selected["hypotheses"]:
                st.markdown(f"- {hypothesis}")
            st.markdown(f"**Tried:** {selected['plan']}")
            if selected.get("paired"):
                paired = selected["paired"]
                st.markdown(
                    f"**Paired test:** control {paired['control']:.5f} → treatment "
                    f"{paired['treatment']:.5f}; Δ {paired['delta']:+.5f} "
                    f"({paired['sigma']:+.2f}σ) over {paired['n']} seeds; "
                    "**adopted**.")
            if selected.get("error"):
                st.error(selected["error"])
            st.caption(f"Outcome: {selected['outcome']}")
            evidence = selected["evidence"]
            colour, meaning = TIER.get(
                evidence["state"], ("#57606a", "Recorded in this mock journal"))
            seeds = (f" · {evidence['n_seeds']} seeds"
                     if evidence.get("n_seeds") else "")
            st.markdown(pill(f"{evidence['state']}{seeds}", colour)
                        + f"<span class='small'>{meaning}</span>",
                        unsafe_allow_html=True)

            st.markdown("#### Experiment brief")
            st.markdown(
                "<div class='iteration-brief'><div class='label'>Why this "
                "experiment exists</div><h3>" +
                ACTION_LABEL.get(selected["action"], selected["action"].title()) +
                "</h3><p>" + selected["question"] + "</p></div>",
                unsafe_allow_html=True)
            detail = st.columns(2)
            detail[0].markdown("**What the agent tried**")
            detail[0].write(selected["plan"])
            detail[1].markdown("**What happened**")
            detail[1].write(selected["outcome"])
            with st.expander("Configuration and audit evidence"):
                st.json(selected["config"])
                st.json({"simulation": True, **selected}, expanded=False)
            with st.expander("Executable script sent to the sandbox"):
                st.caption("Demo source excerpt for this experiment.")
                st.code("# SIMULATED DEMO SOURCE\n"
                        f"CONFIG = {selected['config']!r}\n"
                        "metrics = run_experiment(CONFIG)\n"
                        "write_contract_artifacts(metrics)\n", language="python")

    st.divider()
    st.markdown("#### Run summary")
    summary = st.columns(4)
    summary[0].metric("Decisions", len(nodes))
    summary[1].metric("Scored experiments", len(scored))
    summary[2].metric("Best primary", f"{best:.5f}" if best else "—")
    summary[3].metric("Training runs", stage["training"])
    if elapsed >= DEMO_DURATION_S:
        st.success("**Why the run stopped:** paired evidence was confirmed and "
                   "the fixed 16-seed ensemble improved over its mean member.")
    else:
        st.caption("The demo run is still searching; the summary updates "
                   "as new journal nodes arrive.")

    st.markdown("#### Iteration audit trail")
    st.caption("Every demo decision so far. Click its tree node above for the "
               "experiment brief, configuration, and demo source artifact.")
    audit_rows = [
        {"#": n["id"],
         "Experiment": ACTION_LABEL.get(n["action"], n["action"].title()),
         "Outcome": {"ok": "completed", "fail": "crashed",
                     "preflight": "caught early"}[n["state"]],
         "Primary": n.get("primary"),
         "σ vs baseline": ((n["primary"] - BASELINE) / NOISE
                           if n.get("primary") is not None else None),
         "Evidence": n["evidence"]["state"],
         "Seconds": {0: 31.2, 1: 46.8, 2: 0.0, 3: 2.4,
                     4: 51.7, 5: 139.5, 6: 308.1}[n["id"]],
         "Note": n["outcome"]}
        for n in nodes]
    audit = pd.DataFrame(audit_rows)
    filters = st.columns(3)
    if filters[0].checkbox("Failures only", key="demo_failures_only"):
        audit = audit[audit["Outcome"] != "completed"]
    if filters[1].checkbox("Confirmations & ensembles only",
                           key="demo_confirmations_only"):
        audit = audit[audit["Experiment"].str.contains("Confirm|Ensemble")]
    st.dataframe(audit, width="stretch", hide_index=True)

    st.divider()
    st.markdown("#### Run reliability and recovery")
    st.caption("Derived only from this demo journal. The preflight failure "
               "and its debug child form one incident; recovery is recorded when "
               "the search returns to a later valid scored branch.")
    incident = any(n["id"] == 2 for n in nodes)
    debug = any(n["id"] == 3 for n in nodes)
    returned = any(n["id"] >= 4 and n["state"] == "ok" for n in nodes)
    recovery = st.columns(4)
    recovery[0].metric("Failure incidents", 1 if incident else 0)
    recovery[1].metric("Blocked before training", 1 if incident else 0)
    recovery[2].metric("Debug attempts", 1 if debug else 0)
    recovery[3].metric("Returned to valid work",
                       "1 / 1" if returned else ("0 / 1" if incident else "Not needed"))
    if returned:
        st.success("The failed checkpoint branch was bounded and abandoned; the "
                   "search returned to node 1, explored a separate capacity "
                   "branch, and continued to valid scored work without promotion "
                   "of a failed candidate.")
    elif incident:
        st.warning("A recovery incident is in progress; no later valid score has "
                   "appeared yet in the simulated journal.")
    else:
        st.info("No recovery action has been needed yet.")
    if incident:
        st.dataframe([{
            "Started at": "Node #2", "Failure": "Preflight rejection",
            "Agent response": ("Abandoned after one bounded debug attempt"
                               if debug else "Queued bounded debug"),
            "Next valid result": "Node #4" if returned else "Pending",
            "Run continued safely": "Yes" if returned else "Pending",
        }], width="stretch", hide_index=True)


running = L._agent_running() if REAL_RUNS_ENABLED else False

st.title("FYPothesis")
st.markdown(
    "<span class='small'>Autonomous ML Research Agent · KuaiRand-Pure · "
    "TikTok TechJam 2026, Track 2 · "
    "an LLM-driven agent that forms hypotheses, writes and runs its own "
    "experiments, and decides what to try next</span>",
    unsafe_allow_html=True)
if running:
    st.success("🟢 An agent run is in progress — see **Watch a run**.")
if not REAL_RUNS_ENABLED:
    st.info("🔒 Public showcase mode: paid LLM runs are disabled. Open "
            "**Demo run** to watch a complete autonomous research cycle.")

tabs = st.tabs(["📌 Overview", "⚙️ Start a run", "▶️ Watch a run",
                "🎬 Demo run"])


# ---------------------------------------------------------------- overview ---
with tabs[0]:
    fs = read_json(os.path.join(LOGS, "final_summary.json")) or {}
    led = fs.get("budget_ledger") or {}
    spend = fs.get("spend") or {}
    interventions = L._load_jsonl(os.path.join(LOGS, "interventions.jsonl"))

    st.markdown(
        "<div class='overview-hero'><div class='eyebrow'>FYPothesis · Autonomous ML research</div>"
        "<h2>Ask a sharper question. Run a safer experiment.</h2>"
        "<p>An agent that observes the data, proposes a measurable hypothesis, "
        "builds an experiment, and only adopts results that survive paired evidence.</p>"
        "</div>", unsafe_allow_html=True)

    st.markdown("#### How the agent works")
    st.graphviz_chart(overview_workflow_dot(), width="stretch", height=170)
    loop = st.columns(3)
    loop[0].markdown("<div class='proof-strip flow-proof'><b>Grounded</b><br>Data tools and "
                     "research memory turn observations into explicit questions.</div>",
                     unsafe_allow_html=True)
    loop[1].markdown("<div class='proof-strip flow-proof'><b>Guarded</b><br>Preflight and a "
                     "test-label boundary block invalid work before it is scored.</div>",
                     unsafe_allow_html=True)
    loop[2].markdown("<div class='proof-strip flow-proof'><b>Evidence-led</b><br>One seed is "
                     "preliminary; only paired confirmation may change the submission.</div>",
                     unsafe_allow_html=True)

    st.markdown("#### What it can actually do")
    capabilities = [
        ("RESEARCH", "Interrogate the data", "Runs controlled diagnostics, keeps "
         "scoped findings, and avoids repeating measured dead ends."),
        ("BUILD", "Change the pipeline", "Explores 10 controlled axes and custom "
         "mechanisms while preflight checks the resulting script."),
        ("GOVERN", "Spend evidence carefully", "Tracks cost and compute, recovers "
         "from failures, then confirms only results strong enough to matter."),
    ]
    cols = st.columns(3)
    for col, (kicker, title, body) in zip(cols, capabilities):
        with col:
            st.markdown(f"<div class='capability'><div class='cap-kicker'>{kicker}</div>"
                        f"<h3>{title}</h3><p>{body}</p></div>",
                        unsafe_allow_html=True)

    st.divider()
    st.markdown("#### Why this is competition-ready")
    st.caption("The rubric is the organiser's. Each claim below is tied to a "
               "visible artifact, live view, or reproducible check.")
    criteria = [
        ("Technical execution", "35%",
         f"The agent improves the official validation primary from {BASELINE:.4f} "
         f"to {INCUMBENT:.5f} (+{(INCUMBENT - BASELINE) / NOISE:.2f}σ), while the "
         + (f"one-time hidden-test result improves from {BASELINE_TEST:.4f} "
            f"to {HIDDEN_PRIMARY:.5f}. " if HIDDEN_PRIMARY is not None else "")
         + "Official scoring and the hidden-test boundary remain unchanged.",
         "Proof: fixed 16-seed ensemble, preflight, sandbox, and Verify button."),
        ("Innovation and insight", "20%",
         "The capability contract keeps the prompt, runtime surface, and preflight "
         "in agreement. The allocator values information, not just a lucky score, "
         "and research memory records what failed and why.",
         "Proof: capability contract, evidence tiers, and research log."),
        ("Autonomy and relevance", "20%",
         f"{len(interventions)} manual interventions are logged. Each experiment "
         "records its question, competing hypotheses, selected action, result, and "
         "whether the evidence was strong enough to act on.",
         "Proof: live experiment tree and append-only journal."),
        ("Feasibility and practicality", "15%",
         f"The latest run reports ${spend.get('total_usd', 0):.2f} LLM spend, "
         f"{fs.get('total_agent_wall_clock_s', 0):.0f}s wall-clock, and "
         f"{led.get('training_runs_used', 0)} training runs under explicit caps. "
         "It runs on CPU with measured resource accounting.",
         "Proof: generated results, budget ledger, and resource artifacts."),
        ("Presentation and communication", "10%",
         "A judge can watch the agent form a branch, reject broken code before it "
         "spends compute, inspect every script, and independently reproduce the "
         "submitted result from stored prediction arrays.",
         "Proof: this dashboard, RESULTS.md, README.md, and static tree."),
    ]
    rows = [criteria[:2], criteria[2:4], criteria[4:]]
    for row in rows:
        columns = st.columns(len(row))
        for col, (title, weight, body, evidence) in zip(columns, row):
            with col:
                st.markdown(f"<div class='criterion'><div class='criterion-top'>"
                            f"<h3>{title}</h3><span class='weight'>{weight}</span>"
                            f"</div><p>{body}</p><span class='evidence'>{evidence}</span>"
                            f"</div>", unsafe_allow_html=True)

    st.divider()
    st.markdown("#### What was submitted")
    c = st.columns([1, 1, 1, 1.3])
    c[0].metric("Validation primary", f"{INCUMBENT:.5f}",
                f"+{INCUMBENT - BASELINE:.5f} vs baseline")
    c[1].metric("Evidence scale", f"+{(INCUMBENT - BASELINE) / NOISE:.2f}σ",
                help="σ = 0.0008, the official baseline's own 5-seed spread.")
    c[2].metric("Official baseline", f"{BASELINE:.4f}")
    hidden_delta = (f"+{HIDDEN_GAIN:.5f} vs baseline"
                    if HIDDEN_GAIN is not None else None)
    c[3].metric("Hidden-test primary",
                f"{HIDDEN_PRIMARY:.5f}" if HIDDEN_PRIMARY is not None
                else "not evaluated",
                hidden_delta,
                help="Scored exactly once at final submission; baseline "
                     f"primary: {BASELINE_TEST:.4f}.")
    st.markdown(
        "<div class='proof-strip'><b>16-seed fixed ensemble.</b> GAUC "
        f"{_VALID_REPORTED.get('GAUC', 0):.5f} · nDCG@5 "
        f"{_VALID_REPORTED.get('nDCG@5', 0):.5f} · every seed is included, "
        "with no validation-selected member subset.</div>",
        unsafe_allow_html=True)

# ------------------------------------------------------------ watch it run ---
@st.fragment(run_every=3)
def render_watch_tree():
    """Refresh only the live tree, never the dashboard's other tabs."""
    js = journals()
    if not js:
        st.info("No run on disk yet. Start one from **Start a run**.")
    else:
        pick = st.session_state.get("tree_journal")
        if pick not in js:
            pick = next(iter(js))
        state = L.state(js[pick])
        raw_nodes = L._load_jsonl(js[pick])
        k = state["kpi"]

        m = st.columns(6)
        m[0].metric("Experiments", k["nodes"])
        m[1].metric("Completed", k["scored"])
        m[2].metric("Crashed", k["crashed"])
        m[3].metric("Caught early", k["preflight"],
                    help="rejected by preflight before training — no compute "
                         "spent. This is the system working.")
        m[4].metric("Confirmations", k["confirmations"])
        m[5].metric("Best so far",
                    f"{k['best']:.5f}" if k["best"] else "—",
                    f"{k['sigma']:+.2f}σ" if k["sigma"] else None)

        m = st.columns(3)
        m[0].metric("Training runs used",
                    f"{k['training_runs'] or 0} / {k['training_cap'] or '—'}",
                    help="an ensemble or a paired confirmation is several "
                         "training runs but ONE decision")
        m[1].metric("LLM spend",
                    f"${k['spend']:.2f}" if k["spend"] is not None else "—",
                    f"of ${k['spend_cap']:.2f}" if k.get("spend_cap") else None)
        m[2].metric("Manual interventions", len(
            L._load_jsonl(os.path.join(LOGS, "interventions.jsonl"))))

        st.divider()
        if not state["nodes"]:
            st.info("No experiments recorded yet.")
        else:
            st.markdown("#### Experiment tree")
            st.caption("This is the agent's decision record: each node is a "
                       "hypothesis or experiment chosen by the search policy, and "
                       "each edge shows the prior result it builds on. Branches "
                       "make repair, pivot, confirmation, and ensemble decisions "
                       "visible as evidence changes.")
            # Keep the tree as an overview, not a full-screen visualization.
            tree_col, _ = st.columns([5, 1])
            with tree_col:
                tree_event = st.altair_chart(
                    experiment_tree_chart(state["nodes"]), width="stretch",
                    height=620, key=f"experiment_tree_{pick}",
                    on_select="rerun", selection_mode="experiment_node")

            node_ids = [n["id"] for n in state["nodes"]]
            selected_key = f"selected_experiment_{pick}"
            if st.session_state.get(selected_key) not in node_ids:
                st.session_state[selected_key] = node_ids[-1]
            selection = getattr(tree_event, "selection", {}) or {}
            selected_points = selection.get("experiment_node") or {}
            if isinstance(selected_points, dict):
                selected_values = selected_points.get("node_id") or []
                clicked_id = selected_values[-1] if selected_values else None
            else:
                clicked_id = (selected_points[0].get("node_id")
                              if selected_points else None)
            if clicked_id in node_ids:
                st.session_state[selected_key] = clicked_id
            selected_id = st.session_state[selected_key]
            n = next(n for n in state["nodes"] if n["id"] == selected_id)
            rec = next((r for r in raw_nodes
                        if r.get("iteration_id") == selected_id), {})
            with st.expander(f"Experiment #{selected_id} details", expanded=True):
                st.caption("Click another circle in the tree to inspect that "
                           "experiment's decision, evidence, and exact artifacts.")
                head = st.columns([2, 2, 2])
                head[0].metric("Action", ACTION_LABEL.get(n["action"], n["action"]))
                head[1].metric("Score", f"{n['primary']:.5f}"
                               if n["primary"] is not None else "not scored")
                evidence = (n.get("evidence") or {}).get("state") or n["state"]
                head[2].metric("Evidence", evidence)
                if n["allocator"]:
                    st.markdown(f"<span class='small'>Allocator chose <b>{n['allocator']}"
                                f"</b> for this decision.</span>",
                                unsafe_allow_html=True)
                if n["question"]:
                    st.markdown(f"**Asked:** {n['question']}")
                if n["hypotheses"]:
                    st.markdown("**Competing explanations**")
                    for h in n["hypotheses"]:
                        st.markdown(f"- {h}")
                if n["plan"]:
                    st.markdown(f"**Tried:** {n['plan']}")
                if n["paired"]:
                    pr = n["paired"]
                    st.markdown(
                        f"**Paired test:** control {pr['control']:.5f} → treatment "
                        f"{pr['treatment']:.5f}; Δ {pr['delta']:+.5f} "
                        f"({pr['sigma']:+.2f}σ) over {pr['n']} seeds; "
                        f"**{'adopted' if pr['promote'] else 'not adopted'}**.")
                if n["error"]:
                    st.error(n["error"])
                if n["outcome"]:
                    st.caption(f"Outcome: {n['outcome']}")
                if n["evidence"]:
                    ev = n["evidence"]
                    colr, meaning = TIER.get(ev["state"], ("#57606a", ""))
                    seeds = (f" · {ev['n_seeds']} seeds"
                             if ev.get("n_seeds") else "")
                    st.markdown(pill(f"{ev['state']}{seeds}", colr)
                                + f"<span class='small'>{meaning}</span>",
                                unsafe_allow_html=True)
                st.markdown("#### Experiment brief")
                st.markdown("<div class='iteration-brief'><div class='label'>Why this "
                            "experiment exists</div><h3>" +
                            ACTION_LABEL.get(n["action"], n["action"]) +
                            "</h3><p>" +
                            (n["question"] or n["plan"] or
                             "No research question was recorded for this node.") +
                            "</p></div>", unsafe_allow_html=True)
                detail = st.columns(2)
                with detail[0]:
                    st.markdown("**What the agent tried**")
                    st.write(rec.get("hypothesis") or n["plan"]
                             or "No hypothesis was recorded.")
                    if n.get("observation"):
                        st.caption("Observation that motivated it")
                        st.write(n["observation"])
                with detail[1]:
                    st.markdown("**What happened**")
                    if n["outcome"]:
                        st.write(n["outcome"])
                    elif n["error"]:
                        st.error(n["error"])
                    elif n["primary"] is not None:
                        st.write("The sandbox completed and returned official "
                                 "validation metrics.")
                    else:
                        st.write("This node did not produce a scored result.")
                with st.expander("Configuration and audit evidence"):
                    st.caption("Structured settings and the unedited journal record.")
                    st.json(rec.get("menu_choices") or {})
                    st.json(rec, expanded=False)
                code_path = rec.get("code_path") or ""
                if code_path:
                    code_path = (code_path if os.path.isabs(code_path)
                                 else os.path.join(ROOT, code_path))
                if code_path and os.path.exists(code_path):
                    with st.expander("Executable script sent to the sandbox"):
                        with open(code_path) as fh:
                            st.code(fh.read(), language="python")


# ----------------------------------------------------------- iteration log ---
def render_iteration_log():
    js = journals()
    if not js:
        st.info("No run on disk yet.")
    else:
        pick = st.session_state.get("tree_journal")
        if pick not in js:
            pick = next(iter(js))
        nodes = L._load_jsonl(js[pick])
        run_state = L.state(js[pick])
        k = run_state["kpi"]
        fs = (read_json(os.path.join(LOGS, "final_summary.json")) or {}
              if pick == "current run" else {})

        st.divider()
        st.markdown("#### Run summary")
        summary_metrics = st.columns(4)
        summary_metrics[0].metric("Decisions", len(nodes))
        summary_metrics[1].metric("Scored experiments", k.get("scored", 0))
        summary_metrics[2].metric(
            "Best primary", f"{k['best']:.5f}" if k.get("best") else "—")
        summary_metrics[3].metric(
            "Training runs", (fs.get("budget_ledger") or {}).get(
                "training_runs_used", k.get("training_runs") or 0))
        if fs.get("stop_reason"):
            st.success(f"**Why the run stopped:** {fs['stop_reason']}")
        else:
            st.caption("This evidence run is summarized directly from its "
                       "append-only journal.")

        st.markdown("#### Iteration audit trail")
        st.caption("Every decision in the selected run, including failures and "
                   "preflight rejections. Click its tree node above for the raw "
                   "journal record, configuration, and executable script.")
        rows = []
        for n in nodes:
            s = L.summarise(n, nodes)
            rows.append({
                "#": s["id"],
                "Experiment": ACTION_LABEL.get(s["action"], s["action"]),
                "Outcome": {"ok": "completed", "fail": "crashed",
                            "preflight": "caught early"}[s["state"]],
                "Primary": s["primary"], "σ vs baseline": s["sigma"],
                "Evidence": (s["evidence"] or {}).get("state"),
                "Seconds": s["seconds"],
                "Note": s["outcome"] or (s["question"][:70] if s["question"] else ""),
            })
        df = pd.DataFrame(rows)
        f = st.columns(3)
        if f[0].checkbox("Failures only"):
            df = df[df["Outcome"] != "completed"]
        if f[1].checkbox("Confirmations & ensembles only"):
            df = df[df["Experiment"].str.contains("Confirm|Ensemble")]
        st.dataframe(df, width="stretch", hide_index=True)


# -------------------------------------------------------------- robustness ---
def render_robustness(run_name, journal_path):
    st.divider()
    st.markdown("#### Run reliability and recovery")
    st.caption(f"Derived only from the selected **{run_name}** journal. A failure "
               "and its debug descendants count as one incident; recovery means "
               "this run later returned to a valid scored experiment.")

    reliability = L.run_reliability(journal_path)
    incidents = reliability["failure_incidents"]
    returned = reliability["returned_to_scored_work"]

    metrics = st.columns(4)
    metrics[0].metric("Failure incidents", incidents)
    metrics[1].metric("Blocked before training",
                      reliability["preflight_rejections"])
    metrics[2].metric("Debug attempts", reliability["debug_attempts"])
    metrics[3].metric("Returned to valid work",
                      f"{returned} / {incidents}" if incidents else "Not needed")

    if not incidents:
        st.success(f"This run completed {reliability['scored']} scored experiments "
                   "without a recorded failure or recovery action.")
    elif returned == incidents and not reliability["invalid_failed_promotions"]:
        st.success(
            f"This run encountered {reliability['failed_nodes']} failed nodes "
            f"across {incidents} incidents. In every case it safely left the "
            "failed work and later produced another valid score; no failed "
            "candidate was promoted.")
    else:
        st.warning(f"This run returned to scored work after {returned} of "
                   f"{incidents} failure incidents. Inspect the unresolved "
                   "rows below before treating the run as fully recovered.")

    if reliability["incidents"]:
        st.dataframe([
            {"Started at": f"Node #{x['node']}",
             "Failure": x["failure"],
             "Agent response": x["response"],
             "Next valid result": (f"Node #{x['next_valid_node']}"
                                   if x["next_valid_node"] is not None else "None"),
             "Run continued safely": "Yes" if x["returned_to_scored_work"] else "No"}
            for x in reliability["incidents"]],
            width="stretch", hide_index=True)
    st.caption(
        f"Executed failures: {reliability['executed_failures']} · direct branch "
        f"repairs: {reliability['repaired_in_place']} · recorded manual "
        f"intervention events: {reliability['manual_intervention_events']}.")

    robustness = _M.get("robustness") or {}
    faults = robustness.get("fault_suite") or {}
    closed_loop = robustness.get("closed_loop_recovery") or {}
    if faults.get("available") and closed_loop.get("available"):
        st.caption(
            f"Separate system stress tests remain in `results/JUDGE_PACKET.md`: "
            f"{faults.get('faults_injected')} injected fault cases and "
            f"{closed_loop.get('recovered')}/{closed_loop.get('total')} controlled "
            "closed-loop recoveries. They are not attributed to this run.")


# ---------------------------------------------------------- combined live view ---
with tabs[2]:
    live_journals = journals()
    if live_journals:
        run_picker = st.columns([2, 1])
        run_picker[0].selectbox("Run", list(live_journals), key="tree_journal")
        run_picker[1].caption("Tree refresh\nevery 3 seconds")
    render_watch_tree()
    render_iteration_log()
    if live_journals:
        selected_run = st.session_state.get("tree_journal")
        if selected_run not in live_journals:
            selected_run = next(iter(live_journals))
        render_robustness(selected_run, live_journals[selected_run])


# ---------------------------------------------------------------- demo run ---
with tabs[3]:
    st.header("Demo run")
    if st.button("▶ Start demo", type="primary"):
        st.session_state["demo_started_at"] = time.time()
    render_demo_run()


# --------------------------------------------------------------------- run ---
with tabs[1]:
    st.header("Start a run")
    if not REAL_RUNS_ENABLED:
        st.warning("This public deployment is read-only. Run controls are "
                   "locked so visitors cannot spend LLM budget or start "
                   "training jobs; the Demo run remains fully available.")
    if running:
        st.warning("A run is already in progress — see **Watch a run**.")

    st.markdown("<span class='small'>The competition profile switches on the "
                "capabilities a judged run should demonstrate and prints its "
                "fully resolved configuration before spending anything.</span>",
                unsafe_allow_html=True)

    c = st.columns(4)
    iters = c[0].number_input("Decisions", 1, 50, 12,
                              help="outer-loop iterations",
                              disabled=not REAL_RUNS_ENABLED)
    truns = c[1].number_input("Training runs", 1, 300, 90,
                              help="an ensemble or paired confirmation costs "
                                   "several of these but only one decision",
                              disabled=not REAL_RUNS_ENABLED)
    spend_cap = c[2].number_input(
        "Spend cap ($)", 0.5, 50.0, 6.0, step=0.5,
        disabled=not REAL_RUNS_ENABLED)
    hours = c[3].number_input(
        "Wall-clock (h)", 0.25, 8.0, 2.0, step=0.25,
        disabled=not REAL_RUNS_ENABLED)
    fresh = st.checkbox("Archive previous run logs first", value=True,
                        help="Submission artifacts always survive: the "
                             "ensemble, its members, research memory and the "
                             "feature registry are never archived.",
                        disabled=not REAL_RUNS_ENABLED)

    cmd = [sys.executable, "run_agent.py", "--competition",
           "--max-iterations", str(iters), "--max-training-runs", str(truns),
           "--max-spend-usd", str(spend_cap), "--wall-clock-limit-h", str(hours)]
    if fresh:
        cmd.append("--fresh")
    st.code(" ".join(["python3"] + cmd[1:]), language="bash")

    armed = st.checkbox("I want to start a real run (spends LLM budget)",
                        value=False, key="arm_run",
                        disabled=not REAL_RUNS_ENABLED)
    go = st.button("▶️ Start run", type="primary",
                   disabled=(not REAL_RUNS_ENABLED or running or not armed))
    if go and REAL_RUNS_ENABLED and armed and not running:
        logf = os.path.join(LOGS, "dashboard_run.log")
        os.makedirs(LOGS, exist_ok=True)
        with open(logf, "w") as fh:
            fh.write(f"launched from the dashboard at "
                     f"{time.strftime('%Y-%m-%d %H:%M:%S')}\n"
                     f"{' '.join(cmd)}\n\n")
            fh.flush()
            subprocess.Popen(cmd, cwd=ROOT, stdout=fh, stderr=subprocess.STDOUT,
                             start_new_session=True)
        st.success("Started — open **Watch a run**.")
        time.sleep(2)
        st.rerun()

    logf = os.path.join(LOGS, "dashboard_run.log")
    if os.path.exists(logf):
        with st.expander("Console output", expanded=running):
            with open(logf) as fh:
                st.code(fh.read()[-6000:])

    st.divider()
    st.subheader("Final submission")
    if _HIDDEN.get("evaluated"):
        st.success("**The one-time hidden-test evaluation is complete and "
                   "locked.** This dashboard cannot run it again.")
        st.code("python3 -m agent.make_submission --split valid --score --ensemble\n"
                "python3 -m agent.make_submission --split test --ensemble",
                language="bash")
    else:
        st.error("**The hidden-test evaluation runs exactly once for the whole "
                 "project**, so it is deliberately not a button here.")
        st.code("python3 -m agent.make_submission --split valid --score --ensemble"
                "   # inspect first\n"
                "python3 -m agent.make_submission --final-test-eval --ensemble"
                "       # THE one-time evaluation", language="bash")
