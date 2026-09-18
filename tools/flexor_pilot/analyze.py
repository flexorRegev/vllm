"""Calibration analysis for the diffusion structured-read pilot (E1-E4).

Reads the per-(ticket, condition, draw, question) JSONL produced by the pilot runner and emits a
Markdown report. The pilot is ~90 rows per condition, so every number here is directional; each
table therefore carries its n and its error count.

Run:
    uv run --with pandas --with numpy --with scikit-learn python tools/flexor_pilot/analyze.py \
        <results.jsonl> [--out report.md]
"""

from __future__ import annotations

import argparse
import json
import math
from collections.abc import Callable, Iterable, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

CONDITIONS: tuple[str, ...] = ("k1", "k1_draws", "k2", "k4", "k8", "k4_never_accept")
QUESTIONS: tuple[str, ...] = ("urgent", "category", "tone")
DRAWS_CONDITION: str = "k1_draws"
MEAN_CONDITION: str = "k1_mean4"
COVERAGES: tuple[float, ...] = (1.0, 0.9, 0.75, 0.5)
ECE_BINS: int = 5
TRAJECTORY_FEATURES: tuple[str, ...] = (
    "maxprob",
    "margin",
    "label_entropy",
    "final_entropy_full",
    "flip_count",
    "stabilization_step",
    "traj_mean_entropy",
)
REQUIRED_FIELDS: tuple[str, ...] = (
    "ticket_id",
    "condition",
    "question_id",
    "labels",
    "gt",
    "final_probs",
)


# ---------------------------------------------------------------- loading


def load_records(path: Path) -> tuple[list[dict[str, Any]], list[str]]:
    """Parse JSONL, dropping rows that cannot be interpreted; returns (rows, problems)."""
    rows: list[dict[str, Any]] = []
    problems: list[str] = []
    for lineno, line in enumerate(path.read_text().splitlines(), start=1):
        text = line.strip()
        if not text:
            continue
        try:
            record = json.loads(text)
        except json.JSONDecodeError as exc:
            problems.append("line %d: not valid JSON (%s)" % (lineno, exc.msg))
            continue
        missing = [field for field in REQUIRED_FIELDS if field not in record]
        if missing:
            problems.append("line %d: missing fields %s" % (lineno, ", ".join(missing)))
            continue
        probs = record["final_probs"]
        if len(probs) != len(record["labels"]):
            problems.append("line %d: final_probs length != labels length" % lineno)
            continue
        if abs(float(sum(probs)) - 1.0) > 1e-3:
            problems.append("line %d: final_probs sum to %.4f, not 1" % (lineno, float(sum(probs))))
        rows.append(record)
    return rows, problems


def coverage_report(rows: Sequence[dict[str, Any]]) -> tuple[pd.DataFrame, list[str]]:
    """Every (ticket, condition, draw) must carry all three questions; report gaps rather than crash."""
    frame = pd.DataFrame(
        [
            {
                "ticket_id": r["ticket_id"],
                "condition": r["condition"],
                "draw": int(r.get("draw", 0) or 0),
                "question_id": r["question_id"],
            }
            for r in rows
        ]
    )
    gaps: list[str] = []
    if frame.empty:
        return frame, ["no rows parsed"]
    tickets = sorted(frame["ticket_id"].unique())
    summary: list[dict[str, Any]] = []
    for condition, part in frame.groupby("condition"):
        per_slot = part.groupby(["ticket_id", "draw"])["question_id"].nunique()
        incomplete = per_slot[per_slot < len(QUESTIONS)]
        missing_tickets = sorted(set(tickets) - set(part["ticket_id"].unique()))
        summary.append(
            {
                "condition": condition,
                "tickets": part["ticket_id"].nunique(),
                "draws": part["draw"].nunique(),
                "rows": len(part),
                "slots_missing_questions": int(len(incomplete)),
                "tickets_absent": len(missing_tickets),
            }
        )
        for (ticket, draw), count in incomplete.items():
            gaps.append("%s / %s / draw %d: only %d of 3 questions" % (condition, ticket, draw, count))
        if missing_tickets:
            gaps.append("%s: no rows for tickets %s" % (condition, ", ".join(missing_tickets)))
    unknown = sorted(set(frame["condition"].unique()) - set(CONDITIONS))
    if unknown:
        gaps.append("unexpected conditions present: %s" % ", ".join(unknown))
    return pd.DataFrame(summary).sort_values("condition"), gaps


# ---------------------------------------------------------------- derivations


def entropy(probs: Sequence[float]) -> float:
    array = np.asarray(probs, dtype=float)
    array = array[array > 0.0]
    return float(-np.sum(array * np.log(array)))


def trajectory_features(per_step_probs: Sequence[Sequence[float]] | None) -> dict[str, float]:
    if not per_step_probs:
        return {
            "flip_count": math.nan,
            "stabilization_step": math.nan,
            "traj_mean_entropy": math.nan,
            "step1_maxprob": math.nan,
            "step1_argmax": math.nan,
        }
    argmaxes = [int(np.argmax(step)) for step in per_step_probs]
    flips = sum(1 for i in range(1, len(argmaxes)) if argmaxes[i] != argmaxes[i - 1])
    final = argmaxes[-1]
    # first 1-indexed step from which the argmax never changes again
    stabilization = len(argmaxes)
    for index in range(len(argmaxes) - 1, -1, -1):
        if argmaxes[index] != final:
            break
        stabilization = index + 1
    return {
        "flip_count": float(flips),
        "stabilization_step": float(stabilization),
        "traj_mean_entropy": float(np.mean([entropy(step) for step in per_step_probs])),
        "step1_maxprob": float(np.max(per_step_probs[0])),
        "step1_argmax": float(argmaxes[0]),
    }


def derive_row(record: dict[str, Any]) -> dict[str, Any]:
    labels: list[str] = list(record["labels"])
    probs = np.asarray(record["final_probs"], dtype=float)
    order = np.argsort(probs)[::-1]
    top = int(order[0])
    second = float(probs[order[1]]) if len(order) > 1 else 0.0
    gt = record["gt"]
    alt = record.get("alt")
    pred = labels[top]
    row: dict[str, Any] = {
        "ticket_id": record["ticket_id"],
        "borderline": bool(record.get("borderline", False)),
        "condition": record["condition"],
        "draw": int(record.get("draw", 0) or 0),
        "steps": int(record.get("steps", 0) or 0),
        "question_id": record["question_id"],
        "labels": labels,
        "gt": gt,
        "alt": alt,
        "final_probs": probs.tolist(),
        "pred": pred,
        "correct_strict": bool(pred == gt),
        "correct_lenient": bool(pred == gt or (alt is not None and pred == alt)),
        "maxprob": float(probs[top]),
        "margin": float(probs[top] - second),
        "label_entropy": entropy(probs),
        "final_entropy_full": float(record.get("final_entropy_full", math.nan)),
        "label_mass": float(record.get("label_mass", math.nan)),
        "argmax_is_label": bool(record.get("argmax_is_label", True)),
        "wall_ms": float(record.get("wall_ms", math.nan)),
        "agreement": math.nan,
        "draw_std": math.nan,
    }
    row.update(trajectory_features(record.get("per_step_probs")))
    return row


def build_mean_condition(frame: pd.DataFrame) -> pd.DataFrame:
    """k1_mean4: probability-average the 4 independent k1 draws (E3's equal-compute arm vs k4)."""
    source = frame[frame["condition"] == DRAWS_CONDITION]
    if source.empty:
        return pd.DataFrame(columns=frame.columns)
    built: list[dict[str, Any]] = []
    for (ticket, question), part in source.groupby(["ticket_id", "question_id"]):
        labels = list(part.iloc[0]["labels"])
        stacked = np.vstack([np.asarray(p, dtype=float) for p in part["final_probs"]])
        mean_probs = stacked.mean(axis=0)
        top = int(np.argmax(mean_probs))
        per_draw_argmax = [int(np.argmax(p)) for p in stacked]
        record: dict[str, Any] = {
            "ticket_id": ticket,
            "borderline": bool(part.iloc[0]["borderline"]),
            "condition": MEAN_CONDITION,
            "draw": 0,
            "steps": int(part.iloc[0]["steps"]),
            "question_id": question,
            "labels": labels,
            "gt": part.iloc[0]["gt"],
            "alt": part.iloc[0]["alt"],
            "final_probs": mean_probs.tolist(),
            "final_entropy_full": float(part["final_entropy_full"].mean()),
            "label_mass": float(part["label_mass"].mean()),
            "argmax_is_label": bool(part["argmax_is_label"].all()),
            "wall_ms": float(part["wall_ms"].sum()),
            "per_step_probs": None,
            # agreement = share of draws whose own argmax matches the averaged argmax
            "agreement": float(np.mean([a == top for a in per_draw_argmax])),
            # spread of the probability the draws put on the averaged winner
            "draw_std": float(np.std(stacked[:, top])),
        }
        derived = derive_row(
            {
                **{k: record[k] for k in ("ticket_id", "condition", "question_id", "labels", "gt", "final_probs")},
                "borderline": record["borderline"],
                "alt": record["alt"],
                "draw": 0,
                "steps": record["steps"],
                "final_entropy_full": record["final_entropy_full"],
                "label_mass": record["label_mass"],
                "argmax_is_label": record["argmax_is_label"],
                "wall_ms": record["wall_ms"],
                "per_step_probs": None,
            }
        )
        derived["agreement"] = record["agreement"]
        derived["draw_std"] = record["draw_std"]
        built.append(derived)
    return pd.DataFrame(built)


# ---------------------------------------------------------------- metrics


def safe_auroc(scores: np.ndarray, correct: np.ndarray) -> float:
    mask = ~np.isnan(scores)
    if mask.sum() < 2 or len(set(correct[mask].tolist())) < 2:
        return math.nan
    return float(roc_auc_score(correct[mask], scores[mask]))


def multiclass_brier(frame: pd.DataFrame, correct_column: str) -> float:
    """Multi-class Brier: mean squared error between the label distribution and the one-hot target."""
    total = 0.0
    for _, row in frame.iterrows():
        probs = np.asarray(row["final_probs"], dtype=float)
        target = np.zeros_like(probs)
        accepted = {row["gt"]} if correct_column == "correct_strict" else {row["gt"], row["alt"]}
        for index, label in enumerate(row["labels"]):
            if label in accepted:
                target[index] = 1.0
        if target.sum() > 0:
            target = target / target.sum()
        total += float(np.sum((probs - target) ** 2))
    return total / max(len(frame), 1)


def expected_calibration_error(confidence: np.ndarray, correct: np.ndarray, bins: int = ECE_BINS) -> float:
    """Equal-mass binning: with ~90 rows, equal-width bins leave bins nearly empty."""
    if len(confidence) == 0:
        return math.nan
    order = np.argsort(confidence)
    chunks = np.array_split(order, min(bins, len(confidence)))
    error = 0.0
    for chunk in chunks:
        if len(chunk) == 0:
            continue
        error += len(chunk) / len(confidence) * abs(confidence[chunk].mean() - correct[chunk].mean())
    return float(error)


def selective_accuracy(scores: np.ndarray, correct: np.ndarray, coverage: float) -> float:
    mask = ~np.isnan(scores)
    scores, correct = scores[mask], correct[mask]
    if len(scores) == 0:
        return math.nan
    keep = max(1, int(round(coverage * len(scores))))
    order = np.argsort(scores)[::-1][:keep]
    return float(correct[order].mean())


def score_columns(frame: pd.DataFrame) -> dict[str, np.ndarray]:
    """Ranking scores; all oriented so that higher = more likely correct."""
    scores: dict[str, np.ndarray] = {
        "maxprob": frame["maxprob"].to_numpy(dtype=float),
        "margin": frame["margin"].to_numpy(dtype=float),
        "-label_entropy": -frame["label_entropy"].to_numpy(dtype=float),
        "-final_entropy_full": -frame["final_entropy_full"].to_numpy(dtype=float),
        "agreement": frame["agreement"].to_numpy(dtype=float),
        "-flip_count": -frame["flip_count"].to_numpy(dtype=float),
        "-stabilization_step": -frame["stabilization_step"].to_numpy(dtype=float),
    }
    return {name: values for name, values in scores.items() if not np.all(np.isnan(values))}


def condition_metrics(frame: pd.DataFrame, correct_column: str) -> dict[str, Any]:
    correct = frame[correct_column].to_numpy(dtype=float)
    confidence = frame["maxprob"].to_numpy(dtype=float)
    row: dict[str, Any] = {
        "n": len(frame),
        "errors": int((1 - correct).sum()),
        "accuracy": float(correct.mean()) if len(frame) else math.nan,
        "auroc_maxprob": safe_auroc(confidence, correct),
        "brier_multiclass": multiclass_brier(frame, correct_column),
        "ece_maxprob": expected_calibration_error(confidence, correct),
    }
    for coverage in COVERAGES:
        key = "sel@%d%%" % int(coverage * 100)
        row[key + "_maxprob"] = selective_accuracy(confidence, correct, coverage)
        row[key + "_invent"] = selective_accuracy(-frame["label_entropy"].to_numpy(dtype=float), correct, coverage)
    return row


def metrics_table(frame: pd.DataFrame, correct_column: str, group: Sequence[str]) -> pd.DataFrame:
    rows = [{**dict(zip(group, key if isinstance(key, tuple) else (key,))), **condition_metrics(part, correct_column)} for key, part in frame.groupby(list(group))]
    return pd.DataFrame(rows)


def alternative_score_table(frame: pd.DataFrame, correct_column: str) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for condition, part in frame.groupby("condition"):
        correct = part[correct_column].to_numpy(dtype=float)
        for name, values in score_columns(part).items():
            entry: dict[str, Any] = {
                "condition": condition,
                "score": name,
                "n": int((~np.isnan(values)).sum()),
                "errors": int((1 - correct).sum()),
                "auroc": safe_auroc(values, correct),
            }
            for coverage in COVERAGES:
                entry["sel@%d%%" % int(coverage * 100)] = selective_accuracy(values, correct, coverage)
            rows.append(entry)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- temperature scaling


def scaled_probs(logits: np.ndarray, temperature: float) -> np.ndarray:
    shifted = logits / temperature
    shifted = shifted - shifted.max(axis=1, keepdims=True)
    exponentiated = np.exp(shifted)
    return exponentiated / exponentiated.sum(axis=1, keepdims=True)


def nll(logits: np.ndarray, targets: np.ndarray, temperature: float) -> float:
    probs = scaled_probs(logits, temperature)
    picked = probs[np.arange(len(targets)), targets]
    return float(-np.mean(np.log(np.clip(picked, 1e-12, None))))


def fit_temperature(logits: np.ndarray, targets: np.ndarray) -> float:
    """Grid then local refinement; one parameter over <=90 rows does not warrant an optimiser dependency."""
    grid = np.concatenate([np.linspace(0.2, 3.0, 57), np.linspace(3.0, 10.0, 15)])
    best = float(min(grid, key=lambda t: nll(logits, targets, float(t))))
    low, high = max(0.05, best - 0.1), best + 0.1
    fine = np.linspace(low, high, 41)
    return float(min(fine, key=lambda t: nll(logits, targets, float(t))))


def padded_logits(frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Pad label distributions to a common width so questions with different arities pool into one fit."""
    width = max(len(row) for row in frame["final_probs"])
    logits = np.full((len(frame), width), -50.0)
    targets = np.zeros(len(frame), dtype=int)
    for position, (_, row) in enumerate(frame.iterrows()):
        probs = np.clip(np.asarray(row["final_probs"], dtype=float), 1e-12, None)
        logits[position, : len(probs)] = np.log(probs)
        targets[position] = row["labels"].index(row["gt"]) if row["gt"] in row["labels"] else 0
    return logits, targets


def temperature_table(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for condition, part in frame.groupby("condition"):
        part = part.reset_index(drop=True)
        logits, targets = padded_logits(part)
        correct = part["correct_strict"].to_numpy(dtype=float)
        before = scaled_probs(logits, 1.0).max(axis=1)
        temperature = fit_temperature(logits, targets)
        after = scaled_probs(logits, temperature).max(axis=1)
        loo = np.zeros(len(part))
        for ticket in part["ticket_id"].unique():
            held = (part["ticket_id"] == ticket).to_numpy()
            fitted = fit_temperature(logits[~held], targets[~held]) if (~held).sum() > 0 else temperature
            loo[held] = scaled_probs(logits[held], fitted).max(axis=1)
        rows.append(
            {
                "condition": condition,
                "n": len(part),
                "errors": int((1 - correct).sum()),
                "T": temperature,
                "nll_before": nll(logits, targets, 1.0),
                "nll_after": nll(logits, targets, temperature),
                "ece_before": expected_calibration_error(before, correct),
                "ece_after": expected_calibration_error(after, correct),
                "ece_after_loo_ticket": expected_calibration_error(loo, correct),
            }
        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- trajectory views (E2)


def step_confidence_table(frame: pd.DataFrame, raw: Sequence[dict[str, Any]]) -> pd.DataFrame:
    by_key = {(r["ticket_id"], r["condition"], int(r.get("draw", 0) or 0), r["question_id"]): r for r in raw}
    rows: list[dict[str, Any]] = []
    for condition, part in frame.groupby("condition"):
        curves: dict[bool, list[list[float]]] = {True: [], False: []}
        for _, row in part.iterrows():
            record = by_key.get((row["ticket_id"], condition, int(row["draw"]), row["question_id"]))
            steps = record.get("per_step_probs") if record else None
            if not steps:
                continue
            curves[bool(row["correct_strict"])].append([float(np.max(step)) for step in steps])
        if not curves[True] and not curves[False]:
            continue
        depth = max(len(c) for group in curves.values() for c in group)
        for step in range(depth):
            entry: dict[str, Any] = {"condition": condition, "step": step + 1}
            for flag, key in ((True, "correct"), (False, "incorrect")):
                values = [c[step] for c in curves[flag] if len(c) > step]
                entry["mean_maxprob_" + key] = float(np.mean(values)) if values else math.nan
                entry["n_" + key] = len(values)
            rows.append(entry)
    return pd.DataFrame(rows)


def flip_outcome_table(frame: pd.DataFrame, raw: Sequence[dict[str, Any]]) -> pd.DataFrame:
    by_key = {(r["ticket_id"], r["condition"], int(r.get("draw", 0) or 0), r["question_id"]): r for r in raw}
    rows: list[dict[str, Any]] = []
    for condition, part in frame.groupby("condition"):
        changed = corrections = regressions = neutral = trajectories = 0
        for _, row in part.iterrows():
            record = by_key.get((row["ticket_id"], condition, int(row["draw"]), row["question_id"]))
            steps = record.get("per_step_probs") if record else None
            if not steps:
                continue
            trajectories += 1
            first_label = row["labels"][int(np.argmax(steps[0]))]
            if first_label == row["pred"]:
                continue
            changed += 1
            step1_correct = first_label == row["gt"]
            if row["correct_strict"] and not step1_correct:
                corrections += 1
            elif not row["correct_strict"] and step1_correct:
                regressions += 1
            else:
                neutral += 1
        if trajectories == 0:
            continue
        rows.append(
            {
                "condition": condition,
                "n_trajectories": trajectories,
                "step1_differs_from_final": changed,
                "corrections": corrections,
                "regressions": regressions,
                "wrong_to_wrong": neutral,
            }
        )
    return pd.DataFrame(rows)


def borderline_table(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (condition, borderline), part in frame.groupby(["condition", "borderline"]):
        rows.append(
            {
                "condition": condition,
                "group": "borderline" if borderline else "clear",
                "n": len(part),
                "errors": int((~part["correct_strict"]).sum()),
                "mean_maxprob": float(part["maxprob"].mean()),
                "mean_label_entropy": float(part["label_entropy"].mean()),
                "accuracy_strict": float(part["correct_strict"].mean()),
                "accuracy_lenient": float(part["correct_lenient"].mean()),
            }
        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- E4 logistic regression


def loo_logistic_table(frame: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    rows: list[dict[str, Any]] = []
    notes: list[str] = []
    for condition, part in frame.groupby("condition"):
        usable = part.dropna(subset=list(TRAJECTORY_FEATURES))
        if usable.empty:  # conditions without trajectories have no flip/stabilization features
            continue
        correct = usable["correct_strict"].to_numpy(dtype=int)
        errors = int((1 - correct).sum())
        if errors < 5:
            notes.append("%s: only %d errors, LOO logistic regression skipped" % (condition, errors))
            continue
        features = usable[list(TRAJECTORY_FEATURES)].to_numpy(dtype=float)
        predictions = np.zeros(len(usable))
        for ticket in usable["ticket_id"].unique():
            held = (usable["ticket_id"] == ticket).to_numpy()
            if len(set(correct[~held].tolist())) < 2:
                predictions[held] = math.nan
                continue
            model = Pipeline([("scale", StandardScaler()), ("lr", LogisticRegression(max_iter=1000, C=1.0))])
            model.fit(features[~held], correct[~held])
            predictions[held] = model.predict_proba(features[held])[:, 1]
        rows.append(
            {
                "condition": condition,
                "n": len(usable),
                "errors": errors,
                "auroc_loo_logreg": safe_auroc(predictions, correct.astype(float)),
                "auroc_maxprob": safe_auroc(usable["maxprob"].to_numpy(dtype=float), correct.astype(float)),
            }
        )
    return pd.DataFrame(rows), notes


# ---------------------------------------------------------------- report


def to_markdown(frame: pd.DataFrame, empty_note: str = "_no rows_") -> str:
    if frame is None or frame.empty:
        return empty_note
    def cell(value: Any) -> str:
        if value is None or (isinstance(value, float) and math.isnan(value)):
            return "n/a"
        if isinstance(value, (float, np.floating)):
            return "%.3f" % round(float(value), 3)
        return str(value)

    headers = [str(c) for c in frame.columns]
    body = [[cell(value) for value in row] for row in frame.itertuples(index=False, name=None)]
    widths = [max(len(headers[i]), *(len(row[i]) for row in body)) if body else len(headers[i]) for i in range(len(headers))]
    lines = ["| " + " | ".join(h.ljust(widths[i]) for i, h in enumerate(headers)) + " |"]
    lines.append("| " + " | ".join("-" * widths[i] for i in range(len(headers))) + " |")
    lines.extend("| " + " | ".join(row[i].ljust(widths[i]) for i in range(len(headers))) + " |" for row in body)
    return "\n".join(lines)


def section(title: str, body: str) -> str:
    return "## %s\n\n%s\n" % (title, body)


def build_report(frame: pd.DataFrame, raw: Sequence[dict[str, Any]], coverage: pd.DataFrame, gaps: Iterable[str], problems: Iterable[str]) -> str:
    parts: list[str] = ["# Diffusion structured-read pilot: calibration report\n"]
    parts.append(
        "n is small (30 tickets x 3 questions per condition). Treat every number as directional: "
        "AUROC and ECE over ~90 rows with a handful of errors carry wide intervals.\n"
    )

    issues = list(problems) + list(gaps)
    parts.append(
        section(
            "1. Coverage",
            to_markdown(coverage)
            + "\n\n"
            + ("**Gaps**\n\n" + "\n".join("- " + g for g in issues) if issues else "No parse errors or coverage gaps."),
        )
    )

    for column, name in (("correct_strict", "strict"), ("correct_lenient", "lenient")):
        parts.append(section("2. Per-condition metrics (%s)" % name, to_markdown(metrics_table(frame, column, ["condition"]))))
    parts.append(
        section(
            "3. Per-question metrics (strict)",
            to_markdown(metrics_table(frame, "correct_strict", ["condition", "question_id"])),
        )
    )
    for column, name in (("correct_strict", "strict"), ("correct_lenient", "lenient")):
        parts.append(
            section(
                "4. Alternative ranking scores (%s)" % name,
                "AUROC and selective accuracy only; ECE is undefined for non-probability scores.\n\n"
                + to_markdown(alternative_score_table(frame, column)),
            )
        )
    parts.append(
        section(
            "5. Temperature scaling (strict, pooled label logits per condition)",
            "One T per condition fit on log(final_probs) by NLL against gt. The leave-one-ticket-out column "
            "shows how much of the in-sample ECE gain is fitting noise.\n\n" + to_markdown(temperature_table(frame)),
        )
    )
    parts.append(
        section(
            "6. E2: confidence by step",
            "Separating correct/incorrect curves = the trajectory carries information; both rising together = sharpening only.\n\n"
            + to_markdown(step_confidence_table(frame, raw), "_no trajectory rows_")
            + "\n\n"
            + to_markdown(flip_outcome_table(frame, raw), "_no trajectory rows_"),
        )
    )
    parts.append(section("7. Borderline vs clear (strict accuracy, both confidences)", to_markdown(borderline_table(frame))))

    pair = frame[frame["condition"].isin([MEAN_CONDITION, "k4"])]
    parts.append(
        section(
            "8. E3: steps vs draws at equal compute",
            "4 x K=1 averaged (k1_mean4) against 1 x K=4.\n\n"
            + to_markdown(
                metrics_table(pair, "correct_strict", ["condition"])[
                    ["condition", "n", "errors", "accuracy", "auroc_maxprob", "ece_maxprob", "sel@75%_maxprob"]
                ]
                if not pair.empty
                else pair
            ),
        )
    )
    table, notes = loo_logistic_table(frame)
    parts.append(
        section(
            "9. E4: leave-one-ticket-out logistic regression",
            "Features: " + ", ".join(TRAJECTORY_FEATURES) + ". Trajectory features earn their place only if "
            "LOO AUROC beats maxprob alone.\n\n" + to_markdown(table, "_no condition had usable trajectory features_")
            + ("\n\n" + "\n".join("- " + n for n in notes) if notes else ""),
        )
    )
    return "\n".join(parts)


def main() -> None:
    parser = argparse.ArgumentParser(description="Calibration analysis for the diffusion structured-read pilot")
    parser.add_argument("results", type=Path, help="JSONL of per-(ticket, condition, draw, question) records")
    parser.add_argument("--out", type=Path, default=Path("tools/flexor_pilot/results/report.md"))
    args = parser.parse_args()

    raw, problems = load_records(args.results)
    if not raw:
        raise ValueError("no usable records in the results file")
    coverage, gaps = coverage_report(raw)
    frame = pd.DataFrame([derive_row(record) for record in raw])
    frame = pd.concat([frame, build_mean_condition(frame)], ignore_index=True)

    report = build_report(frame, raw, coverage, gaps, problems)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(report)
    print(report)


if __name__ == "__main__":
    main()
