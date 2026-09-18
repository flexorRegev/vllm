"""Synthetic fixture matching the pilot JSONL contract, for exercising analyze.py.

Planted effects, so the report can be checked against known ground truth:
  - k1, k2, k4, k8, k1_draws: maxprob is informative (correct rows get higher confidence).
  - k4_never_accept: uniformly overconfident, so AUROC ~= 0.5 and ECE is large.
  - accuracy rises with K; borderline tickets are harder and less confident.
  - trajectories contain both corrections (step 1 wrong -> final right) and regressions.

Run: uv run --with numpy python tools/flexor_pilot/make_fixture.py [--out path.jsonl]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

QUESTION_LABELS: dict[str, list[str]] = {
    "urgent": ["yes", "no"],
    "category": ["billing", "outage", "feature_request", "how_to"],
    "tone": ["calm", "annoyed", "furious"],
}
CONDITION_STEPS: dict[str, int] = {"k1": 1, "k1_draws": 1, "k2": 2, "k4": 4, "k8": 8, "k4_never_accept": 4}
CONDITION_ACCURACY: dict[str, float] = {
    "k1": 0.72,
    "k1_draws": 0.72,
    "k2": 0.78,
    "k4": 0.84,
    "k8": 0.86,
    "k4_never_accept": 0.80,
}


def dirichlet_probs(rng: np.random.Generator, size: int, winner: int, confidence: float) -> list[float]:
    alpha = np.full(size, 1.0)
    alpha[winner] = confidence
    probs = rng.dirichlet(alpha)
    if int(np.argmax(probs)) != winner:  # keep the intended argmax
        probs[winner], probs[int(np.argmax(probs))] = probs[int(np.argmax(probs))], probs[winner]
    return [float(p) for p in probs]


def make_trajectory(rng: np.random.Generator, size: int, final: list[float], steps: int, flip_to: int | None) -> list[list[float]]:
    trajectory: list[list[float]] = []
    final_array = np.asarray(final)
    for step in range(steps):
        weight = (step + 1) / steps
        if step == 0 and flip_to is not None:
            base = dirichlet_probs(rng, size, flip_to, 3.0)
            trajectory.append(base)
            continue
        noise = rng.dirichlet(np.full(size, 2.0))
        blended = weight * final_array + (1 - weight) * noise
        trajectory.append([float(p) for p in blended / blended.sum()])
    trajectory[-1] = [float(p) for p in final_array]
    return trajectory


def build(rng: np.random.Generator, tickets: list[dict[str, Any]]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for condition, steps in CONDITION_STEPS.items():
        draws = 4 if condition == "k1_draws" else 1
        for ticket in tickets:
            for question, labels in QUESTION_LABELS.items():
                gt = ticket["gt"][question]
                alt = ticket.get("alt", {}).get(question)
                for draw in range(draws):
                    base = CONDITION_ACCURACY[condition] - (0.18 if ticket["borderline"] else 0.0)
                    is_correct = bool(rng.random() < base)
                    winner = labels.index(gt)
                    if not is_correct:
                        others = [i for i in range(len(labels)) if i != winner]
                        winner = int(rng.choice(others))
                    if condition == "k4_never_accept":
                        confidence = 14.0  # confident whether right or wrong -> uninformative maxprob
                    else:
                        confidence = (11.0 if is_correct else 3.2) - (2.0 if ticket["borderline"] else 0.0)
                    probs = dirichlet_probs(rng, len(labels), winner, max(confidence, 1.2))
                    trajectory = None
                    if steps > 1:
                        flip_to = None
                        if rng.random() < 0.35:
                            flip_to = int(rng.choice([i for i in range(len(labels)) if i != winner]))
                        trajectory = make_trajectory(rng, len(labels), probs, steps, flip_to)
                    label_mass = float(min(1.0, 0.82 + 0.17 * rng.random()))
                    records.append(
                        {
                            "ticket_id": ticket["id"],
                            "borderline": ticket["borderline"],
                            "condition": condition,
                            "draw": draw,
                            "steps": steps,
                            "question_id": question,
                            "labels": labels,
                            "gt": gt,
                            "alt": alt,
                            "final_probs": probs,
                            "final_entropy_full": float(-sum(p * np.log(p) for p in probs) + 1.5 * (1 - label_mass)),
                            "label_mass": label_mass,
                            "argmax_is_label": True,
                            "per_step_probs": trajectory,
                            "per_step_entropy_full": (
                                [float(-sum(p * np.log(max(p, 1e-12)) for p in step)) for step in trajectory] if trajectory else None
                            ),
                            "wall_ms": float(120 * steps + rng.normal(0, 12)),
                        }
                    )
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a synthetic pilot results fixture")
    parser.add_argument("--tickets", type=Path, default=Path("tools/flexor_pilot/tickets.json"))
    parser.add_argument("--out", type=Path, default=Path("tools/flexor_pilot/results/fixture_synthetic.jsonl"))
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    tickets = json.loads(args.tickets.read_text())["tickets"]
    records = build(np.random.default_rng(args.seed), tickets)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(json.dumps(record) for record in records) + "\n")
    print("wrote %d records to %s" % (len(records), args.out))


if __name__ == "__main__":
    main()
