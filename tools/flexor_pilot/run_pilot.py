"""Pilot sweep of DiffusionGemma structured reads over a labelled ticket set.

Same harness as ``tools/flexor_modal_e2e.py``: the fork overlaid on the vLLM
nightly image, the FP8 checkpoint on one H100, ``structured_server.py`` in
front of it. Every ticket is read under each step / noise-draw condition and
one JSONL record per (ticket, condition, draw, question) comes back.

    uvx --from modal modal run tools/flexor_pilot/run_pilot.py
"""

from __future__ import annotations

import json
import pathlib
import time

import modal

FORK_ROOT = pathlib.Path(__file__).resolve().parent.parent.parent
PILOT_DIR = FORK_ROOT / "tools" / "flexor_pilot"
MODEL = "RedHatAI/diffusiongemma-26B-A4B-it-FP8-dynamic"
CANVAS = 64
EXAMPLE_DIR = "/fork/examples/features/diffusion_reads"
DATASET = "/fork/pilot/tickets.json"

# Trajectory is export only, so it is on everywhere: every condition then
# reports its label distribution from the same place, the last step's
# temperature-1 label logprobs.
CONDITIONS = {
    "k1": {"steps": 1, "samples": 1, "trajectory": True},
    "k1_draws": {"steps": 1, "samples": 1, "trajectory": True},
    "k2": {"steps": 2, "fixed_steps": True, "trajectory": True, "samples": 1},
    "k4": {"steps": 4, "fixed_steps": True, "trajectory": True, "samples": 1},
    "k8": {"steps": 8, "fixed_steps": True, "trajectory": True, "samples": 1},
    "k4_never_accept": {"steps": 4, "fixed_steps": True, "trajectory": True,
                        "slots_never_accept": True, "samples": 1},
}
DRAWS = {"k1_draws": 4}
BASE_SEED = 20260918

image = (
    modal.Image.from_registry("vllm/vllm-openai:nightly", add_python=None)
    .entrypoint([])
    .run_commands(
        "set -e; PY=$(command -v python3 || echo /opt/venv/bin/python3); echo using $PY; "
        "ln -sf $PY /usr/local/bin/python"
    )
    .add_local_dir(str(FORK_ROOT / "vllm"), remote_path="/fork/vllm", ignore=["**/__pycache__/**", "**/*.pyc"])
    .add_local_dir(str(FORK_ROOT / "examples"), remote_path="/fork/examples", ignore=["**/__pycache__/**", "**/*.pyc"])
    .add_local_dir(str(PILOT_DIR), remote_path="/fork/pilot", ignore=["**/__pycache__/**", "results/**"])
)

app = modal.App("flexor-diffusion-pilot", image=image)
hf_cache = modal.Volume.from_name("diffusiongemma-hf-cache", create_if_missing=True)


@app.function(
    gpu="H100",
    timeout=90 * 60,
    secrets=[modal.Secret.from_name("hf-access-token")],
    volumes={"/root/.cache/huggingface": hf_cache},
)
def run_pilot() -> str:
    import math
    import os
    import shutil
    import subprocess
    import sys
    import urllib.error
    import urllib.request

    import vllm as installed_vllm

    installed_dir = os.path.dirname(installed_vllm.__file__)
    copied = 0
    for root, _dirs, files in os.walk(installed_dir):
        rel = os.path.relpath(root, installed_dir)
        dest_dir = os.path.normpath(os.path.join("/fork/vllm", rel))
        for name in files:
            dest = os.path.join(dest_dir, name)
            if os.path.exists(dest) or name.endswith(".pyc"):
                continue
            os.makedirs(dest_dir, exist_ok=True)
            shutil.copy(os.path.join(root, name), dest)
            copied += 1
    print("filled %d files from the installed package" % copied, flush=True)

    data = json.load(open(DATASET))
    schema_questions = data["schema"]["questions"]
    tickets = data["tickets"]
    env = dict(os.environ, PYTHONPATH="/fork", HF_HOME="/root/.cache/huggingface")

    def post(url: str, body: dict, timeout: int = 900) -> dict:
        req = urllib.request.Request(
            url, data=json.dumps(body).encode(), headers={"content-type": "application/json"}
        )
        return json.load(urllib.request.urlopen(req, timeout=timeout))

    def wait_health(url: str, log_path: str, proc: subprocess.Popen, budget_s: int) -> float:
        started = time.time()
        while time.time() - started < budget_s:
            if proc.poll() is not None:
                print(open(log_path).read()[-8000:], flush=True)
                raise RuntimeError("server exited before becoming healthy")
            try:
                with urllib.request.urlopen(url, timeout=5) as r:
                    if r.status == 200:
                        return time.time() - started
            except (urllib.error.URLError, OSError):
                pass
            time.sleep(10)
        raise RuntimeError("server did not become healthy in budget")

    server_log = "/tmp/vllm_server.log"
    server_cmd = [
        sys.executable, "-m", "vllm.entrypoints.cli.main", "serve", MODEL,
        "--diffusion-config", json.dumps({"canvas_length": CANVAS}),
        "--max-logprobs", "32", "--enable-prefix-caching", "--async-scheduling",
        "--attention-backend", "TRITON_ATTN", "--max-num-seqs", "32", "--port", "8000",
        "--served-model-name", "dgemma",
    ]
    with open(server_log, "w") as fh:
        server = subprocess.Popen(server_cmd, cwd="/fork", env=env, stdout=fh, stderr=subprocess.STDOUT)
    interposer = None
    records: list[dict] = []
    try:
        boot_s = wait_health("http://127.0.0.1:8000/health", server_log, server, 25 * 60)
        print("server healthy after %.1fs" % boot_s, flush=True)
        interposer_log = "/tmp/structured_server.log"
        with open(interposer_log, "w") as fh:
            interposer = subprocess.Popen(
                [sys.executable, EXAMPLE_DIR + "/structured_server.py", "--upstream", "http://127.0.0.1:8000",
                 "--tokenizer", MODEL, "--canvas", str(CANVAS), "--port", "8011"],
                cwd="/fork", env=env, stdout=fh, stderr=subprocess.STDOUT)
        wait_health("http://127.0.0.1:8011/health", interposer_log, interposer, 5 * 60)

        def read(flags: dict, text: str, seed: int) -> tuple[dict, float]:
            body = {
                "model": "dgemma",
                "seed": seed,
                "messages": [
                    {"role": "system", "content": json.dumps(dict(flags, questions=schema_questions))},
                    {"role": "user", "content": json.dumps({"ticket": text})},
                ],
            }
            started = time.time()
            resp = post("http://127.0.0.1:8011/v1/chat/completions", body)
            return json.loads(resp["choices"][0]["message"]["content"]), (time.time() - started) * 1e3

        # Warm the sampler's shapes: the first read of a shape pays a Triton
        # JIT compile worth tens of seconds.
        warm_ms = []
        for flags in (CONDITIONS["k1"], CONDITIONS["k8"]):
            _body, ms = read(flags, tickets[0]["text"], BASE_SEED)
            warm_ms.append(ms)
        print("warmup reads: %s" % ["%.0f ms" % m for m in warm_ms], flush=True)
        _body, k1_warm = read(CONDITIONS["k1"], tickets[0]["text"], BASE_SEED)
        _body, k8_warm = read(CONDITIONS["k8"], tickets[0]["text"], BASE_SEED)
        print("warm read wall time: k1 %.0f ms, k8 %.0f ms" % (k1_warm, k8_warm), flush=True)

        label_ids = {q["id"]: question_label_ids(q) for q in schema_questions}
        names = {q["id"]: label_names(q) for q in schema_questions}

        for ticket in tickets:
            seed = BASE_SEED + int(ticket["id"][1:])
            for cond, flags in CONDITIONS.items():
                for draw in range(DRAWS.get(cond, 1)):
                    body, wall_ms = read(flags, ticket["text"], seed + draw * 7919)
                    traj = body["diagnostics"]["trajectory"]
                    traj = traj[0] if isinstance(traj, list) and traj else None
                    for qi, q in enumerate(schema_questions):
                        qid = q["id"]
                        diag = body["diagnostics"]["questions"][qid]
                        if traj is not None:
                            idx = [traj["label_token_ids"].index(i) for i in label_ids[qid]]
                            per_step = [softmax([s["label_logprobs"][qi][j] for j in idx]) for s in traj["steps"]]
                            per_step_h = [s["entropy"][qi] for s in traj["steps"]]
                            final_probs = per_step[-1]
                            final_h = per_step_h[-1]
                        else:
                            probs = body["answers"][qid]["probabilities"]
                            final_probs = [probs[n] for n in names[qid]]
                            per_step, per_step_h, final_h = None, None, diag["entropy"][0]
                        if any(math.isnan(p) for p in final_probs):
                            raise RuntimeError("NaN probability in a read")
                        records.append({
                            "ticket_id": ticket["id"], "borderline": bool(ticket.get("borderline")),
                            "condition": cond, "draw": draw, "steps": flags["steps"],
                            "question_id": qid, "labels": names[qid],
                            "gt": ticket["gt"][qid], "alt": (ticket.get("alt") or {}).get(qid),
                            "final_probs": final_probs, "final_entropy_full": final_h,
                            "label_mass": diag["label_mass"], "argmax_is_label": diag["argmax_is_label"],
                            "per_step_probs": per_step, "per_step_entropy_full": per_step_h,
                            "wall_ms": wall_ms,
                        })
            print("ticket %s done (%d records)" % (ticket["id"], len(records)), flush=True)
    finally:
        for proc in (interposer, server):
            if proc is not None and proc.poll() is None:
                proc.terminate()
                proc.wait(timeout=60)

    expected = len(tickets) * sum(DRAWS.get(c, 1) for c in CONDITIONS) * len(schema_questions)
    if len(records) != expected:
        raise RuntimeError("record count does not match the ticket x condition x question grid")
    summarize(records)
    payload = "".join(json.dumps(r) + "\n" for r in records)
    out_dir = "/root/.cache/huggingface/pilot_results"
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "pilot_%s.jsonl" % time.strftime("%Y%m%d_%H%M%S"))
    open(out_path, "w").write(payload)
    hf_cache.commit()
    print("volume copy: " + out_path, flush=True)
    return payload


def softmax(logprobs: list[float]) -> list[float]:
    import math

    mx = max(logprobs)
    ex = [math.exp(v - mx) for v in logprobs]
    total = sum(ex)
    return [e / total for e in ex]


def label_names(q: dict) -> list[str]:
    if q["type"] in ("noul", "bool", "boolean"):
        return ["yes", "no"]
    if q["type"] == "choice":
        return [o["name"] if isinstance(o, dict) else str(o) for o in q["options"]]
    return [str(v) for v in q["levels"]]


def structured_server():
    """The interposer module with the globals its helpers read already set."""
    import sys
    import types

    sys.path.insert(0, EXAMPLE_DIR)
    import structured_server as ss

    if ss.TOK is None:
        from transformers import AutoTokenizer

        ss.ARGS = types.SimpleNamespace(model="dgemma", upstream="http://127.0.0.1:8000")
        ss.CANVAS_LEN = CANVAS
        ss.init_tokenizer(AutoTokenizer.from_pretrained(MODEL))
    return ss


def question_label_ids(q: dict) -> list[int]:
    """This question's own label token ids, in label order."""
    ss = structured_server()
    parsed = ss.parse_schema({"questions": [q]})
    _template, slots = ss.resolve_template(parsed["questions"], ss.SCAFFOLD, "", parsed["format"])
    return slots[0]["label_ids"]


def summarize(records: list[dict]) -> None:
    conds: dict[str, list[dict]] = {}
    for r in records:
        conds.setdefault(r["condition"], []).append(r)
    in_label = sum(1 for r in records if r["argmax_is_label"]) / len(records)
    print("argmax_is_label rate: %.3f over %d slots" % (in_label, len(records)), flush=True)
    for cond, rows in conds.items():
        hits = 0
        for r in rows:
            top = r["labels"][max(range(len(r["final_probs"])), key=lambda i: r["final_probs"][i])]
            hits += int(top == r["gt"])
        mean_top = sum(max(r["final_probs"]) for r in rows) / len(rows)
        mean_ms = sum(r["wall_ms"] for r in rows) / len(rows)
        print("SUMMARY %-16s n=%3d acc=%.3f mean_max_prob=%.4f mean_wall_ms=%.0f" % (
            cond, len(rows), hits / len(rows), mean_top, mean_ms), flush=True)


@app.local_entrypoint()
def main() -> None:
    payload = run_pilot.remote()
    out_dir = PILOT_DIR / "results"
    out_dir.mkdir(exist_ok=True)
    path = out_dir / ("pilot_%s.jsonl" % time.strftime("%Y%m%d_%H%M%S"))
    path.write_text(payload)
    print("results: %s (%d records)" % (path, payload.count("\n")))
