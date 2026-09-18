"""End-to-end check of the fork's DiffusionGemma structured reads on a Modal H100.

Serves the FP8 checkpoint with this fork's sampler, puts ``structured_server.py``
in front of it, and runs five checks: plain generation, a one-step read, a
multi-step read with ``fixed_steps`` / ``trajectory`` / ``slots_never_accept``,
the same read as a raw upstream request, and the timings of all of it.

    uvx --from modal modal run tools/flexor_modal_e2e.py
"""

from __future__ import annotations

import pathlib

import modal

FORK_ROOT = pathlib.Path(__file__).resolve().parent.parent
MODEL = "RedHatAI/diffusiongemma-26B-A4B-it-FP8-dynamic"
CANVAS = 64
EXAMPLE_DIR = "/fork/examples/features/diffusion_reads"

image = (
    modal.Image.from_registry("vllm/vllm-openai:nightly", add_python=None)
    .entrypoint([])
    .run_commands(
        "set -e; PY=$(command -v python3 || echo /opt/venv/bin/python3); echo using $PY; "
        "ln -sf $PY /usr/local/bin/python"
    )
    .add_local_dir(str(FORK_ROOT / "vllm"), remote_path="/fork/vllm", ignore=["**/__pycache__/**", "**/*.pyc"])
    .add_local_dir(str(FORK_ROOT / "examples"), remote_path="/fork/examples", ignore=["**/__pycache__/**", "**/*.pyc"])
)

app = modal.App("flexor-vllm-diffusion-e2e", image=image)
hf_cache = modal.Volume.from_name("diffusiongemma-hf-cache", create_if_missing=True)

TICKET = (
    "Subject: EVERYTHING IS DOWN. Our entire production dashboard has been returning 502 for the last "
    "40 minutes. Nobody on my team can log in, our customers are calling us, and we have a board demo "
    "at noon. This is the third outage this month. I am done being patient. Fix this NOW or we are "
    "cancelling the contract today."
)

SCHEMA_QUESTIONS = [
    {"id": "urgent", "type": "noul", "instructions": "Does the customer need a reply within the hour?"},
    {
        "id": "category",
        "type": "choice",
        "instructions": "Which bucket does this ticket belong to?",
        "options": [
            {"name": "billing", "description": "invoices, charges, refunds"},
            {"name": "outage", "description": "the service is unavailable or erroring"},
            {"name": "feature_request", "description": "asks for something new"},
            {"name": "how_to", "description": "asks how to use an existing feature"},
        ],
    },
    {
        "id": "tone",
        "type": "score",
        "instructions": "How angry is the customer?",
        "levels": ["calm", "annoyed", "furious"],
    },
]

EXPECTED = {"urgent": "yes", "category": "outage", "tone": "furious"}


@app.function(
    gpu="H100",
    timeout=45 * 60,
    secrets=[modal.Secret.from_name("hf-access-token")],
    volumes={"/root/.cache/huggingface": hf_cache},
)
def run_e2e() -> int:
    import json
    import os
    import shutil
    import subprocess
    import sys
    import time
    import urllib.error
    import urllib.request

    import vllm as installed_vllm

    # Fill whatever the fork tree lacks from the installed package (compiled
    # extensions, _version.py) without clobbering fork sources.
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

    env = dict(os.environ, PYTHONPATH="/fork", HF_HOME="/root/.cache/huggingface")
    failures: list[str] = []

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
            print("... waiting %.0fs; log tail:" % (time.time() - started), flush=True)
            print("".join(open(log_path).readlines()[-6:]), flush=True)
        raise RuntimeError("server did not become healthy in budget")

    def check(name: str, ok: bool, detail: str = "") -> None:
        print("CHECK %s: %s %s" % (name, "PASS" if ok else "FAIL", detail), flush=True)
        if not ok:
            failures.append(name + " " + detail)

    server_log = "/tmp/vllm_server.log"
    server_cmd = [
        sys.executable, "-m", "vllm.entrypoints.cli.main", "serve", MODEL,
        "--diffusion-config", json.dumps({"canvas_length": CANVAS}),
        "--max-logprobs", "32", "--enable-prefix-caching", "--async-scheduling",
        "--attention-backend", "TRITON_ATTN", "--max-num-seqs", "32", "--port", "8000",
        "--served-model-name", "dgemma",
    ]
    print("server command: " + " ".join(server_cmd), flush=True)
    with open(server_log, "w") as fh:
        server = subprocess.Popen(server_cmd, cwd="/fork", env=env, stdout=fh, stderr=subprocess.STDOUT)
    interposer = None
    try:
        boot_s = wait_health("http://127.0.0.1:8000/health", server_log, server, 20 * 60)
        print("server healthy after %.1fs" % boot_s, flush=True)
        hf_cache.commit()

        # 1. plain generation
        plain_body = {
            "model": "dgemma",
            "messages": [{"role": "user", "content": "In two sentences, what is a database index?"}],
            "max_tokens": 64,
            "chat_template_kwargs": {"enable_thinking": False},
        }
        print("REQUEST 1: " + json.dumps(plain_body), flush=True)
        t0 = time.time()
        plain = post("http://127.0.0.1:8000/v1/chat/completions", plain_body)
        plain_ms = (time.time() - t0) * 1e3
        print("RESPONSE 1: " + json.dumps(plain), flush=True)
        text = plain["choices"][0]["message"]["content"] or ""
        check("1-plain-generation", len(text.strip()) > 20, "%d chars in %.0f ms" % (len(text), plain_ms))

        # interposer
        interposer_log = "/tmp/structured_server.log"
        interposer_cmd = [
            sys.executable, EXAMPLE_DIR + "/structured_server.py",
            "--upstream", "http://127.0.0.1:8000", "--tokenizer", MODEL,
            "--canvas", str(CANVAS), "--port", "8011",
        ]
        print("interposer command: " + " ".join(interposer_cmd), flush=True)
        with open(interposer_log, "w") as fh:
            interposer = subprocess.Popen(interposer_cmd, cwd="/fork", env=env, stdout=fh, stderr=subprocess.STDOUT)
        wait_health("http://127.0.0.1:8011/health", interposer_log, interposer, 5 * 60)

        def read(schema: dict, tag: str) -> tuple[dict, float]:
            body = {
                "model": "dgemma",
                "messages": [
                    {"role": "system", "content": json.dumps(schema)},
                    {"role": "user", "content": json.dumps({"ticket": TICKET})},
                ],
            }
            print("REQUEST %s: %s" % (tag, json.dumps(body)), flush=True)
            started = time.time()
            resp = post("http://127.0.0.1:8011/v1/chat/completions", body)
            elapsed = (time.time() - started) * 1e3
            print("RESPONSE %s: %s" % (tag, json.dumps(resp)), flush=True)
            return json.loads(resp["choices"][0]["message"]["content"]), elapsed

        # 2. one-step read, default samples policy (run twice: the first read
        # pays the sampler's compile/capture for a new shape)
        _warm, one_cold_ms = read({"questions": SCHEMA_QUESTIONS}, "2-cold")
        one_step, one_ms = read({"questions": SCHEMA_QUESTIONS}, "2")
        for qid, want in EXPECTED.items():
            got = one_step["answers"][qid]
            key = {"urgent": "noul", "category": "choice", "tone": "level"}[qid]
            name = got["label"] if qid == "urgent" else got[key]
            print("  %s -> %s conf=%.4f probs=%s entropy=%s" % (
                qid, name, got["confidence"], json.dumps(got["probabilities"]),
                json.dumps(one_step["diagnostics"]["questions"][qid]["entropy"])), flush=True)
            check("2-label-" + qid, name == want, "got " + str(name))
        print("check 2 wall time: cold %.0f ms, warm %.0f ms (%d reads)" % (
            one_cold_ms, one_ms, one_step["diagnostics"]["samples"]["n"]), flush=True)

        # 3. multi-step read with the fork's flags
        multi_schema = {"questions": SCHEMA_QUESTIONS, "steps": 4, "fixed_steps": True,
                        "trajectory": True, "samples": 1}
        _warm3, multi_cold_ms = read(multi_schema, "3-cold")
        multi, multi_ms = read(multi_schema, "3")
        traj = multi["diagnostics"]["trajectory"]
        ok = isinstance(traj, list) and len(traj) == 1 and isinstance(traj[0], dict)
        check("3-trajectory-present", ok, "type " + type(traj).__name__)
        if ok:
            steps = traj[0]["steps"]
            check("3-step-count", len(steps) == 4, "%d steps" % len(steps))
            label_ids = traj[0]["label_token_ids"]
            print("trajectory positions=%s label_token_ids=%s" % (traj[0]["positions"], label_ids), flush=True)
            print_trajectory(multi_schema, traj[0])
        for qid, want in EXPECTED.items():
            got = multi["answers"][qid]
            name = got["label"] if qid == "urgent" else got.get("choice", got.get("level"))
            check("3-label-" + qid, name == want, "got " + str(name))
        print("check 3 wall time: cold %.0f ms, warm %.0f ms" % (multi_cold_ms, multi_ms), flush=True)
        print("=== server log lines around the warm reads ===", flush=True)
        for line in open(server_log).readlines()[-25:]:
            if "Diffusion" in line or "throughput" in line or "Avg" in line:
                print("  " + line.rstrip(), flush=True)

        ablation, ablation_ms = read(dict(multi_schema, slots_never_accept=True), "3-ablation")
        print("slots_never_accept answers: %s (%.0f ms)" % (
            json.dumps({k: [v["label"], v["confidence"]] for k, v in ablation["answers"].items()}), ablation_ms),
            flush=True)
        if isinstance(ablation["diagnostics"]["trajectory"], list):
            print_trajectory(multi_schema, ablation["diagnostics"]["trajectory"][0])

        # 4. the same read as a raw upstream request
        raw, slots = raw_read_body(multi_schema)
        print("REQUEST 4: " + json.dumps(raw), flush=True)
        t0 = time.time()
        raw_resp = post("http://127.0.0.1:8000/v1/chat/completions", raw)
        raw_ms = (time.time() - t0) * 1e3
        raw_traj = raw_resp["choices"][0].get("diffusion_trajectory")
        print("RESPONSE 4 trajectory: " + json.dumps(raw_traj), flush=True)
        check("4-raw-trajectory", isinstance(raw_traj, dict) and len(raw_traj.get("steps", [])) == 4,
              "steps %s" % (len(raw_traj.get("steps", [])) if isinstance(raw_traj, dict) else None))
        if isinstance(raw_traj, dict):
            check("4-positions", raw_traj["positions"] == [s["pos"] for s in slots],
                  "%s vs %s" % (raw_traj["positions"], [s["pos"] for s in slots]))
        print("check 4 wall time: %.0f ms" % raw_ms, flush=True)

        # 5. timings
        print("TIMINGS server_boot_s=%.1f plain_ms=%.0f read1_cold_ms=%.0f read1_warm_ms=%.0f "
              "read4step_cold_ms=%.0f read4step_warm_ms=%.0f ablation_ms=%.0f raw_ms=%.0f" % (
                  boot_s, plain_ms, one_cold_ms, one_ms, multi_cold_ms, multi_ms, ablation_ms, raw_ms), flush=True)
        check("5-timing", True, "collected")
    finally:
        for proc in (interposer, server):
            if proc is not None and proc.poll() is None:
                proc.terminate()
                proc.wait(timeout=60)
        print("=== server log tail ===", flush=True)
        print("".join(open(server_log).readlines()[-40:]), flush=True)

    print("FAILURES: %s" % (failures or "none"), flush=True)
    return 1 if failures else 0


def print_trajectory(schema: dict, traj: dict) -> None:
    """Per-step label probabilities for each question, from the step logprobs."""
    import math

    label_ids = traj["label_token_ids"]
    for qi, q in enumerate(schema["questions"]):
        names = label_names(q)
        ids = question_label_ids(q)
        print("  %s (pos %d):" % (q["id"], traj["positions"][qi]), flush=True)
        for step in traj["steps"]:
            lps = [step["label_logprobs"][qi][label_ids.index(i)] for i in ids]
            mx = max(lps)
            ex = [math.exp(v - mx) for v in lps]
            probs = [e / sum(ex) for e in ex]
            best = max(range(len(probs)), key=lambda k: probs[k])
            print("    step %d argmax_id=%s entropy=%.3f best=%s %s" % (
                step["step"], step["argmax_id"][qi], step["entropy"][qi], names[best],
                " ".join("%s=%.3f" % (n, p) for n, p in zip(names, probs))), flush=True)


def label_names(q: dict) -> list[str]:
    if q["type"] in ("noul", "bool", "boolean"):
        return ["yes", "no"]
    if q["type"] == "choice":
        return [o["name"] for o in q["options"]]
    return [str(v) for v in q["levels"]]


def question_label_ids(q: dict) -> list[int]:
    """This question's own label ids, in label order, from the shared template."""
    ss = structured_server()
    parsed = ss.parse_schema({"questions": [q]})
    _template, slots = ss.resolve_template(parsed["questions"], ss.SCAFFOLD, "", parsed["format"])
    return slots[0]["label_ids"]


def structured_server():
    """The interposer module, with the globals its helpers read already set:
    importing it runs no main(), so the tokenizer and canvas are unset."""
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


def raw_read_body(schema_value: dict) -> tuple[dict, list[dict]]:
    """The request the interposer sends for one read, rebuilt with its own helpers."""
    import json

    ss = structured_server()
    schema = ss.parse_schema(schema_value)
    template, slots = ss.template_for(schema, ss.SCAFFOLD, "")
    body = {
        "model": "dgemma",
        "messages": [
            {"role": "system", "content": ss.system_text(schema)},
            {"role": "user", "content": json.dumps({"ticket": TICKET})},
        ],
        "max_tokens": len(template) + 1,
        "logprobs": True,
        "top_logprobs": ss.TOPK,
        "logprob_token_ids": ss.label_id_union(slots),
        "return_tokens_as_token_ids": True,
        "chat_template_kwargs": {"enable_thinking": False},
        "vllm_xargs": {
            "diffusion_seed_canvas": ss.build_canvas(template, slots, 42),
            "diffusion_canvas_length": ss.canvas_width(template),
            "diffusion_slot_positions": [s["pos"] for s in slots],
            "diffusion_max_steps": 4,
            "diffusion_read_only": True,
            "diffusion_trajectory": True,
            "diffusion_fixed_steps": True,
        },
    }
    return body, slots


@app.local_entrypoint()
def main() -> None:
    rc = run_e2e.remote()
    print("e2e exit code: %d" % rc)
    raise SystemExit(rc)
