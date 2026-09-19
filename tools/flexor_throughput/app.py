"""Throughput harness for DiffusionGemma structured reads on one H100.

Same overlay as ``tools/flexor_modal_e2e.py`` (vLLM nightly image, the fork on
PYTHONPATH, the FP8 checkpoint) but without the interposer process: the class
holds the server for its whole lifetime and drives it with an async client, so
a read costs one HTTP round trip inside the container.

    uvx --from modal modal run tools/flexor_throughput/app.py --mode smoke
    uvx --from modal modal run tools/flexor_throughput/app.py --mode probe
    uvx --from modal modal run tools/flexor_throughput/app.py --mode bench
"""

import json
import pathlib
import time
from typing import Any

import modal

FORK_ROOT = pathlib.Path(__file__).resolve().parent.parent.parent
RESULTS_DIR = pathlib.Path(__file__).resolve().parent / "results"
MODEL = "RedHatAI/diffusiongemma-26B-A4B-it-FP8-dynamic"
CANVAS = 64
EXAMPLE_DIR = "/fork/examples/features/diffusion_reads"
SERVER_LOG = "/tmp/vllm_server.log"

image = (
    modal.Image.from_registry("vllm/vllm-openai:nightly", add_python=None)
    .entrypoint([])
    .run_commands(
        "set -e; PY=$(command -v python3 || echo /opt/venv/bin/python3); echo using $PY; "
        "ln -sf $PY /usr/local/bin/python",
        "python -c 'import httpx' || python -m pip install httpx",
    )
    .add_local_dir(str(FORK_ROOT / "vllm"), remote_path="/fork/vllm", ignore=["**/__pycache__/**", "**/*.pyc"])
    .add_local_dir(str(FORK_ROOT / "examples"), remote_path="/fork/examples", ignore=["**/__pycache__/**", "**/*.pyc"])
)

app = modal.App("flexor-diffusion-throughput", image=image)
hf_cache = modal.Volume.from_name("diffusiongemma-hf-cache", create_if_missing=True)

# Ten questions with single-character ids: the answer template has to fit the
# 64-row canvas, and the interposer refuses any label that is not one token
# (choice -> "A".."D", score -> "1".."3", noul -> "yes"/"no").
BENCH_SCHEMA: dict[str, Any] = {
    "questions": [
        {"id": "a", "type": "noul", "instructions": "Does the customer need a reply within the hour?"},
        {"id": "b", "type": "noul", "instructions": "Has the customer threatened to cancel?"},
        {"id": "c", "type": "noul", "instructions": "Did the agent resolve the issue in this conversation?"},
        {"id": "d", "type": "noul", "instructions": "Is a refund or credit being discussed?"},
        {"id": "e", "type": "choice", "instructions": "Which bucket does this conversation belong to?",
         "options": [{"name": "billing", "description": "invoices, charges, refunds"},
                     {"name": "outage", "description": "the service is unavailable or erroring"},
                     {"name": "feature_request", "description": "asks for something new"},
                     {"name": "how_to", "description": "asks how to use an existing feature"}]},
        {"id": "f", "type": "choice", "instructions": "Which product surface is involved?",
         "options": [{"name": "dashboard", "description": "the web UI"},
                     {"name": "api", "description": "the public API or SDKs"},
                     {"name": "billing_portal", "description": "invoices and payment methods"},
                     {"name": "mobile", "description": "the phone apps"}]},
        {"id": "g", "type": "choice", "instructions": "Who moved the conversation forward last?",
         "options": [{"name": "agent", "description": "the support agent"},
                     {"name": "customer", "description": "the customer"},
                     {"name": "bot", "description": "an automated reply"},
                     {"name": "nobody", "description": "the thread stalled"}]},
        {"id": "h", "type": "choice", "instructions": "What should happen next?",
         "options": [{"name": "escalate", "description": "hand to engineering"},
                     {"name": "reply", "description": "an agent answers"},
                     {"name": "close", "description": "nothing left to do"},
                     {"name": "wait", "description": "waiting on the customer"}]},
        {"id": "i", "type": "score", "instructions": "How angry is the customer?",
         "levels": ["calm", "annoyed", "furious"]},
        {"id": "j", "type": "score", "instructions": "How severe is the business impact?",
         "levels": ["low", "medium", "high"]},
    ]
}


def structured_server(tokenizer: Any) -> Any:
    """The interposer module used as a library: importing it runs no main(), so
    the globals its helpers read (ARGS, CANVAS_LEN, TOK) are set here."""
    import sys
    import types

    sys.path.insert(0, EXAMPLE_DIR)
    import structured_server as ss

    if ss.TOK is None:
        ss.ARGS = types.SimpleNamespace(model="dgemma", upstream="http://127.0.0.1:8000")
        ss.CANVAS_LEN = CANVAS
        ss.init_tokenizer(tokenizer)
    return ss


def label_names(question: dict) -> list[str]:
    return [name for name, _desc in question["choices"]]


def scrape_log(offset: int) -> tuple[dict[str, Any], int]:
    """vLLM's own counters emitted while one bench point ran."""
    import re

    with open(SERVER_LOG) as fh:
        fh.seek(offset)
        chunk = fh.read()
        new_offset = fh.tell()
    kv = [float(v) for v in re.findall(r"KV cache usage: ([\d.]+)%", chunk)]
    prompt_tp = [float(v) for v in re.findall(r"Prompt throughput: ([\d.]+) tokens/s", chunk)]
    diffusion = [line.strip() for line in chunk.splitlines() if "DiffusionDecoding metrics" in line]
    recompiles = len(re.findall(r"Recompiling function", chunk))
    hit_limit = "recompile_limit" in chunk or "cache_size_limit" in chunk
    lines = chunk.splitlines()
    trace: list[str] = []
    for index, line in enumerate(lines):
        if "Traceback (most recent call last)" in line:
            # Keep the whole ERROR block: the exception line is the last one,
            # and the diffusion frames sit between it and the sampler call.
            for text in lines[index : index + 200]:
                if "ERROR" not in text and text.strip():
                    break
                trace.append(text.strip())
            break
    return {
        "kv_max_pct": max(kv) if kv else None,
        "kv_mean_pct": sum(kv) / len(kv) if kv else None,
        "prompt_tps_max": max(prompt_tp) if prompt_tp else None,
        "diffusion_lines": diffusion[-2:],
        "recompiles": recompiles,
        "hit_recompile_limit": hit_limit,
        "traceback": trace,
    }, new_offset


@app.cls(
    gpu="H100",
    timeout=3 * 60 * 60,
    secrets=[modal.Secret.from_name("hf-access-token")],
    volumes={"/root/.cache/huggingface": hf_cache},
    max_containers=1,
)
class Throughput:
    max_num_seqs: int = modal.parameter(default=128)
    max_num_batched_tokens: int = modal.parameter(default=16384)

    @modal.enter()
    def start(self) -> None:
        self._boot()
        from transformers import AutoTokenizer

        self.tok = AutoTokenizer.from_pretrained(MODEL)
        self.ss = structured_server(self.tok)
        self._warmup()

    def _boot(self) -> None:
        import os
        import shutil
        import subprocess
        import sys
        import urllib.error
        import urllib.request

        import vllm as installed_vllm

        # Fill whatever the fork tree lacks from the installed package
        # (compiled extensions, _version.py) without clobbering fork sources.
        installed_dir = os.path.dirname(installed_vllm.__file__)
        for root, _dirs, files in os.walk(installed_dir):
            rel = os.path.relpath(root, installed_dir)
            dest_dir = os.path.normpath(os.path.join("/fork/vllm", rel))
            for name in files:
                dest = os.path.join(dest_dir, name)
                if os.path.exists(dest) or name.endswith(".pyc"):
                    continue
                os.makedirs(dest_dir, exist_ok=True)
                shutil.copy(os.path.join(root, name), dest)

        env = dict(os.environ, PYTHONPATH="/fork", HF_HOME="/root/.cache/huggingface",
                   TORCH_LOGS="recompiles")
        cmd = [
            sys.executable, "-m", "vllm.entrypoints.cli.main", "serve", MODEL,
            "--diffusion-config", json.dumps({"canvas_length": CANVAS}),
            "--max-logprobs", "32", "--enable-prefix-caching", "--async-scheduling",
            "--attention-backend", "TRITON_ATTN",
            "--max-num-seqs", str(self.max_num_seqs),
            "--max-num-batched-tokens", str(self.max_num_batched_tokens),
            "--port", "8000", "--served-model-name", "dgemma",
        ]
        self.server_cmd = " ".join(cmd)
        print("server command: " + self.server_cmd, flush=True)
        with open(SERVER_LOG, "w") as fh:
            self.server = subprocess.Popen(cmd, cwd="/fork", env=env, stdout=fh, stderr=subprocess.STDOUT)

        started = time.time()
        while time.time() - started < 25 * 60:
            if self.server.poll() is not None:
                print(open(SERVER_LOG).read()[-8000:], flush=True)
                raise RuntimeError("server exited before becoming healthy")
            try:
                with urllib.request.urlopen("http://127.0.0.1:8000/health", timeout=5) as r:
                    healthy = r.status == 200
            except (urllib.error.URLError, OSError):
                healthy = False
            if healthy:
                break
            time.sleep(10)
        else:
            raise RuntimeError("server did not become healthy in budget")
        print("server healthy after %.1fs" % (time.time() - started), flush=True)
        hf_cache.commit()

    def _alive(self) -> bool:
        import urllib.error
        import urllib.request

        if self.server.poll() is not None:
            return False
        try:
            with urllib.request.urlopen("http://127.0.0.1:8000/health", timeout=5) as r:
                return bool(r.status == 200)
        except (urllib.error.URLError, OSError):
            return False

    def _restart(self) -> None:
        """An EngineCore crash takes the server with it; the probe keeps going
        on a fresh one so later cases still run."""
        if self.server.poll() is None:
            self.server.kill()
            self.server.wait(timeout=60)
        self._boot()
        self._warmup()

    def _warmup(self) -> None:
        """A few reads at every canvas shape the class will use, so the Triton
        JIT for the sampler and the logprob kernels is paid before any timing."""
        import urllib.request

        plan = self._plan(BENCH_SCHEMA)
        items = make_items(self.tok, 2) + [{"id": "short", "text": "Customer: the dashboard is down."}]
        started = time.time()
        for steps in (1, 4):
            for item in items:
                body = json.dumps(self._body(plan, item, steps)).encode()
                req = urllib.request.Request("http://127.0.0.1:8000/v1/chat/completions", data=body,
                                             headers={"content-type": "application/json"})
                urllib.request.urlopen(req, timeout=600).read()
        print("warmup done in %.1fs" % (time.time() - started), flush=True)

    @modal.exit()
    def stop(self) -> None:
        if self.server.poll() is None:
            self.server.terminate()
            self.server.wait(timeout=60)
        print("=== server log tail ===\n" + "".join(open(SERVER_LOG).readlines()[-30:]), flush=True)

    def _plan(self, schema_value: dict) -> dict[str, Any]:
        """Template, slots and system prompt for one schema, built once: every
        item then shares the system prefix, which is what the prefix cache
        hits (the conversations themselves differ)."""
        ss = self.ss
        schema = ss.parse_schema(schema_value)
        template, slots = ss.template_for(schema, ss.SCAFFOLD, "")
        return {
            "schema": schema,
            "template": template,
            "slots": slots,
            "system": ss.system_text(schema),
            "label_ids": ss.label_id_union(slots),
            "canvas": ss.canvas_width(template),
        }

    def _body(self, plan: dict, item: dict, steps: int, flags: dict | None = None) -> dict:
        import zlib

        flags = flags or {}
        ss = self.ss
        seed = zlib.crc32(item["id"].encode())  # stable across runs, unlike hash()
        return {
            "model": "dgemma",
            "messages": [
                {"role": "system", "content": plan["system"]},
                {"role": "user", "content": item["text"]},
            ],
            "max_tokens": len(plan["template"]) + 1,
            "logprobs": True,
            "top_logprobs": ss.TOPK,
            "logprob_token_ids": plan["label_ids"],
            "return_tokens_as_token_ids": True,
            "chat_template_kwargs": {"enable_thinking": False},
            "vllm_xargs": {
                "diffusion_canvas_length": plan["canvas"],
                "diffusion_max_steps": steps,
                "diffusion_read_only": True,
                **({"diffusion_seed_canvas": ss.build_canvas(plan["template"], plan["slots"], seed)}
                   if flags.get("seed_canvas", True) else {}),
                **({"diffusion_slot_positions": [s["pos"] for s in plan["slots"]]}
                   if flags.get("seed_canvas", True) and flags.get("slot_positions", True) else {}),
                **({"diffusion_fixed_steps": True} if flags.get("fixed_steps", True) else {}),
            },
        }

    def _answers(self, plan: dict, payload: dict, temperature: float | None) -> dict[str, Any]:
        import math

        content = payload["choices"][0]["logprobs"]["content"]
        answers: dict[str, Any] = {}
        for question, slot in zip(plan["schema"]["questions"], plan["slots"]):
            top = {int(t["token"].split(":")[1]): t["logprob"] for t in content[slot["pos"]]["top_logprobs"]}
            dist = self.ss.slot_distribution(top, slot["label_ids"])
            names = label_names(question)
            best = max(range(len(dist["probs"])), key=lambda k: dist["probs"][k])
            entry = {
                "label": names[best],
                "probs": dict(zip(names, dist["probs"])),
                "label_mass": dist["label_mass"],
                "argmax_is_label": dist["argmax_is_label"],
                "calibrated": None,
            }
            if temperature is not None:
                floor = min(top.values()) - 5.0
                lps = [top.get(i, floor) / temperature for i in slot["label_ids"]]
                mx = max(lps)
                ex = [math.exp(v - mx) for v in lps]
                entry["calibrated"] = dict(zip(names, [e / sum(ex) for e in ex]))
            answers[question["id"]] = entry
        return answers

    async def _read_many(
        self, items: list[dict], schema: dict, steps: int, concurrency: int, temperature: float | None,
        flags: dict | None = None
    ) -> tuple[list[dict], dict[str, Any]]:
        import asyncio

        import httpx

        plan = self._plan(schema)
        gate = asyncio.Semaphore(concurrency)
        out: list[dict | None] = [None] * len(items)

        async def run(index: int, item: dict, client: httpx.AsyncClient) -> None:
            async with gate:
                started = time.perf_counter()
                try:
                    resp = await client.post("/v1/chat/completions", json=self._body(plan, item, steps, flags))
                except httpx.HTTPError as exc:
                    out[index] = {"id": item["id"], "error": repr(exc), "latency_ms": (time.perf_counter() - started) * 1e3}
                    return
                elapsed = (time.perf_counter() - started) * 1e3
                if resp.status_code != 200:
                    out[index] = {"id": item["id"], "error": "HTTP %d %s" % (resp.status_code, resp.text[:400]),
                                  "latency_ms": elapsed}
                    return
                out[index] = {"id": item["id"], "answers": self._answers(plan, resp.json(), temperature),
                              "latency_ms": elapsed}

        limits = httpx.Limits(max_connections=concurrency + 8, max_keepalive_connections=concurrency + 8)
        wall_start = time.perf_counter()
        async with httpx.AsyncClient(base_url="http://127.0.0.1:8000", timeout=900.0, limits=limits) as client:
            await asyncio.gather(*[run(i, item, client) for i, item in enumerate(items)])
        wall = time.perf_counter() - wall_start

        results = [r for r in out if r is not None]
        ok = [r for r in results if "error" not in r]
        lat = sorted(r["latency_ms"] for r in ok)
        pick = lambda q: lat[min(len(lat) - 1, int(q * len(lat)))] if lat else None  # noqa: E731
        summary = {
            "n": len(results),
            "n_failed": len(results) - len(ok),
            "wall_s": wall,
            "req_per_s": len(ok) / wall if wall > 0 else 0.0,
            "decisions_per_s": len(ok) * len(plan["schema"]["questions"]) / wall if wall > 0 else 0.0,
            "p50_ms": pick(0.50),
            "p95_ms": pick(0.95),
            "p99_ms": pick(0.99),
            "steps": steps,
            "concurrency": concurrency,
            "errors": [r["error"] for r in results if "error" in r][:3],
        }
        return results, summary

    @modal.method()
    async def read_many(
        self,
        items: list[dict],
        schema: dict,
        steps: int = 4,
        concurrency: int = 96,
        temperature: float | None = None,
    ) -> dict[str, Any]:
        results, summary = await self._read_many(items, schema, steps, concurrency, temperature)
        return {"results": results, "summary": summary}

    @modal.method()
    async def probe(self, cases: list[dict], n_items: int = 200) -> list[dict]:
        """One container, many small points: isolates which xarg combination and
        which concurrency turns a multi-step read into a 500."""
        items = make_items(self.tok, n_items)
        offset = pathlib.Path(SERVER_LOG).stat().st_size
        out: list[dict] = []
        for case in cases:
            flags = {"seed_canvas": case.get("seed_canvas", True),
                     "slot_positions": case.get("slot_positions", True),
                     "fixed_steps": case.get("fixed_steps", True)}
            _results, summary = await self._read_many(
                items, BENCH_SCHEMA, case["steps"], case["concurrency"], None, flags)
            metrics, offset = scrape_log(offset)
            row = {"name": case["name"], **flags, "steps": case["steps"], "concurrency": case["concurrency"],
                   "n": summary["n"], "n_failed": summary["n_failed"], "errors": summary["errors"],
                   "recompiles": metrics["recompiles"], "hit_recompile_limit": metrics["hit_recompile_limit"],
                   "traceback": metrics["traceback"]}
            out.append(row)
            row["engine_alive"] = self._alive()
            print("probe %s -> failed %d/%d recompiles=%d limit=%s %s"
                  % (case["name"], summary["n_failed"], summary["n"], metrics["recompiles"],
                     metrics["hit_recompile_limit"], (summary["errors"] or [""])[0][:200]), flush=True)
            if metrics["traceback"]:
                print("TRACEBACK %s\n%s" % (case["name"], "\n".join(metrics["traceback"])), flush=True)
            if not row["engine_alive"]:
                print("engine died on %s; restarting" % case["name"], flush=True)
                self._restart()
                offset = pathlib.Path(SERVER_LOG).stat().st_size
        return out

    @modal.method()
    async def bench(
        self,
        n_items: int = 200,
        concurrencies: list[int] | None = None,
        steps_list: list[int] | None = None,
    ) -> dict[str, Any]:
        concurrencies = concurrencies or [1, 8, 32, 64, 128]
        steps_list = steps_list or [1, 4]
        items = make_items(self.tok, n_items)
        lengths = sorted(len(self.tok.encode(i["text"], add_special_tokens=False)) for i in items)
        print("workload: %d items, prompt tokens min=%d median=%d max=%d"
              % (len(items), lengths[0], lengths[len(lengths) // 2], lengths[-1]), flush=True)

        offset = pathlib.Path(SERVER_LOG).stat().st_size
        rows: list[dict] = []
        for steps in steps_list:
            for concurrency in concurrencies:
                await self._read_many(items[:8], BENCH_SCHEMA, steps, min(concurrency, 8), None)  # warmup, discarded
                _stale, offset = scrape_log(offset)
                _results, summary = await self._read_many(items, BENCH_SCHEMA, steps, concurrency, None)
                metrics, offset = scrape_log(offset)
                summary["server"] = metrics
                rows.append(summary)
                print("point steps=%d conc=%d -> %.2f req/s p95=%.0f ms failed=%d"
                      % (steps, concurrency, summary["req_per_s"], summary["p95_ms"] or 0, summary["n_failed"]),
                      flush=True)
        return {
            "rows": rows,
            "server_cmd": self.server_cmd,
            "max_num_seqs": self.max_num_seqs,
            "max_num_batched_tokens": self.max_num_batched_tokens,
            "prompt_tokens": {"min": lengths[0], "median": lengths[len(lengths) // 2], "max": lengths[-1]},
        }


def make_items(tokenizer: Any, count: int) -> list[dict]:
    """``count`` distinct support conversations of roughly two thousand tokens,
    trimmed on the token axis so every prompt lands in the 1,900-2,100 band."""
    import random

    subjects = ["the billing dashboard", "the export job", "the API rate limiter", "single sign-on",
                "the mobile app", "webhook delivery", "the audit log", "seat management",
                "the data importer", "scheduled reports"]
    symptoms = ["returns a 502 for every request", "hangs for about four minutes and then times out",
                "shows yesterday's numbers", "drops every third event", "rejects a valid token",
                "logs the user out mid-session", "duplicates rows on retry", "silently truncates the file",
                "reports success but writes nothing", "charges the old plan price"]
    asks = ["I need this fixed before our board demo", "please credit the invoice",
            "can you tell me whether the data is safe", "we need a workaround today",
            "escalate this to someone who can actually change it", "just tell me when it will be done",
            "our compliance team is asking for an incident note", "I want a call with your engineer"]
    moods = ["I have been patient long enough.", "This is the third week in a row.",
             "We are evaluating alternatives.", "I appreciate the help so far.",
             "Nobody on my team can work like this.", "The last agent promised a fix and vanished."]
    filler = ["I checked our network and it is fine on our side.",
              "I attached the request ids from this morning in the previous message.",
              "The same account works from a different browser, which makes no sense to me.",
              "Our integration has not changed in two months.",
              "We saw the same behaviour in the sandbox environment.",
              "I reproduced it with a brand new user as well.",
              "The retry succeeded once out of maybe ten attempts.",
              "Support told us last time that this was already fixed."]

    items: list[dict] = []
    for index in range(count):
        rng = random.Random(9000 + index)
        turns = ["Customer: We are seeing a problem with %s. It %s. %s"
                 % (rng.choice(subjects), rng.choice(symptoms), rng.choice(asks))]
        while True:
            text = "\n".join(turns)
            n_tokens = len(tokenizer.encode(text, add_special_tokens=False))
            if n_tokens >= 1900:
                break
            turns.append("Agent: Thanks for the detail. %s Could you confirm the account id and the exact time?"
                         % rng.choice(filler))
            turns.append("Customer: %s %s It still %s when I try %s."
                         % (rng.choice(moods), rng.choice(filler), rng.choice(symptoms), rng.choice(subjects)))
        if n_tokens > 2100:
            ids = tokenizer.encode(text, add_special_tokens=False)[:2050]
            text = tokenizer.decode(ids)
        items.append({"id": "conv-%04d" % index, "text": text})
    return items


def render(payload: dict[str, Any], extra: dict[str, Any] | None = None) -> str:
    lines = [
        "# DiffusionGemma structured reads: throughput bench",
        "",
        "Model `%s`, one H100, canvas %d, 10 questions per read." % (MODEL, CANVAS),
        "",
        "```",
        payload["server_cmd"],
        "```",
        "",
        "Prompt tokens per conversation: min %d, median %d, max %d."
        % (payload["prompt_tokens"]["min"], payload["prompt_tokens"]["median"], payload["prompt_tokens"]["max"]),
        "",
        "| steps | concurrency | req/s | decisions/s | p50 ms | p95 ms | p99 ms | failed | KV max % | prompt tok/s |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in payload["rows"]:
        server = row.get("server") or {}
        lines.append("| %d | %d | %.2f | %.1f | %.0f | %.0f | %.0f | %d | %s | %s |" % (
            row["steps"], row["concurrency"], row["req_per_s"], row["decisions_per_s"],
            row["p50_ms"] or 0, row["p95_ms"] or 0, row["p99_ms"] or 0, row["n_failed"],
            "%.1f" % server["kv_max_pct"] if server.get("kv_max_pct") is not None else "-",
            "%.0f" % server["prompt_tps_max"] if server.get("prompt_tps_max") is not None else "-"))
    failures = [(row["steps"], row["concurrency"], row["errors"]) for row in payload["rows"] if row["errors"]]
    if failures:
        lines += ["", "## Failures", ""]
        for steps, concurrency, errors in failures:
            lines.append("- steps=%d concurrency=%d: `%s`" % (steps, concurrency, errors[0]))
    if extra:
        lines += ["", "## max_num_batched_tokens comparison", "",
                  "| max_num_batched_tokens | steps | concurrency | req/s | p95 ms | failed |",
                  "| --- | --- | --- | --- | --- | --- |"]
        for label, row in extra.items():
            lines.append("| %s | %d | %d | %.2f | %.0f | %d |" % (
                label, row["steps"], row["concurrency"], row["req_per_s"], row["p95_ms"] or 0, row["n_failed"]))
    return "\n".join(lines) + "\n"


@app.local_entrypoint()
def main(mode: str = "smoke", n_items: int = 200, budget_compare: bool = True,
         steps: str = "1,4", concurrencies: str = "1,8,32,64,128") -> None:
    if mode == "smoke":
        runner = Throughput(max_num_seqs=128, max_num_batched_tokens=16384)
        items = [{"id": "smoke-%d" % i, "text": "Customer: %s is down and I need it fixed now." % s}
                 for i, s in enumerate(["the dashboard", "the API", "billing", "export", "SSO"])]
        payload = runner.read_many.remote(items, BENCH_SCHEMA, 4, 5, 2.0)
        print(json.dumps(payload["summary"], indent=2))
        print(json.dumps(payload["results"][0]["answers"], indent=2)[:2000])
        return

    if mode == "probe":
        runner = Throughput(max_num_seqs=128, max_num_batched_tokens=16384)
        cases = [
            {"name": "s4-c8-full-cold", "steps": 4, "concurrency": 8},
            {"name": "s4-c8-full-warm", "steps": 4, "concurrency": 8},
            {"name": "s4-c8-no-slot-positions", "steps": 4, "concurrency": 8, "slot_positions": False},
            {"name": "s4-c8-no-fixed-steps", "steps": 4, "concurrency": 8, "fixed_steps": False},
        ] + [
            {"name": "s4-c%d-churn" % c, "steps": 4, "concurrency": c} for c in (1, 8, 32, 64, 128)
        ]
        rows = runner.probe.remote(cases)
        print(json.dumps(rows, indent=2)[:12000])
        return

    if mode == "debug":
        runner = Throughput(max_num_seqs=128, max_num_batched_tokens=16384)
        payload = runner.bench.remote(16, [8], [4])
        print(json.dumps(payload["rows"], indent=2)[:6000])
        return

    if mode != "bench":
        raise ValueError("mode must be smoke, probe, debug or bench")

    runner = Throughput(max_num_seqs=128, max_num_batched_tokens=16384)
    steps_list = [int(v) for v in steps.split(",")]
    concurrency_list = [int(v) for v in concurrencies.split(",")]
    payload = runner.bench.remote(n_items, concurrency_list, steps_list)
    extra: dict[str, Any] = {}
    if budget_compare:
        best = max((r for r in payload["rows"] if r["n_failed"] == 0), key=lambda r: r["req_per_s"])
        extra["16384"] = best
        small = Throughput(max_num_seqs=128, max_num_batched_tokens=8192)
        small_payload = small.bench.remote(n_items, [best["concurrency"]], [best["steps"]])
        extra["8192"] = small_payload["rows"][0]
    text = render(payload, extra)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULTS_DIR / ("bench_%s.md" % time.strftime("%Y%m%dT%H%M%S"))
    out.write_text(text)
    print(text)
    print("wrote " + str(out))
