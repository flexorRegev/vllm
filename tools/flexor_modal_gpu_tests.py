"""Run the fork's GPU-only DiffusionGemma sampler tests on a Modal GPU.

The vllm/vllm-openai nightly image supplies torch, the compiled vllm ops and
the CUDA stack; this fork's ``vllm`` and ``tests`` trees are overlaid on top so
the sampler under test is the local source. The compiled extension modules are
copied from the installed package into the overlay so ``import vllm._C`` keeps
resolving.

    uvx --from modal modal run tools/flexor_modal_gpu_tests.py
    uvx --from modal modal run tools/flexor_modal_gpu_tests.py --pytest-args "tests/v1/sample/test_diffusion_gemma_reads.py -k clamp"
"""

from __future__ import annotations

import pathlib

import modal

FORK_ROOT = pathlib.Path(__file__).resolve().parent.parent
DEFAULT_TESTS = "tests/test_sampling_params.py tests/v1/sample/test_diffusion_gemma_reads.py"

image = (
    modal.Image.from_registry("vllm/vllm-openai:nightly", add_python=None)
    .entrypoint([])
    .pip_install("pytest")
    .add_local_dir(str(FORK_ROOT / "vllm"), remote_path="/fork/vllm", ignore=["**/__pycache__/**", "**/*.pyc"])
    .add_local_dir(str(FORK_ROOT / "tests"), remote_path="/fork/tests", ignore=["**/__pycache__/**", "**/*.pyc"])
)

app = modal.App("flexor-vllm-fork-gpu-tests", image=image)


@app.function(gpu="L4", timeout=30 * 60)
def run_tests(pytest_args: str) -> int:
    import glob
    import os
    import shutil
    import subprocess
    import sys

    import vllm as installed_vllm

    installed_dir = os.path.dirname(installed_vllm.__file__)
    for so in glob.glob(os.path.join(installed_dir, "*.so")):
        shutil.copy(so, "/fork/vllm/")
    for name in ("version.py", "_version.py"):
        src = os.path.join(installed_dir, name)
        if os.path.exists(src) and not os.path.exists(f"/fork/vllm/{name}"):
            shutil.copy(src, "/fork/vllm/")

    env = dict(os.environ, PYTHONPATH="/fork", VLLM_LOGGING_LEVEL="WARNING")
    cmd = [sys.executable, "-m", "pytest", *pytest_args.split(), "-q", "-p", "no:cacheprovider", "--noconftest", "-rs"]
    print("running:", " ".join(cmd), flush=True)
    proc = subprocess.run(cmd, cwd="/fork", env=env)
    return proc.returncode


@app.local_entrypoint()
def main(pytest_args: str = DEFAULT_TESTS) -> None:
    rc = run_tests.remote(pytest_args)
    print(f"pytest exit code: {rc}")
    raise SystemExit(rc)
