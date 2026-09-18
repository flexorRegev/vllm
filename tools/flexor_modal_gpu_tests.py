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
    # The image ships python3 (system or /opt/venv) but no `python` on PATH, and
    # its venv may have no pip; resolve both before installing pytest.
    .run_commands(
        "set -e; PY=$(command -v python3 || echo /opt/venv/bin/python3); echo using $PY; "
        "ln -sf $PY /usr/local/bin/python; "
        "($PY -m pip install pytest || uv pip install --python $PY pytest)"
    )
    .add_local_dir(str(FORK_ROOT / "vllm"), remote_path="/fork/vllm", ignore=["**/__pycache__/**", "**/*.pyc"])
    .add_local_dir(str(FORK_ROOT / "tests"), remote_path="/fork/tests", ignore=["**/__pycache__/**", "**/*.pyc"])
)

app = modal.App("flexor-vllm-fork-gpu-tests", image=image)


@app.function(gpu="L4", timeout=30 * 60)
def run_tests(pytest_args: str) -> int:
    import os
    import shutil
    import subprocess
    import sys

    import vllm as installed_vllm

    # Fill in whatever the fork tree lacks from the installed package, without
    # clobbering fork sources: compiled extensions (`_C`, `_vllm_fa2_C`, ...)
    # in any subpackage, and build-generated files such as `_version.py`.
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
    print(f"filled {copied} files from the installed package", flush=True)

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
