"""CPU-only shell integration tests: Git, Python, sbatch and srun are stand-ins."""

import os
import re
import shutil
import subprocess
from pathlib import Path

import pytest


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts/slurm"
STORAGE = (
    "HF_HOME", "HF_HUB_CACHE", "BENCH_CACHE_DIR", "BENCH_OUTPUT_ROOT", "XDG_CACHE_HOME",
    "HF_XET_CACHE", "HF_ASSETS_CACHE", "HF_MODULES_CACHE", "TORCH_HOME", "TRITON_CACHE_DIR",
    "CUDA_CACHE_PATH", "PIP_CACHE_DIR", "TMPDIR", "TMP", "TEMP", "PYTHONPYCACHEPREFIX",
)


def shell_path(path):
    """Translate Windows test paths for Git Bash; leave POSIX paths unchanged."""
    value = Path(path).resolve().as_posix()
    return f"/{value[0].lower()}{value[2:]}" if os.name == "nt" else value


@pytest.fixture(scope="module")
def bash():
    candidates = ([r"C:\Program Files\Git\bin\bash.exe"] if os.name == "nt" else [])
    candidates.append(shutil.which("bash"))
    executable = next((value for value in candidates if value and Path(value).is_file()), None)
    if executable is None:
        pytest.skip("Bash unavailable; run shell integration tests on Linux or with Git Bash")
    return executable


@pytest.fixture
def shell_run(tmp_path, bash):
    home, scratch, commands = (tmp_path / name for name in ("home", "scratch", "commands"))
    for path in (home, scratch, commands):
        path.mkdir()

    def stub(name, body):
        path = commands / name
        path.write_text("#!/bin/bash\nset -eu\n" + body + "\n", encoding="utf-8", newline="\n")
        path.chmod(0o755)
        return path

    stub("git", 'if [[ "$*" = "rev-parse HEAD" ]]; then exit "${TEST_GIT_EXIT:-0}"; fi\nprintf "mock Git\\n"')
    stub("sha256sum", 'cat >/dev/null; printf "train: OK\\nval: OK\\n"')
    stub("df", "exit 0")
    stub("nvidia-smi", "exit 0")
    python = stub("python", "exit 0")
    stub("srun", 'printf "%s\\n" "$@" >"$TEST_ARGS"\n'
         + 'for name in ' + " ".join((*STORAGE, "BENCH_PYTHON")) + '; do\n'
         + '  printf "%s=%s\\n" "$name" "${!name}"\ndone >"$TEST_STORAGE"\nexit "${TEST_SRUN_EXIT:-0}"')
    stub("sbatch", 'printf "%s\\n" "$@" >"$TEST_ARGS"')
    environment = os.environ.copy()
    for name in (*STORAGE, "BENCH_LOG_DIR", "BENCH_SHOTS", "SBATCH_ARRAY_INX", "TRANSFORMERS_CACHE", "PYTORCH_TRANSFORMERS_CACHE",
                 "PYTORCH_PRETRAINED_BERT_CACHE", "BASH_ENV", "ENV"):
        environment.pop(name, None)
    environment.update(HOME=shell_path(home), BENCH_SCRATCH_ROOT=shell_path(scratch),
                       BENCH_PYTHON=shell_path(python), SLURM_SUBMIT_DIR=shell_path(tmp_path),
                       SLURM_JOB_ID="123", SLURM_ARRAY_TASK_ID="5",
                       TEST_ARGS=shell_path(tmp_path / "args"), TEST_STORAGE=shell_path(tmp_path / "storage"))

    def run(script="benchmark.sbatch", args=("ministral", "smoke", "v1"), overrides=None):
        env = {**environment, **(overrides or {})}
        # Preserve Linux/Git Bash utilities; fake commands always precede them.
        return subprocess.run(
            [bash, "--noprofile", "--norc", "-c", 'export PATH="$1:/usr/bin:/bin"; shift; exec bash "$@"',
             "test", shell_path(commands), shell_path(SCRIPTS / script), *args],
            env=env, cwd=tmp_path, capture_output=True, text=True, timeout=30,
        )

    public_paths = {name: environment[name] for name in ("HOME", "BENCH_SCRATCH_ROOT", "BENCH_PYTHON")}
    return run, public_paths, tmp_path


@pytest.mark.parametrize("name", ("benchmark.sbatch", "submit_benchmark.sh"))
def test_shell_syntax_and_lf(bash, name):
    path = SCRIPTS / name
    assert b"\r" not in path.read_bytes()
    result = subprocess.run([bash, "--noprofile", "--norc", "-n", shell_path(path)], capture_output=True, text=True, timeout=15)
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("model,precision", [("ministral", "bf16-from-fp8"), ("granite", "bf16")])
@pytest.mark.parametrize("mode", ("smoke", "full"))
def test_scratch_defaults_and_original_run_settings(shell_run, model, precision, mode):
    run, env, root = shell_run
    result = run(args=(model, mode, "v1"))
    assert result.returncode == 0, result.stderr
    storage = dict(line.split("=", 1) for line in (root / "storage").read_text().splitlines())
    assert all(storage[name].startswith(env["BENCH_SCRATCH_ROOT"] + "/") for name in STORAGE)
    assert storage["BENCH_PYTHON"] == env["BENCH_PYTHON"]  # Preserve venv invocation path.
    arguments = (root / "args").read_text().splitlines()
    for flag, value in (("--shots", "5"), ("--run-id", f"{precision}-{mode}-v1"),
                        ("--output-root", storage["BENCH_OUTPUT_ROOT"]), ("--cache-dir", storage["HF_HUB_CACHE"]),
                        ("--seed", "0"), ("--max-new-tokens", "512"), ("--context-limit", "4096"),
                        ("--expected-records", "475")):
        assert arguments[arguments.index(flag) + 1] == value
    assert ("--max-examples" in arguments) == (mode == "smoke")
    assert not list((root / "home").iterdir())
    assert not list((root / "scratch").rglob(".benchmark-write-check.*"))


@pytest.mark.parametrize("command,code", (("GIT", 128), ("SRUN", 17)))
def test_failed_command_logs_line_and_exact_status(shell_run, command, code):
    run, _, root = shell_run
    result = run(overrides={f"TEST_{command}_EXIT": str(code)})
    assert result.returncode == code
    assert re.search(rf"ERROR file=.* line=\d+ exit={code} command=", result.stderr)
    assert f"exit={code}" in result.stderr and "EXIT job=123 task=5" in result.stderr
    if command == "GIT":
        assert r"command=git\ rev-parse\ HEAD" in result.stderr
        assert not (root / "args").exists()  # No benchmark process reached.
    else:
        assert "command=srun" in result.stderr


@pytest.mark.parametrize("name", ("BENCH_SCRATCH_ROOT", "BENCH_OUTPUT_ROOT", "HF_HOME", "HF_HUB_CACHE",
                                  "TMPDIR", "XDG_CACHE_HOME", "TRITON_CACHE_DIR", "TRANSFORMERS_CACHE"))
def test_home_paths_are_rejected_before_gpu(shell_run, name):
    run, env, root = shell_run
    result = run(overrides={name: env["HOME"]})
    assert result.returncode != 0
    assert "HOME" in result.stderr and not (root / "args").exists()
    assert not list((root / "home").iterdir())


def test_explicit_storage_paths_and_conflicting_hub_cache(shell_run):
    run, env, root = shell_run
    cache = env["BENCH_SCRATCH_ROOT"] + "/existing-cache"
    output = env["BENCH_SCRATCH_ROOT"] + "/archived-layout"
    artifact = root / "scratch/archived-layout/previous-run/predictions.jsonl"
    artifact.parent.mkdir(parents=True)
    artifact.write_bytes(b"original experiment artifact\n")
    result = run(overrides={"HF_HUB_CACHE": cache, "BENCH_CACHE_DIR": cache, "BENCH_OUTPUT_ROOT": output})
    assert result.returncode == 0, result.stderr
    storage = (root / "storage").read_text()
    assert f"HF_HUB_CACHE={cache}" in storage and f"BENCH_OUTPUT_ROOT={output}" in storage
    assert artifact.read_bytes() == b"original experiment artifact\n"
    (root / "args").unlink()
    result = run(overrides={"HF_HUB_CACHE": cache, "BENCH_CACHE_DIR": cache + "-different"})
    assert result.returncode != 0 and "same model cache" in result.stderr
    assert not (root / "args").exists()


def test_venv_interpreter_symlink_is_not_replaced_by_system_python(shell_run):
    run, env, root = shell_run
    # Emulate a normal POSIX venv symlink, without requiring Windows symlink privileges.
    realpath = root / "commands/realpath"
    realpath.write_text('#!/bin/bash\nif [[ "${@: -1}" = "$BENCH_PYTHON" ]]; then\n'
                        '  printf "/usr/bin/python3\\n"\nelse\n  exec /usr/bin/realpath "$@"\nfi\n',
                        encoding="utf-8", newline="\n")
    realpath.chmod(0o755)
    result = run()
    assert result.returncode == 0, result.stderr
    assert f"BENCH_PYTHON={env['BENCH_PYTHON']}" in (root / "storage").read_text()


@pytest.mark.parametrize("relative", (True, False))
def test_invalid_output_destination_fails_before_gpu_and_preserves_files(shell_run, relative):
    run, env, root = shell_run
    existing = root / "scratch/file-not-directory"
    existing.write_bytes(b"keep this file")
    destination = "relative/output" if relative else shell_path(existing)
    result = run(overrides={"BENCH_OUTPUT_ROOT": destination})
    assert result.returncode != 0 and not (root / "args").exists()
    assert existing.read_bytes() == b"keep this file"


def test_missing_scratch_root_does_not_fall_back_to_home(shell_run):
    run, _, root = shell_run
    result = run(overrides={"BENCH_SCRATCH_ROOT": ""})
    assert result.returncode != 0 and "BENCH_SCRATCH_ROOT" in result.stderr
    assert not (root / "args").exists()


@pytest.mark.parametrize("custom", (False, True))
def test_submission_routes_both_logs_before_slurm_starts(shell_run, custom):
    run, env, root = shell_run
    log_dir = env["BENCH_SCRATCH_ROOT"] + ("/custom-logs" if custom else "/logs/slurm")
    overrides = {"BENCH_LOG_DIR": log_dir} if custom else {}
    result = run("submit_benchmark.sh", overrides=overrides)
    assert result.returncode == 0, result.stderr
    arguments = (root / "args").read_text().splitlines()
    assert f"--output={log_dir}/%x-%A_%a.out" in arguments
    assert f"--error={log_dir}/%x-%A_%a.err" in arguments
    assert "--export=ALL" in arguments and "--open-mode=append" in arguments
    assert "--array=0,3,5%1" in arguments
    assert arguments[-3:] == ["ministral", "smoke", "v1"]
    assert not list((root / "home").iterdir())


def test_submission_rejects_home_logs(shell_run):
    run, env, root = shell_run
    result = run("submit_benchmark.sh", overrides={"BENCH_LOG_DIR": env["HOME"]})
    assert result.returncode != 0 and "outside HOME" in result.stderr
    assert not (root / "args").exists()


@pytest.mark.parametrize("selection,expected", [(None, "0,3,5%1"), ("all", "0,3,5%1"),
                                               ("0", "0"), ("3", "3"), ("5", "5")])
@pytest.mark.parametrize("model,mode", [("ministral", "smoke"), ("granite", "full")])
def test_explicit_shot_selection_passes_exact_sbatch_arguments(shell_run, selection, expected, model, mode):
    run, env, root = shell_run
    # A stale inherited Slurm selector must not expand a single-shot retry or
    # unexpectedly narrow the documented default array.
    overrides = {"SBATCH_ARRAY_INX": "0,3,5%1" if selection == "5" else "5"}
    if selection is not None:
        overrides["BENCH_SHOTS"] = selection
    result = run("submit_benchmark.sh", args=(model, mode, "v1"), overrides=overrides)
    assert result.returncode == 0, result.stderr
    log_dir = env["BENCH_SCRATCH_ROOT"] + "/logs/slurm"
    assert (root / "args").read_text().splitlines() == [
        "--export=ALL", "--open-mode=append", f"--array={expected}",
        f"--output={log_dir}/%x-%A_%a.out", f"--error={log_dir}/%x-%A_%a.err",
        shell_path(SCRIPTS / "benchmark.sbatch"), model, mode, "v1",
    ]


@pytest.mark.parametrize("selection", ("", "1", "2", "4", "05", "0,3,5", "5%1", " 5", "5 ", "ALL", "5;exit 0"))
def test_invalid_shot_selection_never_submits(shell_run, selection):
    run, _, root = shell_run
    result = run("submit_benchmark.sh", overrides={"BENCH_SHOTS": selection})
    assert result.returncode == 2 and "BENCH_SHOTS must be" in result.stderr
    assert not (root / "args").exists()
    assert not (root / "scratch/logs").exists()  # Reject before storage side effects.
