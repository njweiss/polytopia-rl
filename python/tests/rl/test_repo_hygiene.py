"""Milestone 16b tests 8: repo and workflow hygiene.

Checks (docs/MILESTONE_16b.md, Rule 8, Clarifications B5 & B6):
- Empty `src/polytopia_rl/` directory is removed.
- Stale lines in `CLAUDE.md` and `docs/ORCHESTRATOR.md` are cleaned up (`docs/SPEC.md` is the sole
  authoritative spec; `docs/STATUS.md` holds current status).
- `pixi.toml` supports `osx-arm64` (CPU `pytorch`) alongside `linux-64` (CUDA 12.9 `pytorch` and
  `cuda-cudart-dev` with identical pins).
- `scripts/py_check.sh` and `scripts/rl_check.sh` isolate the built Python extension to
  `target/pyext` inside a git worktree without running `maturin develop`, while preserving main
  checkout behavior.
"""

import json
import os
import pathlib
import shutil
import stat
import subprocess
import sys
import tomllib

REPO = pathlib.Path(__file__).resolve().parents[3]


def test_orphan_src_polytopia_rl_is_removed():
    assert not (REPO / "src" / "polytopia_rl").exists(), "empty orphan src/polytopia_rl/ should be deleted"


def test_docs_stale_sections_are_cleaned():
    claude_md = (REPO / "CLAUDE.md").read_text()
    assert "Future crates (not yet)" not in claude_md, "CLAUDE.md still lists implemented crates as 'not yet'"

    orch_md = (REPO / "docs" / "ORCHESTRATOR.md").read_text()
    assert "Gate 11b" not in orch_md, "docs/ORCHESTRATOR.md still contains the stale M11 status block"
    assert "docs/STATUS.md" in orch_md
    assert "The Claude Doc is the master spec" not in orch_md, "docs/ORCHESTRATOR.md still names Claude Doc as master (B5)"


def test_pixi_toml_adds_osx_arm64_and_preserves_linux_cuda_pins():
    data = tomllib.loads((REPO / "pixi.toml").read_text())
    platforms = data["workspace"]["platforms"]
    has_osx = any(p == "osx-arm64" or (isinstance(p, dict) and p.get("platform") == "osx-arm64") for p in platforms)
    has_linux_cuda = any(isinstance(p, dict) and p.get("platform") == "linux-64" and p.get("cuda") == "12.9" for p in platforms)
    assert has_osx, f"osx-arm64 missing from workspace.platforms: {platforms}"
    assert has_linux_cuda, f"linux-64 CUDA 12.9 platform missing: {platforms}"

    top_deps = data.get("dependencies", {})
    assert "cuda-cudart-dev" not in top_deps, "cuda-cudart-dev must move under target.linux-64.dependencies"
    assert not (isinstance(top_deps.get("pytorch"), dict) and "cuda" in str(top_deps["pytorch"].get("build", ""))), (
        "CUDA pytorch build pin must move under target.linux-64.dependencies"
    )

    linux_deps = data.get("target", {}).get("linux-64", {}).get("dependencies", {})
    assert linux_deps.get("cuda-cudart-dev") == "12.9.*"
    assert linux_deps.get("pytorch") == {"version": ">=2.14.1,<3", "build": "cuda129_generic*"}

    osx_deps = data.get("target", {}).get("osx-arm64", {}).get("dependencies", {})
    assert "pytorch" in osx_deps, "target.osx-arm64.dependencies must include CPU pytorch"
    assert "cuda-cudart-dev" not in osx_deps
    assert "cuda" not in str(osx_deps["pytorch"])


def _write_exec(path: pathlib.Path, content: str) -> None:
    path.write_text(content)
    path.chmod(path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)


def _git(cwd: pathlib.Path, *args: str) -> None:
    subprocess.run(
        ["git", "-c", "user.name=Test", "-c", "user.email=test@example.com", *args],
        cwd=cwd,
        check=True,
        capture_output=True,
        text=True,
    )


def test_py_check_and_rl_check_are_worktree_safe(tmp_path):
    """Run the real `scripts/py_check.sh` and `scripts/rl_check.sh` in a temporary git repo and a
    real git worktree with lightweight tool shims, verifying worktree isolation and main-checkout
    behavior."""
    main_repo = tmp_path / "main"
    (main_repo / "scripts").mkdir(parents=True)
    (main_repo / "python").mkdir()
    (main_repo / "python" / "pyproject.toml").write_text("[project]\nname = 'polytopia-sim'\n")
    for name in ("py_check.sh", "rl_check.sh"):
        dst = main_repo / "scripts" / name
        shutil.copy2(REPO / "scripts" / name, dst)
        dst.chmod(dst.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)

    _git(main_repo, "init")
    _git(main_repo, "add", "scripts/py_check.sh", "scripts/rl_check.sh", "python/pyproject.toml")
    _git(main_repo, "commit", "-m", "init")

    wt = tmp_path / "wt"
    _git(main_repo, "worktree", "add", "-b", "wt-branch", str(wt))

    log_file = tmp_path / "calls.jsonl"
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()

    # Fake `cargo`: just logs its invocation.
    _write_exec(
        bin_dir / "cargo",
        f"""#!{sys.executable}
import json, os, sys
with open({str(log_file)!r}, "a") as f:
    f.write(json.dumps({{"tool": "cargo", "argv": sys.argv[1:], "cwd": os.getcwd()}}) + "\\n")
""",
    )

    # Fake `maturin`: logs invocation; on `build`, writes a real tiny `.whl` zip to `-o`/`--out`
    # (default `../target/wheels`).
    _write_exec(
        bin_dir / "maturin",
        f"""#!{sys.executable}
import json, os, pathlib, sys, zipfile
args = sys.argv[1:]
with open({str(log_file)!r}, "a") as f:
    f.write(json.dumps({{"tool": "maturin", "argv": args, "cwd": os.getcwd()}}) + "\\n")
if args and args[0] == "build":
    out_dir = None
    for i, a in enumerate(args):
        if a in ("-o", "--out") and i + 1 < len(args):
            out_dir = pathlib.Path(args[i + 1])
        elif a.startswith("--out="):
            out_dir = pathlib.Path(a.split("=", 1)[1])
    if out_dir is None:
        out_dir = pathlib.Path("..") / "target" / "wheels"
    out_dir.mkdir(parents=True, exist_ok=True)
    whl = out_dir / "polytopia_sim-0.1.0-cp312-cp312-macosx_11_0_arm64.whl"
    with zipfile.ZipFile(whl, "w") as zf:
        zf.writestr("polytopia_sim/__init__.py", "MARKER = 'worktree'\\n")
""",
    )

    # Fake `python3`: intercepts `-m pytest` to log PYTHONPATH and POLYTOPIA_BIN; delegates all
    # other invocations (e.g. `-m zipfile`, `-c ...`) to the real Python interpreter.
    _write_exec(
        bin_dir / "python3",
        f"""#!{sys.executable}
import json, os, subprocess, sys
args = sys.argv[1:]
if len(args) >= 2 and args[0] == "-m" and args[1] == "pytest":
    with open({str(log_file)!r}, "a") as f:
        f.write(json.dumps({{
            "tool": "pytest",
            "argv": args,
            "cwd": os.getcwd(),
            "pythonpath": os.environ.get("PYTHONPATH", ""),
            "polytopia_bin": os.environ.get("POLYTOPIA_BIN", ""),
        }}) + "\\n")
    sys.exit(0)
sys.exit(subprocess.call([{sys.executable!r}, *args]))
""",
    )

    env = dict(os.environ)
    env["PATH"] = f"{bin_dir}{os.pathsep}{env.get('PATH', '')}"
    env.pop("PYTHONPATH", None)

    def read_calls():
        calls = [json.loads(line) for line in log_file.read_text().splitlines() if line.strip()]
        log_file.write_text("")
        return calls

    # 1. Worktree: py_check.sh must build wheel -> unpack target/pyext -> prepend to PYTHONPATH.
    res = subprocess.run([str(wt / "scripts" / "py_check.sh")], cwd=wt, env=env, capture_output=True, text=True)
    assert res.returncode == 0, res.stderr
    wt_calls = read_calls()
    mat_calls = [c for c in wt_calls if c["tool"] == "maturin"]
    assert len(mat_calls) == 1 and mat_calls[0]["argv"][0] == "build", f"worktree must run maturin build, not develop: {mat_calls}"
    assert (wt / "target" / "pyext" / "polytopia_sim" / "__init__.py").exists(), "wheel must be unpacked into target/pyext"
    assert not (main_repo / "target").exists(), "worktree check must not touch main checkout"
    py_calls = [c for c in wt_calls if c["tool"] == "pytest"]
    assert len(py_calls) == 1
    wt_pyext = str((wt / "target" / "pyext").resolve())
    pp_entries = [str(pathlib.Path(p).resolve()) for p in py_calls[0]["pythonpath"].split(os.pathsep) if p]
    assert wt_pyext in pp_entries, f"worktree py_check.sh must add target/pyext to PYTHONPATH: {py_calls[0]['pythonpath']}"

    # 2. Worktree: rl_check.sh must also put target/pyext on PYTHONPATH.
    res = subprocess.run([str(wt / "scripts" / "rl_check.sh")], cwd=wt, env=env, capture_output=True, text=True)
    assert res.returncode == 0, res.stderr
    rl_calls = read_calls()
    assert len(rl_calls) == 1 and rl_calls[0]["tool"] == "pytest"
    rl_pp = [str(pathlib.Path(p).resolve()) for p in rl_calls[0]["pythonpath"].split(os.pathsep) if p]
    assert wt_pyext in rl_pp, f"worktree rl_check.sh must add target/pyext to PYTHONPATH: {rl_calls[0]['pythonpath']}"

    # 3. Main checkout: py_check.sh still runs `maturin develop` and does not create target/pyext.
    res = subprocess.run([str(main_repo / "scripts" / "py_check.sh")], cwd=main_repo, env=env, capture_output=True, text=True)
    assert res.returncode == 0, res.stderr
    main_calls = read_calls()
    main_mat = [c for c in main_calls if c["tool"] == "maturin"]
    assert len(main_mat) == 1 and main_mat[0]["argv"][0] == "develop", f"main checkout must keep maturin develop: {main_mat}"
    assert not (main_repo / "target" / "pyext").exists()

    # 4. Main checkout: rl_check.sh does not inject target/pyext.
    res = subprocess.run([str(main_repo / "scripts" / "rl_check.sh")], cwd=main_repo, env=env, capture_output=True, text=True)
    assert res.returncode == 0, res.stderr
    main_rl = read_calls()
    assert len(main_rl) == 1 and "pyext" not in main_rl[0]["pythonpath"]
