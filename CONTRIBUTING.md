# Contributing to JOLT

## Workflow

* `main` is protected. Every change goes through a pull request from a branch
  (`feature/...`, `fix/...`, `docs/...`), and CI must pass before merging.
* Install the development tools and the git hooks once:

  ```bash
  pip install --no-deps -e .
  pre-commit install
  ```

* Before opening a pull request, run `pre-commit run --all-files` and `pytest`.
* Changes that can alter numerical output are checked by the maintainers against the CZI DS-10441
  benchmark before merging; say in the pull request which outputs you expect to change
  (`benchmarks/README.md` describes the single-tilt-series check you can run yourself).

## Style

* US English everywhere: code, comments, docstrings, documentation, and command-line help
  ("center", "behavior", "neighbor"). `codespell` checks this.
* Docstrings follow the numpy convention; lines are at most 120 characters. `ruff` checks both.
* Command-line flags use underscores (`--tomo_dims_xyz`, not `--tomo-dims-xyz`).
* Each command lives in `jolt/commands/<name>.py` and provides `add_args(parser=None)` and `main(args)`.
  Keep heavy imports (torch, scipy) inside `main` so that `jolt --help` stays fast.
* Never write paths specific to one computer or cluster into the repository.
* `jolt.yaml` records inputs and settings only. It never stores git or environment information.
* Fail loudly: a step that fails or falls back must say so in the log and in `run_status.json`,
  never silently.
