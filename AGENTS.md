# Repository Guidelines

## Project Structure & Module Organization

`src/` contains reusable implementation code. The four research stages live in
`src/training/`: SCM, predictor, image-model, and counterfactual fine-tuning.
Dataset loading and conditioning are in `src/data/`; causal mechanisms are in
`src/causal/`; VAE/HVAE code is in `src/models/`. Treat `src/pgm/` and
`src/main.py` as compatibility layers: keep them working, but put new logic in
the named stage modules.

Use `scripts/run.py` as the public entry point. Each complete experiment has
one standalone YAML file in `configs/`, for example
`configs/morphomnist_predictor.yaml`. Tests belong in `tests/`: parity tests at
the root, plus `unit/`, `contract/`, and `integration/` coverage.

## Build, Test, and Development Commands

Install the backend-appropriate dependencies, then run tests from the
repository root:

```bash
pip install -r requirements.txt       # CPU development environment
PYTHONPATH=src pytest -q              # complete test suite
python scripts/run.py train-scm --config configs/morphomnist_scm.yaml --dry-run
python scripts/run.py train-predictor --config configs/morphomnist_predictor.yaml
```

Use `--dry-run` to validate a config without loading data or training. Runtime
selection occurs in `scripts/run.py`; do not import JAX-dependent code before
the runtime is configured.

## Coding Style & Naming Conventions

Use Python 3.10+ type hints, four-space indentation, `snake_case` functions
and modules, `PascalCase` classes, and concise module docstrings. Keep stage
modules focused on one workflow. Preserve parity-sensitive numerical code and
legacy checkpoint payload fields unless a migration explicitly changes them.

## Testing Guidelines

Pytest discovers `tests/` with `src/` on its import path. Name files
`test_<area>.py` and tests `test_<behavior>`. Add a unit test for validation or
artifact contracts, and a smoke/contract test when changing a stage. Run the
relevant parity suite before modifying SCM, predictor, or counterfactual math.

## Commit & Pull Request Guidelines

Use short, imperative commit subjects consistent with history, such as `fix
predictor checkpoint resume` or `add SCM artifact contract`. Keep commits
single-purpose. Pull requests should describe the affected stage, config and
artifact compatibility impact, tests run, and any required data/backend setup.
