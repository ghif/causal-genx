# Project agent memory

This file is the project's committed home for project-intrinsic agent knowledge: build, test, release, architecture, and sharp-edge notes that should travel with the code.

- Use the `med-jax` conda environment for JAX/Flax/Orbax validation commands in this checkout: `/mnt/data/miniconda/envs/med-jax/bin/python` with `PYTHONPATH=src`.
- PadChest SCM post-training split/diagnostic artifacts can be regenerated with `scripts/padchest_scm_posttrain.py`; see `journal/2026-09-23-padchest-scm-posttrain-validation-report.md` for the final step-200500 validation command and artifact paths.

## Maintaining this file

Keep this file for knowledge useful to almost every future agent session in this project.
Do not repeat what the codebase already shows; point to the authoritative file or command instead.
Prefer rewriting or pruning existing entries over appending new ones.
When updating this file, preserve this bar for all agents and keep entries concise.
