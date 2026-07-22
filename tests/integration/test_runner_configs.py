import subprocess
import sys
from pathlib import Path

from config import load_experiment


ROOT = Path(__file__).resolve().parents[2]


def test_each_standalone_config_validates_with_runner():
    for command, config in (
        ("train-image-model", "morphomnist_image_model.yaml"),
        ("train-image-model", "morphomnist_image_model_tpu_v6e4.yaml"),
        ("train-scm", "morphomnist_scm.yaml"),
        ("train-predictor", "morphomnist_predictor.yaml"),
        ("finetune-counterfactual", "morphomnist_counterfactual.yaml"),
    ):
        result = subprocess.run(
            [sys.executable, "scripts/run.py", command, "--config", f"configs/{config}", "--dry-run"],
            cwd=ROOT,
            text=True,
            capture_output=True,
            check=True,
        )
        assert f"stage={command}" in result.stdout


def test_scm_dry_run_reports_legacy_run_directory():
    result = subprocess.run(
        [sys.executable, "scripts/run.py", "train-scm", "--config", "configs/morphomnist_scm.yaml", "--dry-run"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=True,
    )
    config = load_experiment(ROOT / "configs/morphomnist_scm.yaml")
    expected = Path(config.artifacts.root) / config.dataset.name / config.artifacts.run_name
    assert str(expected) in result.stdout
