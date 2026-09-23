from config import ExperimentConfig
from training.predictor import _run_arguments


def _predictor_config(dataset_overrides=None):
    raw = {
        "seed": 11,
        "dataset": {
            "name": "padchest",
            "root": "gs://external-cxr-dataset/padchest/unpacked/images-224",
            "input_res": 224,
            "pad": 0,
        },
        "runtime": {"accelerator": "cpu", "precision": "fp32"},
        "artifacts": {"root": "checkpoints", "run_name": "padchest_predictor", "remote_root": ""},
        "optimizer": {"lr": 0.001, "weight_decay": 0.0, "batch_size": 4},
        "workflow": {"type": "train-predictor", "epochs": 1},
    }
    if dataset_overrides:
        raw["dataset"].update(dataset_overrides)
    return ExperimentConfig.model_validate(raw)


def test_predictor_run_arguments_carry_dataset_paths():
    config = _predictor_config(
        {
            "metadata": "gs://external-cxr-dataset/padchest/metadata/PADCHEST.csv",
            "image_prefix": "gs://external-cxr-dataset/padchest/unpacked/images-224",
        }
    )

    args = _run_arguments(config)

    assert args.data_dir == "gs://external-cxr-dataset/padchest/unpacked/images-224"
    assert args.metadata == "gs://external-cxr-dataset/padchest/metadata/PADCHEST.csv"
    assert args.image_prefix == "gs://external-cxr-dataset/padchest/unpacked/images-224"


def test_predictor_run_arguments_keep_dataset_path_defaults_empty():
    args = _run_arguments(_predictor_config({"name": "morphomnist"}))

    assert args.metadata == ""
    assert args.image_prefix == ""
