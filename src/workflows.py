"""Deprecated compatibility aliases; use the named modules in :mod:`training`."""

from training.counterfactual import run as counterfactual
from training.image_model import run as train_image
from training.inference import run as infer
from training.predictor import run as train_predictor
from training.scm import output_dir as scm_output_dir
from training.scm import run as train_scm

__all__ = ["counterfactual", "infer", "scm_output_dir", "train_image", "train_predictor", "train_scm"]
