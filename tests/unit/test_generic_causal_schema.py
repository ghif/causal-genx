import csv

import jax
import numpy as np
import pytest
from flax import nnx

from causal.generic_scm import SchemaDrivenSCM, validate_encoded_values
from config import load_experiment
from contracts import CausalGraphSpec, VariableKind, VariableSpec
from data.padchest import (
    PAD_CHEST_V2_SCHEMA,
    PadChestDataset,
    audit_padchest_v2_rows,
    derive_padchest_age_group,
    derive_padchest_v2_row,
)


def test_generic_mixed_schema_has_parent_conditioned_heads_and_valid_samples():
    schema = CausalGraphSpec(
        "toy",
        (
            VariableSpec("group", VariableKind.CATEGORICAL, encoded_dim=3, categories=("a", "b", "c")),
            VariableSpec("flag", VariableKind.BINARY),
            VariableSpec("score", VariableKind.CONTINUOUS),
            VariableSpec("outcome", VariableKind.CATEGORICAL, encoded_dim=2),
        ),
        edges=(("group", "outcome"), ("flag", "outcome")),
    )
    model = SchemaDrivenSCM(schema, widths=(8,), rngs=nnx.Rngs(0))
    samples = model.sample(16, jax.random.PRNGKey(0))
    validate_encoded_values(samples, schema)
    assert samples["pa"].shape == (16, schema.encoded_dim)
    assert model.log_prob(**{name: samples[name] for name in schema.variable_names})["joint"].shape == (16,)
    assert model.logits("outcome", {"group": samples["group"], "flag": samples["flag"]}).shape == (16, 2)


def test_padchest_v2_derivation_and_audit():
    row = {
        "StudyDate_DICOM": "20140101",
        "PatientBirth": "2000",
        "PatientSex_DICOM": "M",
        "Labels": "['tuberculosis']",
        "Pediatric": "PED",
        "Projection": "AP_horizontal",
        "ViewPosition_DICOM": "POSTEROANTERIOR",
        "Manufacturer_DICOM": "PhilipsMedicalSystems",
    }
    assert derive_padchest_age_group("20140101", "2000") == "5-17"
    encoded = derive_padchest_v2_row(row)
    assert sum(value.size for key, value in encoded.items() if key != "_audit") == 21
    assert np.argmax(encoded["age_group"]) == 1
    assert np.argmax(encoded["scanner"]) == 1
    assert encoded["tb_status"].tolist() == [1.0]
    audit = audit_padchest_v2_rows([row])
    assert audit["scanner_counts"]["PhilipsMedicalSystems"] == 1
    assert audit["pediatric_disagreement_count"] == 0


def test_padchest_v2_unknown_scanner_is_not_collapsed():
    with pytest.raises(ValueError, match="scanner"):
        derive_padchest_v2_row({"StudyDate_DICOM": "20140101", "PatientBirth": "2000", "Manufacturer_DICOM": "Other"})


def test_v2_config_is_typed_and_uses_distinct_artifact_prefix():
    config = load_experiment("configs/padchest_scm_v2_tpu_v6e1.yaml")
    assert config.model.context_dim == PAD_CHEST_V2_SCHEMA.encoded_dim == 21
    assert config.artifacts.run_name != "scm_padchest_tpu_v6e1"
    assert config.artifacts.remote_root.endswith("checkpoints_v2")


def test_scm_metadata_batch_never_reads_padchest_images(tmp_path, monkeypatch):
    path = tmp_path / "metadata.csv"
    fields = ["PatientID", "ImageID", "StudyDate_DICOM", "PatientBirth", "Labels", "PatientSex_DICOM", "Pediatric", "Projection", "ViewPosition_DICOM", "Manufacturer_DICOM"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for index in range(10):
            writer.writerow({"PatientID": str(index), "ImageID": f"{index}.png", "StudyDate_DICOM": "20140101", "PatientBirth": "2000", "Labels": "[]", "PatientSex_DICOM": "M", "Pediatric": "No", "Projection": "PA", "ViewPosition_DICOM": "PA", "Manufacturer_DICOM": "PhilipsMedicalSystems"})
    dataset = PadChestDataset(str(tmp_path), str(path), str(tmp_path), "train", 8, "tb_or_sequelae", schema=PAD_CHEST_V2_SCHEMA)
    monkeypatch.setattr(dataset, "_get_image", lambda _index: (_ for _ in ()).throw(AssertionError("image read")))
    variables = {name: values[:2] for name, values in dataset.samples.items()}
    assert set(variables) == set(PAD_CHEST_V2_SCHEMA.variable_names)
    assert all(value.shape[0] == 2 for value in variables.values())
