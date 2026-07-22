# Adding a dataset domain

1. Add an immutable `CausalGraphSpec` in `src/data/<dataset>.py`.
   Keep variables named, ordered by the schema, and assign the exact encoded dimension used by the image mechanism.
2. Implement a provider in the same module with a `DatasetSpec`, `load_split`, `make_batch`, and stable `fingerprint`.
   `make_batch` must return NCHW images and a mapping of named variables; it must not create a `pa` field.
3. Register the provider in the data registry using its stable dataset name and expose it through `data/__init__.py`.
4. Add one complete experiment YAML under `configs/` and a contract test under `tests/contract/` that verifies split availability, variable dimensions, deterministic evaluation behavior, and a stable fingerprint.
5. If the dataset has a custom SCM, implement the `StructuralCausalModel` protocol and compose it with the generic `DeepStructuralCausalModel` rather than indexing variables in workflow code.

MorphoMNIST is the reference implementation. Keep raw storage, normalization constants, and transform policies inside the dataset provider; shared workflows should only consume contracts.
