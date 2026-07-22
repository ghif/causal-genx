from datasets import MORPHOMNIST_SCHEMA, MorphoMNISTProvider


def test_morphomnist_contract_is_explicit_without_loading_remote_data():
    provider = MorphoMNISTProvider(root="/tmp/morphomnist")
    assert provider.spec.image.channels == 1
    assert provider.schema.variable_names == ("thickness", "intensity", "digit")
    assert provider.schema.encoded_dim == 12
    assert len(provider.fingerprint()) == 16
