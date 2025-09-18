import pinax


def test_version():
    version = getattr(pinax, "__version__", None)
    assert version is not None
    assert isinstance(version, str)
    assert len(version) > 0
    assert len(version) > 0
    assert len(version) > 0
