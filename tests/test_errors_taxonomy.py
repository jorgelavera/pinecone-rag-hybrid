"""Exception taxonomy: one root, named subclasses, stable remediation contract."""

import pytest

from rag import errors


ROOT = "RagError"
DIRECT_SUBCLASSES = (
    "CorpusTooSmallError",
    "MetadataTooLargeError",
    "DimensionMismatchError",
    "IndexConfigMismatchError",
    "GoldenSetError",
    "HydrationError",
    "RetrievalEmptyError",
)


def test_root_error_is_an_exception():
    assert issubclass(errors.RagError, Exception)


@pytest.mark.parametrize("name", DIRECT_SUBCLASSES)
def test_each_failure_mode_inherits_from_the_root(name):
    error_class = getattr(errors, name)
    assert issubclass(error_class, errors.RagError)
    assert error_class.__mro__[1] is errors.RagError


def test_missing_config_is_a_config_error():
    assert issubclass(errors.MissingConfigError, errors.ConfigError)
    assert issubclass(errors.ConfigError, errors.RagError)


def test_message_is_preserved_for_actionable_remediation():
    raised = errors.MissingConfigError("Missing required configuration: INDEX_NAME")
    assert str(raised) == "Missing required configuration: INDEX_NAME"
    assert isinstance(raised, errors.ConfigError)
