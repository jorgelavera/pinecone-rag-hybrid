# Pytest markers

A marker is metadata attached to a test with the `@pytest.mark.<name>`
decorator. Markers let you select, skip, or annotate tests without changing how
they run, which makes them the standard way to separate slow or integration
coverage from the fast default suite.

```python
import pytest

@pytest.mark.slow
def test_full_reindex():
    ...
```

## Selecting tests by marker

`pytest -m slow` runs only the tests marked `slow`. The expression supports
boolean logic, so `-m "slow and not network"` narrows further, and `-m "not
slow"` is the usual local-development default. Marker selection happens during
collection, so deselected tests are reported as such and never execute their
fixtures.

This is the mechanism behind "run only the tests marked as slow": register the
marker, decorate the expensive tests, and select them explicitly in the nightly
job instead of running them on every save.

## Registering custom markers

An unregistered marker still runs, but pytest prints a `PytestUnknownMarkWarning`
for every use, and with `--strict-markers` an unknown marker becomes a hard
error. Register them in `pyproject.toml`:

```toml
[tool.pytest.ini_options]
markers = [
  "slow: full-suite or container-backed tests",
  "integration: needs live credentials",
]
```

Registering is what makes `-m` reliable: a typo such as `@pytest.mark.slwo`
fails loudly instead of silently selecting nothing. The same registration block
can list markers used by a plugin so one file documents the whole vocabulary.

## Built-in skip and xfail markers

`@pytest.mark.skip` removes a test unconditionally, while
`@pytest.mark.skipif(condition, reason=...)` skips it only when the condition
holds, which is how credential-dependent tests stay green on a machine without
keys. `@pytest.mark.xfail` runs the test and marks an expected failure as
acceptable; use `strict=True` so an unexpectedly passing test also fails. Both
markers accept `reason=`, and that string appears in the short test summary, so
write it for the person reading a CI log at midnight.
