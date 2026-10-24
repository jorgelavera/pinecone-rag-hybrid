# Pytest basics

Pytest is a test runner that discovers files named `test_*.py` or `*_test.py`,
imports them, and executes every function whose name starts with `test_`. There
is no base class to inherit and no registration step: a plain function is a test.

```python
# tests/test_arithmetic.py
def test_addition():
    assert 2 + 2 == 4
```

Running `pytest` from the project root collects that function and reports one
passing test. Running `pytest -q` prints a compact summary, and `pytest -v`
prints one line per test with its node id such as
`tests/test_arithmetic.py::test_addition`. A node id is the address you use to
run a single test in isolation, which is the fastest way to iterate while
debugging.

## Plain assertions

Pytest rewrites `assert` statements so a failure prints the actual intermediate
values instead of a bare `AssertionError`. That means you should write ordinary
Python comparisons rather than a library of `assertEqual` helpers:

```python
def test_payload_shape():
    payload = {"status": "ok", "items": [1, 2, 3]}
    assert payload["status"] == "ok"
    assert len(payload["items"]) == 3
```

When the second assertion fails, the report shows both the expression and the
value of `len(payload["items"])`. For approximate floats use `pytest.approx`,
and for exceptions use `pytest.raises` as a context manager.

## Exit codes

The exit code is what Continuous Integration reads. `0` means every collected
test passed, `1` means at least one test failed, `2` means collection was
interrupted (a syntax error or an import problem), and `5` means no tests were
collected at all. A CI job that only checks "did the command fail" will miss the
difference between a red suite and a suite that silently stopped collecting, so
pin the expected code explicitly.

## Useful command line flags

- `-q` / `-v` control report verbosity.
- `-x` stops at the first failure, which is useful in a tight edit loop.
- `--lf` re-runs only the tests that failed last time.
- `-k "fixture and not slow"` selects tests by name expression.
- `-p no:cacheprovider` disables the `.pytest_cache` directory.

Failures are reported with a short test summary followed by a full traceback,
and every fixture that participated in the failing test is listed underneath so
you can see which setup code ran.
