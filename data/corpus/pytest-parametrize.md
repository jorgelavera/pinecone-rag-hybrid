# Pytest parametrize

`@pytest.mark.parametrize` runs the same test body against several inputs. Each
input becomes a separate test item with its own node id and its own pass/fail
line, so one broken case does not hide the cases that still work.

```python
import pytest

@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("7", 7),
        ("-1", -1),
        ("0", 0),
    ],
)
def test_parse_int(raw, expected):
    assert parse_int(raw) == expected
```

The first argument names the parameter, or is a comma-separated string of names
for multi-parameter cases. The second argument is an iterable of value tuples.

## Stacking and indirect parameters

Decorators stack: two `parametrize` decorators produce the cartesian product, so
three widths times two heights yields six tests. Passing `indirect=True` routes
the parameter into a fixture instead of the test body, which is how a fixture is
exercised across several configurations without duplicating setup code:

```python
@pytest.mark.parametrize("client", ["sqlite", "postgres"], indirect=True)
def test_repository(client):
    assert client.ping()
```

## Readable identifiers

By default the generated node id is derived from the parameter values, and
awkward values such as objects or dicts become `param0`, `param1`. Supply
`ids=` for readable names, or wrap a case in `pytest.param` to attach both an id
and a skip marker:

```python
@pytest.mark.parametrize(
    "payload",
    [
        pytest.param({"version": 1}, id="v1"),
        pytest.param({"version": 2}, id="v2"),
        pytest.param({"version": 99}, id="unsupported", marks=pytest.mark.xfail),
    ],
)
def test_payload(payload):
    validate(payload)
```

Readable ids matter in reports: a failing case shows as
`test_payload[unsupported]` instead of `test_payload[payload2]`, which is the
difference between a useful CI failure and a scavenger hunt. When several
parameters share names, pytest joins their ids with a dash; use `ids=` with an
explicit callable when the default concatenation is ambiguous.
