# Pytest mocking

Pytest ships two complementary mocking tools: the `monkeypatch` fixture for
temporary attribute, environment, and path changes, and `unittest.mock` for
recording calls and asserting interactions. Choosing between them is mostly
about intent: patch the environment with `monkeypatch`, and assert behaviour
with a mock object.

## monkeypatch versus a mocking library

`monkeypatch` mutates one attribute on one object and restores it automatically
at teardown. It is the smallest possible change, it needs no extra dependency,
and it fails loudly if the attribute does not exist:

```python
def test_reads_token_from_environment(monkeypatch):
    monkeypatch.setenv("API_TOKEN", "test-token")
    assert load_token() == "test-token"
```

Reach for `unittest.mock.patch` or `MagicMock` when you need to inspect
interactions: how many times a collaborator was called, in what order, with
which arguments, or when you need to simulate return values and side effects on
a large third-party surface. If the only goal is "this value must not leak into
the next test", `monkeypatch` is the right tool and a mock is overhead.

`monkeypatch.setattr(target, value)` also accepts `raising=False`, but prefer
the default so a typo in the attribute name fails instead of silently creating a
new attribute that nothing reads.

## Asserting call arguments

`assert_called_with` compares the most recent call, `assert_called_once_with`
requires exactly one call, and `assert_any_call` matches any call in the
history. Use `call_args_list` when you need to prove ordering:

```python
from unittest.mock import MagicMock

def test_notifier_receives_the_recipient():
    notifier = MagicMock()
    send_invoice(notifier, recipient="ops@example.test", total=42)

    notifier.send.assert_called_once_with("ops@example.test", 42)
```

Configure the mock with `spec=` so calls to methods that do not exist on the
real collaborator raise `AttributeError`; an unspecced `MagicMock` accepts any
attribute and happily approves a typo.

## Patching imports

Patch where the name is looked up, not where it is defined. If
`reports/export.py` does `from storage import save`, then
`monkeypatch.setattr("reports.export.save", fake_save)` is correct, while
patching `storage.save` leaves the already-imported reference untouched. This
single rule explains most "the mock was never called" failures.
