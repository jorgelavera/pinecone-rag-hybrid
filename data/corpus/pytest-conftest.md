# Pytest conftest

`conftest.py` is pytest's local plugin file. Any fixture, hook, or marker helper
defined there becomes available to every test in that directory and its
subdirectories without an import, which makes it the natural home for fixtures
that many test modules share.

```
tests/
├── conftest.py          # fixtures shared by the whole suite
├── unit/
│   ├── conftest.py      # fixtures shared only by unit tests
│   └── test_parser.py
└── integration/
    └── test_api.py
```

Fixtures follow directory scope: a fixture in `tests/conftest.py` is visible to
`tests/unit/test_parser.py`, while a fixture in `tests/unit/conftest.py` is not
visible to `tests/integration/test_api.py`. When two conftest files define the
same fixture name, the one closer to the test wins, which lets a package
override a global fixture without touching it.

## Sharing fixtures across modules

Move the fixture into the nearest common `conftest.py` and delete the local
copy. Because conftest files are imported by pytest rather than by the test
module, tests never import them explicitly; they only name the fixture. That
keeps test modules free of fixture plumbing and makes the shared contract
discoverable by reading one file.

Keep conftest modules small and focused. A conftest that imports application
code at module scope also makes collection fail for every test in the directory
when that import breaks, so prefer lazy imports inside the fixture body.

## Hooks as well as fixtures

Conftest files can implement pytest hooks such as `pytest_configure`,
`pytest_collection_modifyitems`, and `pytest_sessionfinish`. A common pattern is
to register custom markers so `-m slow` works, or to skip integration tests when
credentials are missing:

```python
def pytest_collection_modifyitems(config, items):
    skip = pytest.mark.skip(reason="needs credentials")
    for item in items:
        if "integration" in item.keywords:
            item.add_marker(skip)
```

Prefer a conftest hook over a wrapper script when the behaviour must apply to
every invocation, including a bare `pytest` run in Continuous Integration.
