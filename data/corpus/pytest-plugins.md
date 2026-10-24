# Pytest plugins

A pytest plugin is a module that implements hook functions. Pytest calls every
registered implementation of a hook at the matching point in the run, which is
how fixtures, reporting, and collection are extended without forking the tool.

```python
# myplugin.py
def pytest_configure(config):
    config.addinivalue_line("markers", "slow: container-backed tests")
```

Installing the module as a package entry point under the `pytest11` group makes
it active everywhere. Simpler alternatives are a conftest file (local plugin,
active for one directory tree) or the `-p myplugin` command line flag (active
for one invocation). A conftest is a plugin that simply is not distributed.

## Hooks that run for every session

`pytest_configure` runs once at start-up and is where markers, ini options, and
global objects are registered. `pytest_collection_modifyitems` runs after every
test is collected and is the standard place to reorder, deselect, or skip items.
`pytest_sessionstart` and `pytest_sessionfinish` bracket the whole run, which is
where you start and stop an external service such as a test database.

```python
def pytest_collection_modifyitems(config, items):
    if config.getoption("--fast"):
        keep = [item for item in items if "slow" not in item.keywords]
        config.hook.pytest_deselected(items=[item for item in items if item not in keep])
        items[:] = keep
```

Hooks can also wrap execution. A `hookwrapper=True` generator yields once before
the actual hook runs and once after, which is how timing or logging plugins
observe results without replacing behaviour. Declare the hook name in a
`pytest_plugins` list at the top of a conftest to load a helper module as a
plugin without packaging it.

## Useful plugins in practice

`pytest-cov` adds coverage flags, `pytest-xdist` distributes tests across
workers with `-n auto`, `pytest-mock` provides a `mocker` fixture wrapping
`unittest.mock`, and `pytest-randomly` shuffles execution order to expose hidden
inter-test dependencies. Keep the plugin list short: every plugin adds hooks to
the same lifecycle, and a failing collection is much harder to debug when five
of them can influence it.
