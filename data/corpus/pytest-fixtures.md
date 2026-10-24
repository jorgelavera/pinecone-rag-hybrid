# Pytest fixtures

A fixture is a function decorated with `@pytest.fixture` that supplies data or
state to a test. The test declares the fixture name as a parameter and pytest
injects the returned value, so setup code lives in one place instead of being
copied into every test.

```python
import pytest

@pytest.fixture
def client():
    return build_client(base_url="https://example.test")

def test_status(client):
    assert client.get("/health").status_code == 200
```

## Fixture scopes

Scope decides how often a fixture is created and when it is destroyed. The
default is `function`, so every test gets a fresh instance. `class` shares one
instance across a test class, `module` shares it across one file, `package`
across one package, and `session` across the whole run. Choosing a wider scope
speeds up expensive setup such as a database container, but it also lets state
leak between tests, so widen the scope only when the object is genuinely
immutable or explicitly reset.

```python
@pytest.fixture(scope="session")
def database():
    container = start_postgres()
    yield container
    container.stop()
```

## Yield fixtures

A fixture that needs teardown becomes a generator: everything before `yield` is
setup, the yielded value is injected into the test, and everything after
`yield` runs as teardown even when the test fails. This replaces the classic
`setUp`/`tearDown` pair and keeps setup and cleanup adjacent in one function.
Teardown order is the reverse of setup order, so a fixture that depends on
another one is torn down first.

Use `addfinalizer` instead of `yield` when cleanup must be registered
conditionally or when setup can fail part way through, because a generator
fixture that raises before `yield` never reaches its teardown block.

## Requesting and parametrizing fixtures

The built-in `request` fixture exposes `request.param` for parametrized
fixtures, `request.node` for the current test item, and
`request.addfinalizer()` for dynamic cleanup. Fixtures can also depend on other
fixtures simply by taking them as parameters, which builds a dependency graph
that pytest resolves once per scope.

An `autouse=True` fixture runs for every test in its scope without being
declared as a parameter. Use it sparingly for cross-cutting concerns such as
resetting a global cache, and prefer explicit parameters when the dependency is
part of what the test is proving.
