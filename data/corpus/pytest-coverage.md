# Pytest coverage

`pytest-cov` measures which lines, branches, and files a test run executes. It
is a plugin around `coverage.py`, so the configuration lives in the usual
`[tool.coverage]` tables and the report is only as trustworthy as the suite that
produced it.

```
pytest --cov=src/rag --cov-report=term-missing
```

`--cov` takes the package or path to measure; repeat it for several packages.
`--cov-report` chooses the output format and may also be repeated, so one run can
print a terminal summary and write machine-readable files at the same time.

## Enforcing a threshold in Continuous Integration

`--cov-fail-under=85` makes the command exit non-zero when total coverage drops
below the value, which turns coverage into a gate rather than a dashboard:

```
pytest --cov=src/rag --cov-report=term-missing --cov-fail-under=85
```

Put the flag in the CI job's pytest command, or in `addopts` inside
`[tool.pytest.ini_options]` when the same bar must hold locally. A per-run gate
is more honest than a badgy service, because it fails the exact commit that
introduced the regression. Keep the number stable and raise it deliberately;
a threshold that changes every week trains reviewers to ignore it.

`[tool.coverage.report]` supports `exclude_lines` for defensive branches such as
`if TYPE_CHECKING:` and `raise NotImplementedError`, which keeps the percentage
meaningful instead of rewarding dead code.

## Reports for review

- `--cov-report=term-missing` prints line numbers that were never executed.
- `--cov-report=html` writes `htmlcov/index.html`, which is the format reviewers
  actually browse.
- `--cov-report=xml` writes `coverage.xml` for CI services and pull-request
  annotations.
- `--cov-report=json` writes `coverage.json` for scripts that compute deltas.

Combine the human and machine formats in one invocation,
`--cov-report=html --cov-report=xml`, and publish the HTML directory as a build
artifact so reviewers can open it without re-running the suite. A common failure
mode is measuring the tests themselves: always pass an explicit `--cov=src/...`
path instead of relying on the default, and check the reported file list before
trusting the total.
