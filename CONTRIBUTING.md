# Development

## Set up

Create and activate a Python virtual environment, then install the development
extras from the repository root:

```bash
python -m pip install -e ".[dev,hub,analysis,notebook]"
```

To use the recorded dependency versions, add `-c constraints.txt` to the install
command. The package version is defined once in `src/thermoshift/_version.py`.

## Check a change

```bash
python -m ruff check src tests examples scripts benchmarks
python -m ruff format --check src tests examples scripts benchmarks
python scripts/update_schema.py --check
python -m pytest
```

Tests cover configuration, physical equations, trajectory continuity, sampling,
row coverage, file integrity, interruption recovery, process leases, policy
statistics, CLI behavior, and publication. Publication tests use an in-memory
service that enforces the installed SDK signatures and commit-parent semantics.

When changing numeric logic, compare outcomes with an independent equation or
reference calculation. Preserve the association between observed features, action
probabilities, factual outcomes, and oracle outcomes. Add a targeted regression
for a changed behavior or reproduced failure.

## Update data documentation

`src/thermoshift/schema.py` owns the column definitions. Refresh the Markdown
column reference after editing it:

```bash
python scripts/update_schema.py
```

Dataset-facing text lives in `src/thermoshift/resources/`. Those resources are
included in the wheel and copied into generated releases. The source fingerprint
covers nested package modules and those templates.

The walkthrough uses the bundled sample:

```bash
python scripts/run_notebook.py
```

It executes trusted plain-Python cells and saves execution counts, displayed
expressions, output streams, and errors in the notebook.

## Build distributions

```bash
python -m build
python scripts/check_distribution.py
```

The distribution check verifies packaged code and templates, installs the wheel
into a temporary target, then generates, validates, and replays a small dataset
from that installation. Build outputs go to `dist/`.

Keep runtime environments, caches, generated experiment directories, and test
reports outside versioned source files. The CI workflow runs linting, formatting,
schema checks, tests, and distribution checks on its configured Python versions.
