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
python -m ruff check src tests examples scripts benchmarks docs
python -m ruff format --check src tests examples scripts benchmarks docs
python scripts/update_schema.py --check
python -m pytest
```

The `dev` extra also installs pre-commit. Run the fast, file-aware checks
across the repository before committing:

```bash
python -m pre_commit run --all-files
```

Install the hooks in a local clone with `python -m pre_commit install` if you
want the same checks to run automatically on every commit. The full test suite
remains a separate required check because it is intentionally not run by the
commit hook.

Tests cover configuration, physical equations, trajectory continuity, sampling,
row coverage, file integrity, interruption recovery, process leases, policy
statistics, the temperature baseline, CLI behavior, and publication. Publication
tests use an in-memory service that enforces the installed SDK signatures and
commit-parent semantics.

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
reports outside versioned source files.

## Continuous integration

The workflows in [.github/workflows](.github/workflows) separate the checks:

| Workflow | Checks |
| --- | --- |
| `ubuntu.yml`, `macos.yml`, `windows.yml` | Python 3.11–3.14 tests, dependency consistency, wheel/source builds, installed-wheel generation and exact replay |
| `lint.yml` | Ruff lint, import ordering, and formatting |
| `docs.yml` | Strict Sphinx HTML build, generated schema reference, notebook execution with the bundled sample and a fresh 50,003-decision release, and GitHub Pages deployment from main |
| `codeql.yml` | Python and GitHub Actions security analysis on pushes, pull requests, and a weekly schedule |

Every workflow supports manual dispatch. Test workflows retain JUnit reports for
14 days; documentation checks retain the executed notebook and validation report.
The documentation job checks the bundled sample's integrity and historical
profiles, then runs full validation in an isolated working directory with a fresh
release. Both walkthroughs execute notebook copies without overwriting tracked outputs.
The Markdown documentation builds with Sphinx, MyST, and Furo:

```bash
python -m pip install -r docs/requirements.txt
python -m sphinx -W --keep-going -b html docs docs/_build/html
```

See [docs/development.md](docs/development.md) for local preview and GitHub Pages
setup. Pull requests build the site without deploying. Pushes to `main` and manual
runs on `main` deploy after all documentation checks pass; set the repository's
Pages source to **GitHub Actions** before the first deployment.

These workflows replace `test.yml`. If branch protection requires its old check
names, update the required checks to the new platform test and lint jobs after
their first GitHub run. Configure CodeQL as advanced setup when using `codeql.yml`
instead of a separate default-setup scan.
