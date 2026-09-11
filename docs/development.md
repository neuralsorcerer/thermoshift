# Development and documentation

The [contribution guide](https://github.com/neuralsorcerer/thermoshift/blob/main/CONTRIBUTING.md)
covers package development, scientific regression tests, and distribution checks.
The documentation site builds the Markdown guides directly using Sphinx, MyST,
and the Furo theme. Mermaid diagrams and the model's mathematical equations render
in the HTML output.

## Build locally

From the repository root in an activated Python 3.11+ environment:

```bash
python -m pip install -r docs/requirements.txt
python -m sphinx -W --keep-going -b html docs docs/_build/html
python -m http.server 8000 --bind 127.0.0.1 --directory docs/_build/html
```

Open `http://127.0.0.1:8000` to preview the site. Stop the server with Ctrl+C.
The Sphinx command treats warnings as failures. HTML output and build caches stay
in the ignored `docs/_build/` directory. Search and page navigation are generated
by Sphinx; the site version is read directly from `src/thermoshift/_version.py`.

## Update content

Edit the guides in `docs/` and add new pages to the toctree in `docs/index.md`.
Use relative Markdown links between documentation pages; use repository URLs for
files outside the site. The model page includes the packaged datasheet directly
so its equations remain shared with generated releases.

The column reference is generated from the package schema:

```bash
python -m pip install -e .
python scripts/update_schema.py
python scripts/update_schema.py --check
```

The [quickstart notebook](https://github.com/neuralsorcerer/thermoshift/blob/main/notebooks/quickstart.ipynb)
is checked separately by the documentation workflow. It generates a fresh release
with the active runtime and executes a copy of the notebook against that release.
The executed notebook and validation report are retained as workflow artifacts.

## GitHub Pages

The documentation workflow builds on relevant pushes, pull requests, and manual
runs. It deploys only after its build, schema, and notebook checks pass, on a push
to `main` or a manual run selecting `main`. Pull requests build a downloadable HTML
artifact without deploying.

In repository **Settings → Pages → Build and deployment**, set **Source** to
**GitHub Actions**. The deployment uses the `github-pages` environment and the
workflow's built-in token with Pages and OIDC permissions; no personal token is
required. Its deployment output reports the actual site URL. Repository or
environment policies can require approval before the deployment job runs.
