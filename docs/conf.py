# Copyright (c) Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Build the Markdown guides as a Sphinx site without importing runtime dependencies."""

import os
import runpy
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
project = "ThermoShift"
author = "Soumyadip Sarkar"
copyright = f"{datetime.now(UTC):%Y}, {author}"
release = runpy.run_path(str(ROOT / "src/thermoshift/_version.py"))["__version__"]
version = release

extensions = [
    "myst_parser",
    "sphinx.ext.githubpages",
    "sphinx.ext.mathjax",
    "sphinxcontrib.mermaid",
]
source_suffix = {".md": "markdown", ".rst": "restructuredtext"}
root_doc = "index"
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]
nitpicky = True

myst_enable_extensions = ["colon_fence", "dollarmath", "amsmath"]
myst_heading_anchors = 3
myst_fence_as_directive = ["mermaid"]

html_theme = "furo"
html_title = f"ThermoShift {release} documentation"
html_baseurl = os.environ.get("SPHINX_HTML_BASEURL", "")
html_theme_options = {
    "source_repository": "https://github.com/neuralsorcerer/thermoshift/",
    "source_branch": "main",
    "source_directory": "docs/",
}
html_show_sphinx = False
