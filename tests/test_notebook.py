# Copyright (c) Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import nbformat
import pytest

from thermoshift.notebook import execute


def run_notebook(tmp_path, sources):
    path = tmp_path / "run.ipynb"
    nbformat.write(
        nbformat.v4.new_notebook(cells=[nbformat.v4.new_code_cell(source) for source in sources]),
        path,
    )
    assert execute(path, tmp_path) == len(sources)
    return nbformat.read(path, as_version=4)


def test_only_code_cells_are_counted_and_executed(tmp_path):
    path = tmp_path / "mixed.ipynb"
    nbformat.write(
        nbformat.v4.new_notebook(
            cells=[
                nbformat.v4.new_markdown_cell("# Heading"),
                nbformat.v4.new_code_cell("marker = 'ran'"),
                nbformat.v4.new_raw_cell("raw payload"),
                nbformat.v4.new_markdown_cell("Trailing prose"),
                nbformat.v4.new_code_cell("marker"),
            ]
        ),
        path,
    )
    assert execute(path, tmp_path) == 2
    saved = nbformat.read(path, as_version=4)
    assert [cell.get("execution_count") for cell in saved.cells] == [None, 1, None, None, 2]
    assert saved.cells[-1].outputs[0].data["text/plain"] == "'ran'"
    assert not any("outputs" in cell for cell in saved.cells if cell.cell_type == "markdown")


def test_future_annotations_persist_across_cells(tmp_path):
    notebook = run_notebook(
        tmp_path,
        [
            "from __future__ import annotations",
            "def identity(value: UndefinedType) -> UndefinedType:\n    return value",
            "identity.__annotations__",
        ],
    )
    assert notebook.cells[-1].outputs[0].data["text/plain"] == (
        "{'value': 'UndefinedType', 'return': 'UndefinedType'}"
    )


def test_future_flags_apply_to_parsing_and_final_expressions(tmp_path):
    notebook = run_notebook(tmp_path, ["from __future__ import barry_as_FLUFL", "1 <> 2"])
    assert notebook.cells[-1].outputs[0].data["text/plain"] == "True"


def test_stdout_and_stderr_keep_emission_order(tmp_path):
    notebook = run_notebook(
        tmp_path,
        [
            "import sys\nprint('first')\nprint('second', file=sys.stderr)\nprint('third')\n42",
        ],
    )
    outputs = notebook.cells[0].outputs
    assert [(output.name, output.text) for output in outputs[:-1]] == [
        ("stdout", "first\n"),
        ("stderr", "second\n"),
        ("stdout", "third\n"),
    ]
    assert outputs[-1].data["text/plain"] == "42"


def test_final_expression_formatter_output_is_captured(tmp_path):
    notebook = run_notebook(
        tmp_path,
        [
            "class Result:\n    def __repr__(self):\n        print('formatting')\n        return 'Result()'\nResult()",
        ],
    )
    outputs = notebook.cells[0].outputs
    assert outputs[0].name == "stdout" and outputs[0].text == "formatting\n"
    assert outputs[1].data["text/plain"] == "Result()"


def test_stream_order_is_saved_before_failure(tmp_path):
    path = tmp_path / "failure.ipynb"
    nbformat.write(
        nbformat.v4.new_notebook(
            cells=[
                nbformat.v4.new_code_cell(
                    "import sys\nprint('first', file=sys.stderr)\nprint('second')\nraise ValueError('failed')"
                )
            ]
        ),
        path,
    )
    with pytest.raises(ValueError, match="failed"):
        execute(path, tmp_path)
    notebook = nbformat.read(path, as_version=4)
    assert [(output.name, output.text) for output in notebook.cells[0].outputs[:-1]] == [
        ("stderr", "first\n"),
        ("stdout", "second\n"),
    ]
    assert notebook.cells[0].outputs[-1].output_type == "error"


@pytest.mark.parametrize("expression", ["(yield 1)", "await missing()"])
def test_entire_cell_is_compiled_before_any_statement_runs(tmp_path, expression):
    path = tmp_path / "syntax-error.ipynb"
    nbformat.write(
        nbformat.v4.new_notebook(
            cells=[
                nbformat.v4.new_code_cell(
                    f"from pathlib import Path\nPath('side-effect').touch()\n{expression}"
                )
            ]
        ),
        path,
    )
    with pytest.raises(SyntaxError):
        execute(path, tmp_path)
    assert not (tmp_path / "side-effect").exists()
    notebook = nbformat.read(path, as_version=4)
    assert len(notebook.cells[0].outputs) == 1
    assert notebook.cells[0].outputs[0].ename == "SyntaxError"
