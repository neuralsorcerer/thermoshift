# Copyright (c) Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import ast
import linecache
import sys
import warnings
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier

import nbformat
import pytest

from thermoshift import notebook as runner
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


def test_function_source_is_available_in_later_cells(tmp_path):
    source = "def example(value):\n    return value + 1"
    notebook = run_notebook(tmp_path, [source, "import inspect\ninspect.getsource(example)"])
    assert ast.literal_eval(notebook.cells[-1].outputs[0].data["text/plain"]) == source + "\n"
    assert "run.ipynb:cell-1" not in linecache.cache


def test_compile_time_warning_is_captured(tmp_path, monkeypatch):
    def showwarning(message, category, *_):
        print(f"{category.__name__}: {message}", file=sys.stderr)

    with warnings.catch_warnings():
        warnings.simplefilter("always", SyntaxWarning)
        monkeypatch.setattr(warnings, "showwarning", showwarning)
        notebook = run_notebook(tmp_path, ["1 is 2"])
    warning, result = notebook.cells[0].outputs
    assert warning.name == "stderr" and "SyntaxWarning" in warning.text
    assert result.data["text/plain"] == "False"


def test_cells_share_an_importable_main_module(tmp_path):
    original_module = sys.modules["__main__"]
    notebook = run_notebook(
        tmp_path,
        [
            "from __future__ import annotations\nfrom dataclasses import KW_ONLY, dataclass",
            "@dataclass\nclass Example:\n    _: KW_ONLY\n    value: int",
            "import pickle\npickle.loads(pickle.dumps(Example(value=42)))",
        ],
    )
    assert notebook.cells[-1].outputs[0].data["text/plain"] == "Example(value=42)"
    assert sys.modules["__main__"] is original_module


def test_cells_can_import_from_the_working_directory(tmp_path, monkeypatch):
    original_path = [path for path in sys.path if path]
    monkeypatch.setattr(sys, "path", original_path)
    (tmp_path / "notebook_local_example.py").write_text("value = 42\n", encoding="utf-8")
    try:
        notebook = run_notebook(tmp_path, ["from notebook_local_example import value\nvalue"])
    finally:
        sys.modules.pop("notebook_local_example", None)
    assert notebook.cells[-1].outputs[0].data["text/plain"] == "42"
    assert sys.path is original_path


def test_parallel_calls_keep_their_working_directories_and_outputs(tmp_path):
    directories = [tmp_path / name for name in ("first", "second")]
    start = Barrier(len(directories))

    def run(directory):
        directory.mkdir()
        start.wait(timeout=5)
        return run_notebook(
            directory,
            ["import time\nfrom pathlib import Path\ntime.sleep(0.05)\nprint(Path.cwd().name)"],
        )

    with ThreadPoolExecutor(max_workers=len(directories)) as executor:
        notebooks = list(executor.map(run, directories))
    for directory, notebook in zip(directories, notebooks, strict=True):
        assert [output.text for output in notebook.cells[0].outputs] == [f"{directory.name}\n"]


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


def test_final_expression_binary_image_is_saved(tmp_path):
    notebook = run_notebook(
        tmp_path,
        [
            "class Image:\n    def _repr_png_(self):\n"
            "        return bytes.fromhex('89504e470d0a1a0a')\nImage()",
        ],
    )
    assert notebook.cells[0].outputs[0].data["image/png"] == "iVBORw0KGgo="


def test_invalid_display_metadata_saves_prior_streams_and_failure(tmp_path):
    path = tmp_path / "broken-display.ipynb"
    source = (
        "class Broken:\n    def _repr_mimebundle_(self, **_):\n"
        "        return {'text/plain': 'result'}, {'example': object()}\n"
        "print('before')\nBroken()"
    )
    nbformat.write(nbformat.v4.new_notebook(cells=[nbformat.v4.new_code_cell(source)]), path)
    with pytest.raises(TypeError, match="not JSON serializable"):
        execute(path, tmp_path)
    saved = nbformat.read(path, as_version=4)
    assert saved.metadata.execution_validation.status == "failed"
    stream, error = saved.cells[0].outputs
    assert stream.name == "stdout" and stream.text == "before\n"
    assert error.ename == "TypeError"


def test_unformattable_exception_is_recorded_and_reraised(tmp_path):
    path = tmp_path / "broken-exception.ipynb"
    source = (
        "class BrokenError(Exception):\n    def __str__(self):\n"
        "        raise RuntimeError('unformattable')\n"
        "print('before')\nraise BrokenError()"
    )
    nbformat.write(nbformat.v4.new_notebook(cells=[nbformat.v4.new_code_cell(source)]), path)
    with pytest.raises(Exception) as raised:
        execute(path, tmp_path)
    assert type(raised.value).__name__ == "BrokenError"
    saved = nbformat.read(path, as_version=4)
    assert saved.metadata.execution_validation.status == "failed"
    stream, error = saved.cells[0].outputs
    assert stream.text == "before\n"
    assert error.ename == "BrokenError"
    assert error.evalue == "<exception str() failed>"


def test_failure_to_save_an_error_preserves_the_cell_exception(tmp_path, monkeypatch):
    path = tmp_path / "write-error.ipynb"
    source = "raise ValueError('original failure')"
    nbformat.write(nbformat.v4.new_notebook(cells=[nbformat.v4.new_code_cell(source)]), path)
    atomic_bytes = runner.atomic_bytes

    def reject_failure_output(path, data):
        if b'"status": "failed"' in data:
            raise OSError("simulated write failure")
        atomic_bytes(path, data)

    monkeypatch.setattr(runner, "atomic_bytes", reject_failure_output)
    with pytest.raises(ValueError, match="original failure") as raised:
        execute(path, tmp_path)
    assert raised.value.__notes__ == ["Could not save notebook failure (simulated write failure)"]


def test_stream_order_is_saved_before_failure(tmp_path):
    path = tmp_path / "failure.ipynb"
    original_module = sys.modules["__main__"]
    original_path = sys.path
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
    assert sys.modules["__main__"] is original_module
    assert sys.path is original_path
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
