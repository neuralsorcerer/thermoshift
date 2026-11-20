# Copyright (c) Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Execute trusted plain-Python notebooks and save cell results atomically."""

import __future__

import ast
import contextlib
import io
import linecache
import os
import sys
import traceback
from base64 import b64encode
from pathlib import Path
from threading import RLock
from types import ModuleType

import nbformat
from IPython.core.formatters import DisplayFormatter

from thermoshift.filesystem import atomic_bytes

_EXECUTION_LOCK = RLock()
_BINARY_DISPLAY_TYPES = {"image/png", "image/jpeg", "image/gif", "image/webp", "application/pdf"}


class _CellStream(io.StringIO):
    """Append stream writes in their original order, including writes before errors."""

    def __init__(self, outputs, name):
        super().__init__()
        self.outputs = outputs
        self.name = name

    def write(self, text):
        count = super().write(text)
        if text:
            if (
                self.outputs
                and self.outputs[-1].output_type == "stream"
                and self.outputs[-1].name == self.name
            ):
                self.outputs[-1].text += text
            else:
                self.outputs.append(nbformat.v4.new_output("stream", name=self.name, text=text))
        return count


def execute(path: str | Path, working_directory: str | Path) -> int:
    """Execute and save a notebook while holding its process-wide execution context."""
    with _EXECUTION_LOCK:
        return _execute(path, working_directory)


def _execute(path: str | Path, working_directory: str | Path) -> int:
    path = Path(path).resolve()
    notebook = nbformat.read(path, as_version=4)
    main_module = ModuleType("__main__")
    namespace = vars(main_module)
    formatter = DisplayFormatter()
    future_flags = 0
    future_mask = sum(
        getattr(__future__, name).compiler_flag for name in __future__.all_feature_names
    )
    count = 0
    original = Path.cwd()
    original_module = sys.modules.get("__main__")
    had_main_module = "__main__" in sys.modules
    original_path = sys.path
    previous_sources = {}
    current_code = None
    execution = {
        "status": "running",
        "engine": "in-process Python AST evaluation with IPython display formatting",
        "scope": "trusted plain-Python cells; stdout, stderr and final-expression output",
    }
    notebook.metadata["execution_validation"] = execution
    for cell in notebook.cells:
        if cell.cell_type == "code":
            cell.execution_count = None
            cell.outputs = []

    def save():
        nbformat.validate(notebook)
        atomic_bytes(path, nbformat.writes(notebook).encode("utf-8"))

    save()
    try:
        os.chdir(working_directory)
        sys.path = [str(Path.cwd()), *original_path]
        sys.modules["__main__"] = main_module
        for cell in notebook.cells:
            if cell.cell_type != "code":
                continue
            current_code = cell
            count += 1
            cell.execution_count = count
            stdout = _CellStream(cell.outputs, "stdout")
            stderr = _CellStream(cell.outputs, "stderr")
            filename = f"{path.name}:cell-{count}"
            previous_sources[filename] = linecache.cache.get(filename)
            linecache.cache[filename] = (
                len(cell.source),
                None,
                [line + "\n" for line in cell.source.splitlines()],
                filename,
            )
            result = None
            with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
                tree = compile(
                    cell.source,
                    filename,
                    "exec",
                    flags=future_flags | ast.PyCF_ONLY_AST,
                    dont_inherit=True,
                )
                final_expression = (
                    tree.body.pop() if tree.body and isinstance(tree.body[-1], ast.Expr) else None
                )
                code = compile(tree, filename, "exec", flags=future_flags, dont_inherit=True)
                future_flags |= code.co_flags & future_mask
                expression_code = (
                    compile(
                        ast.Expression(final_expression.value),
                        filename,
                        "eval",
                        flags=future_flags,
                        dont_inherit=True,
                    )
                    if final_expression
                    else None
                )
                exec(code, namespace)  # noqa: S102 - intentionally executes trusted notebook cells
                value = eval(expression_code, namespace) if expression_code else None
                if value is not None:
                    data, metadata = formatter.format(value)
                    data = {
                        mime: b64encode(representation).decode("ascii")
                        if mime in _BINARY_DISPLAY_TYPES and isinstance(representation, bytes)
                        else representation
                        for mime, representation in data.items()
                    }
                    result = nbformat.v4.new_output(
                        "execute_result", execution_count=count, data=data, metadata=metadata
                    )
                    cell.outputs.append(result)
            try:
                save()
            except BaseException:
                # Keep the streams and error if a display result cannot be serialized.
                if result is not None:
                    cell.outputs.pop()
                raise
        notebook.metadata["execution_validation"]["status"] = (
            "all_plain_python_cells_executed_top_to_bottom"
        )
        save()
    except BaseException as error:
        notebook.metadata["execution_validation"]["status"] = (
            "failed" if isinstance(error, Exception) else "interrupted"
        )
        if current_code is not None:
            try:
                message = str(error)
            except BaseException:
                message = "<exception str() failed>"
            current_code.outputs.append(
                nbformat.v4.new_output(
                    "error",
                    ename=type(error).__name__,
                    evalue=message,
                    traceback=traceback.format_exception(type(error), error, error.__traceback__),
                )
            )
        try:
            save()
        except Exception as save_error:
            BaseException.add_note(error, f"Could not save notebook failure ({save_error})")
        raise
    finally:
        for filename, source in previous_sources.items():
            if source is None:
                linecache.cache.pop(filename, None)
            else:
                linecache.cache[filename] = source
        if had_main_module:
            sys.modules["__main__"] = original_module
        else:
            sys.modules.pop("__main__", None)
        sys.path = original_path
        os.chdir(original)
    return count
