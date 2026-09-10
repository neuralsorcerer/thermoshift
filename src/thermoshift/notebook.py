# Copyright (c) Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Execute trusted plain-Python notebooks and save cell results atomically."""

import ast
import contextlib
import io
import os
import traceback
from pathlib import Path

import nbformat
from IPython.core.formatters import DisplayFormatter

from thermoshift.filesystem import atomic_bytes


def execute(path: str | Path, working_directory: str | Path) -> int:
    path = Path(path).resolve()
    notebook = nbformat.read(path, as_version=4)
    namespace = {"__name__": "__main__"}
    formatter = DisplayFormatter()
    count = 0
    original = Path.cwd()
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
        for cell in notebook.cells:
            if cell.cell_type != "code":
                continue
            current_code = cell
            count += 1
            cell.execution_count = count
            stdout, stderr = io.StringIO(), io.StringIO()
            try:
                tree = ast.parse(cell.source)
                final_expression = (
                    tree.body.pop() if tree.body and isinstance(tree.body[-1], ast.Expr) else None
                )
                with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
                    exec(compile(tree, f"{path.name}:cell-{count}", "exec"), namespace)  # noqa: S102 - intentionally executes trusted notebook cells
                    value = (
                        eval(
                            compile(
                                ast.Expression(final_expression.value),
                                f"{path.name}:cell-{count}",
                                "eval",
                            ),
                            namespace,
                        )
                        if final_expression
                        else None
                    )
            finally:
                for name, stream in (("stdout", stdout), ("stderr", stderr)):
                    if stream.getvalue():
                        cell.outputs.append(
                            nbformat.v4.new_output("stream", name=name, text=stream.getvalue())
                        )
            if value is not None:
                data, metadata = formatter.format(value)
                cell.outputs.append(
                    nbformat.v4.new_output(
                        "execute_result", execution_count=count, data=data, metadata=metadata
                    )
                )
            save()
        notebook.metadata["execution_validation"]["status"] = (
            "all_plain_python_cells_executed_top_to_bottom"
        )
        save()
    except BaseException as error:
        notebook.metadata["execution_validation"]["status"] = (
            "failed" if isinstance(error, Exception) else "interrupted"
        )
        if current_code is not None:
            current_code.outputs.append(
                nbformat.v4.new_output(
                    "error",
                    ename=type(error).__name__,
                    evalue=str(error),
                    traceback=traceback.format_exception(type(error), error, error.__traceback__),
                )
            )
        save()
        raise
    finally:
        os.chdir(original)
    return count
