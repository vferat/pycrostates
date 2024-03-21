import ast
import subprocess
import sys
from importlib import import_module
from pathlib import Path

import isort
from mypy import stubgen

import pycrostates

directory = Path(pycrostates.__file__).parent
# remove existing stub files
for file in directory.rglob("*.pyi"):
    file.unlink()
# generate stub files, including private members and docstrings
files = [
    str(file.as_posix())
    for file in directory.rglob("*.py")
    if file.parent.name not in ("commands", "html_templates", "tests")
    and file.name not in ("conftest.py", "_tests.py", "_version.py")
]
stubgen.main(
    [
        "--no-analysis",
        "--no-import",
        "--include-private",
        "--include-docstrings",
        "--output",
        str(directory.parent),
        *files,
    ]
)
stubs = list(directory.rglob("*.pyi"))
config = str((directory.parent / "pyproject.toml"))
config_isort = isort.settings.Config(config)

# expand docstrings and inject into stub files
for stub in stubs:
    module_path = str(stub.relative_to(directory).with_suffix("").as_posix())
    module = import_module(f"{directory.name}.{module_path.replace('/', '.')}")
    module_ast = ast.parse(stub.read_text(encoding="utf-8"))
    objects = [
        node
        for node in module_ast.body
        if isinstance(node, (ast.ClassDef, ast.FunctionDef))
    ]
    for node in objects:
        docstring = getattr(module, node.name).__doc__
        if not docstring and isinstance(node, ast.FunctionDef):
            continue
        elif docstring:
            try:
                node.body[0].value.value = docstring
            except AttributeError:
                continue
        for method in node.body:
            if not isinstance(method, ast.FunctionDef):
                continue
            docstring = getattr(getattr(module, node.name), method.name).__doc__
            if docstring:
                try:
                    method.body[0].value.value = docstring
                except AttributeError:
                    continue
    unparsed = ast.unparse(module_ast)
    # remove unused imports conflicting with arguments, kwargs, method names, ...
    unparsed = unparsed.replace(", verbose as verbose", "")
    unparsed = unparsed.replace(
        "from ..viz import plot_cluster_centers as plot_cluster_centers", ""
    )
    stub.write_text(unparsed, encoding="utf-8")
    # sort imports
    isort.file(stub, config=config_isort)

# run ruff to improve stub style
exec = subprocess.run(["ruff", "format", str(directory), "--config", config])
sys.exit(exec.returncode)
