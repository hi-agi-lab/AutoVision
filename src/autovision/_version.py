# Copyright (c) 2026 AutoVision contributors. Apache-2.0.
"""Single source of truth for the framework version (no import cycles).

Read by three consumers, all of which must stay consistent:

- ``pyproject.toml`` packaging (``[tool.setuptools.dynamic]``, static AST
  read — keep the assignment a plain string literal);
- the public ``autovision.__version__`` export;
- release tooling / tests (version vs ``CHANGELOG.md`` vs the ``vX.Y.Z``
  release tag, see ``docs/release_process.md``).
"""

__all__ = ["__version__"]

__version__ = "0.3.1"
