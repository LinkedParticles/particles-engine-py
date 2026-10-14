# SPDX-FileCopyrightText: 2026 The Particles authors
#
# SPDX-License-Identifier: Apache-2.0

"""Process entry for one ``particles benchmark trip-vs-move`` run.

The parent runs ``python -m particles.api.cli._trip_vs_move_child`` with a
private environment. The run itself lives in the Engine's
``particles.benchmark.update_gate._child``; this Surface module only binds the
operator's own verbs to it, so the benchmark package never imports the CLI.
"""

from __future__ import annotations

import sys

from typer.testing import CliRunner

from particles.api.cli import app
from particles.benchmark.update_gate import _child


def invoke(argv: list[str]) -> tuple[int, str]:
    """Run one ``particles`` command in this process; its exit code and output."""
    result = CliRunner().invoke(app, argv)
    return result.exit_code, result.output


if __name__ == "__main__":
    sys.exit(_child.main(invoke))
