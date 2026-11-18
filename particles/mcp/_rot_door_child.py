# SPDX-FileCopyrightText: 2026 The Particles authors
#
# SPDX-License-Identifier: Apache-2.0

"""Process entry for the Particles memory in the rot door arm.

The parent runs ``python -m particles.mcp._rot_door_child`` with a
private environment. The child's work lives in the Engine's
``particles.benchmark.rot.door_child``; this Surface module only binds the MCP
``query`` tool to it, so the door Particles is scored on is the one an agent
receives and the benchmark package never imports the MCP server.
"""

from __future__ import annotations

import sys

from particles.benchmark.rot import door_child
from particles.mcp.tools.query import query

if __name__ == "__main__":
    sys.exit(door_child.main(query))
