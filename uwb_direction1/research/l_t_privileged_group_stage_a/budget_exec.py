#!/usr/bin/env python3
"""Set a kernel CPU ceiling, then exec the worker in the same PID."""
from __future__ import annotations

import os
import resource
import sys


def main() -> None:
    if len(sys.argv) < 3:
        raise SystemExit("usage: budget_exec.py CPU_SECONDS COMMAND [ARG ...]")
    seconds = int(sys.argv[1])
    if seconds <= 0:
        raise SystemExit("CPU_SECONDS must be positive")
    resource.setrlimit(resource.RLIMIT_CPU, (seconds, seconds))
    command = sys.argv[2:]
    os.execvpe(command[0], command, os.environ)


if __name__ == "__main__":
    main()
