"""Run a command while preventing Windows from sleeping.

Usage:  python tools/run_awake.py <command> [args...]
Calls SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED) so the
system stays awake for the lifetime of the wrapped command (display may
still turn off). State clears automatically when this process exits.
"""
import ctypes
import subprocess
import sys

ES_CONTINUOUS = 0x80000000
ES_SYSTEM_REQUIRED = 0x00000001

if __name__ == "__main__":
    ctypes.windll.kernel32.SetThreadExecutionState(
        ES_CONTINUOUS | ES_SYSTEM_REQUIRED)
    sys.exit(subprocess.call(sys.argv[1:]))
