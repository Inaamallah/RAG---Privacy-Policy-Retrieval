"""Small console helpers shared by the command line entry points."""

import sys


def use_utf8_stdout():
    """
    Switches stdout/stderr to UTF-8.

    Model answers and PDF text routinely contain characters the default
    Windows console codepage cannot encode, which would otherwise make a plain
    `print()` raise UnicodeEncodeError. Unencodable characters are replaced
    rather than raising.
    """
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
