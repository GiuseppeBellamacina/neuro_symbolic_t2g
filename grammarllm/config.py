"""
Centralized configuration for grammarllm.

Provides shared paths and constants used across the package.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

#: Root directory of the grammarllm package.
PACKAGE_DIR: Path = Path(__file__).resolve().parent

#: Temporary directory for logs and debug output.
#: Uses the system temp directory to avoid write-permission issues when
#: the package is installed in a read-only location (e.g. site-packages)
#: and to avoid race conditions between concurrent processes.
TEMP_DIR: Path = Path(tempfile.gettempdir()) / "grammarllm"

#: Log file path.
LOG_FILE: Path = TEMP_DIR / "GRAM-GEN.log"

#: Parsing table JSON output path.
PARSING_TABLE_FILE: Path = TEMP_DIR / "table_parsing.json"

#: Final grammar debug output path.
FINAL_GRAMMAR_FILE: Path = TEMP_DIR / "final_grammar.txt"


def ensure_temp_dir() -> None:
    """Create the temp directory if it doesn't exist."""
    TEMP_DIR.mkdir(parents=True, exist_ok=True)


# Ensure the temp directory exists on import.
ensure_temp_dir()


def ensure_temp_dir() -> None:
    """Create the temp directory if it does not exist."""
    TEMP_DIR.mkdir(parents=True, exist_ok=True)
