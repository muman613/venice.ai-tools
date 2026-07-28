"""Shared Venice.ai helpers.

This module intentionally preserves the import path used by tools that were
originally developed in the colorizetext project.
"""

import os
from pathlib import Path


def get_venice_api_key(api_key=None):
    """Return the Venice API key from an argument, environment, or local env file."""
    if api_key:
        return api_key

    api_key = os.environ.get("VENICE_API_KEY")
    if api_key:
        return api_key

    for env_file in env_file_candidates():
        loaded_key = read_env_file_value(env_file, "VENICE_API_KEY")
        if loaded_key:
            return loaded_key

    return None


def env_file_candidates():
    """Return likely local env files without requiring python-dotenv."""
    project_root = Path(__file__).resolve().parent.parent
    return (
        Path.cwd() / ".env",
        Path.cwd() / "venice.env",
        project_root / ".env",
        project_root / "venice.env",
    )


def read_env_file_value(env_file, key):
    """Read one KEY=value setting from a shell-style env file."""
    if not env_file.is_file():
        return None

    for line in env_file.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export "):].strip()
        name, separator, value = line.partition("=")
        if separator and name.strip() == key:
            return value.strip().strip("\"'")
    return None
