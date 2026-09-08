"""Shared pytest configuration.

The suite has to run green on a machine with no API key, so anything marked
``api`` is skipped rather than failed when ANTHROPIC_API_KEY is unset. Nothing
carries that marker yet -- the convention is established here so the first
test that needs a key inherits it instead of inventing its own skip.
"""

import os

import pytest


def pytest_collection_modifyitems(config, items):
    if os.environ.get("ANTHROPIC_API_KEY"):
        return
    skip_api = pytest.mark.skip(reason="ANTHROPIC_API_KEY is not set")
    for item in items:
        if "api" in item.keywords:
            item.add_marker(skip_api)
