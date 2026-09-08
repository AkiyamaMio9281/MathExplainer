"""The sandbox's security property: generated code cannot read a credential.

The claim under test is not "child_env returns the right dictionary" but "the
child cannot read the key", which is why the central test spawns a real
subprocess and inspects what it actually saw. A dictionary assertion alone
would keep passing through a refactor that changed how the environment is
handed to the child; that one would not.

Three habits worth keeping if these grow:

* Every negative assertion is paired with a positive one. ``"X" not in env``
  is also satisfied by an empty env, so on its own it proves nothing.
* **No environment mapping ever appears inside an assert statement**, and that
  is stricter than it sounds. pytest rewrites assertions and renders every
  subexpression, so ``assert "K" not in names(env)`` prints the whole of
  ``env`` -- helper call included -- when it fails. Derive the names first,
  bind them to a local, and assert against that. Variable names are not
  secrets; values are.
* Nothing asserts with ``failure_text()`` as its message when the child's
  stdout is the environment dump, because that falls back to stdout.

None of this is hypothetical. The first draft of this file was checked by
deliberately breaking ``child_env``, and it printed the developer's real
ANTHROPIC_API_KEY into the terminal twice before the rule above was stated in
a form that actually holds.
"""

from __future__ import annotations

import json
import sys
from typing import Mapping

import pytest

from explainer import sandbox
from explainer.sandbox import child_env, run_python

SENTINEL = "sentinel-3f9a2c-must-not-leak"

# Everything the child is permitted to see: the allowlist, plus the two names
# the sandbox sets for itself.
ALLOWED = set(sandbox._ENV_ALLOWLIST) | {"PYTHONIOENCODING", "PYTHONUNBUFFERED"}

_DUMP_ENV = "import json, os; print(json.dumps(dict(os.environ)))"


def names(env: Mapping[str, str]) -> set[str]:
    """The variable names in *env*.

    Bind the result to a local and assert against that. Calling this inside an
    assert does not help: pytest renders the arguments of any call it finds in
    a failing assertion, so the mapping is printed anyway.
    """
    return set(env)


def test_child_env_inherits_nothing_outside_the_allowlist(monkeypatch):
    monkeypatch.setenv("MX_UNRELATED_VARIABLE", "set in the parent")
    env = child_env()
    present = names(env)
    encoding = env.get("PYTHONIOENCODING")

    unexpected = present - ALLOWED
    assert unexpected == set(), f"inherited: {sorted(unexpected)}"
    # The subset check above is satisfied by {}, so pin the floor as well.
    assert "PATH" in present
    assert encoding == "utf-8"


def test_api_key_is_absent_by_name_and_by_value(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", SENTINEL)
    env = child_env()
    present = names(env)
    # Checking the value too: a copy of the key living under some innocuous
    # name satisfies the name check and is still a leak.
    leaked = sorted(name for name, value in env.items() if value == SENTINEL)

    assert "ANTHROPIC_API_KEY" not in present
    assert leaked == [], f"sentinel reachable under: {leaked}"
    assert "PATH" in present


@pytest.mark.slow
def test_a_real_child_process_cannot_read_the_key(monkeypatch, tmp_path):
    monkeypatch.setenv("ANTHROPIC_API_KEY", SENTINEL)

    done = run_python(["-c", _DUMP_ENV], cwd=tmp_path, timeout=60)
    # Not failure_text() here: this child's stdout is the environment dump,
    # and failure_text falls back to stdout when stderr is empty.
    assert done.ok, f"child exited {done.returncode}, timed_out={done.timed_out}"

    seen = json.loads(done.stdout)
    present = names(seen)
    leaked = sorted(name for name, value in seen.items() if value == SENTINEL)
    unexpected = present - ALLOWED

    assert leaked == [], f"sentinel visible to the child under: {leaked}"
    assert "ANTHROPIC_API_KEY" not in present
    assert unexpected == set(), f"child inherited: {sorted(unexpected)}"
    assert "PATH" in present


def test_a_credential_shaped_allowlist_entry_is_still_dropped(monkeypatch):
    # Setting a credential-shaped variable and asserting it is absent proves
    # nothing by itself: the allowlist already drops every name it does not
    # mention, so that assertion passes with _SECRET_MARKERS emptied out.
    # Putting the name *into* the allowlist is what puts the second filter
    # under test.
    monkeypatch.setattr(
        sandbox, "_ENV_ALLOWLIST", sandbox._ENV_ALLOWLIST + ("MX_TEST_API_KEY",)
    )
    monkeypatch.setenv("MX_TEST_API_KEY", SENTINEL)

    present = names(child_env())
    assert "MX_TEST_API_KEY" not in present
    assert "PATH" in present


@pytest.mark.parametrize(
    "name",
    [
        "MX_KEY",
        "MX_TOKEN",
        "MX_SECRET",
        "MX_PASSWORD",
        "MX_CREDENTIAL",
        "MX_AUTH",
        "mx_lowercase_token",  # the check uppercases before matching
    ],
)
def test_extra_refuses_a_credential_shaped_name(name):
    with pytest.raises(ValueError):
        child_env(extra={name: "value"})


def test_extra_passes_an_ordinary_name():
    # Positive control for the test above: it must not be passing because
    # every extra name is refused.
    value = child_env(extra={"MX_SCENE_ID": "intro"}).get("MX_SCENE_ID")
    assert value == "intro"


@pytest.mark.skipif(sys.platform != "win32", reason="APPDATA is Windows-only")
def test_appdata_survives_because_imports_depend_on_it():
    # HANDOFF.md records this trap: Windows resolves per-user site-packages
    # from APPDATA, so tidying it out of the allowlist makes the sandbox fail
    # with a ModuleNotFoundError that reads like a broken install.
    present = names(child_env())
    assert "APPDATA" in present


@pytest.mark.slow
def test_the_child_can_import_manim(tmp_path):
    # The end-to-end form of the test above, and the one that would actually
    # catch the regression: the allowlist is only correct if the child can
    # still import what it needs to render.
    done = run_python(["-c", "import manim"], cwd=tmp_path, timeout=120)
    assert done.ok, done.failure_text()


@pytest.mark.slow
def test_a_hanging_child_is_killed_and_reported(tmp_path):
    done = run_python(["-c", "import time; time.sleep(30)"], cwd=tmp_path, timeout=2)

    assert done.timed_out
    assert done.returncode is None
    assert not done.ok
    # Killed near the deadline rather than after the sleep finished.
    assert done.seconds < 15
    assert "Timed out after" in done.failure_text()


def test_a_missing_working_directory_raises_and_is_not_created(tmp_path):
    missing = tmp_path / "never-created"

    with pytest.raises(FileNotFoundError):
        run_python(["-c", "print(1)"], cwd=missing, timeout=10)

    assert not missing.exists()


def test_failure_text_keeps_the_tail_where_the_traceback_is():
    done = sandbox.Completed(
        returncode=1,
        stdout="",
        stderr="filler\n" * 2000 + "NameError: name 'NoSuchThing' is not defined",
        seconds=0.1,
        timed_out=False,
    )

    text = done.failure_text(limit=200)
    assert text.startswith("...")
    assert "NameError" in text
    assert len(text) <= 204  # the limit, plus the "...\n" marker


def test_failure_text_falls_back_to_stdout_when_stderr_is_empty():
    done = sandbox.Completed(
        returncode=1, stdout="printed to stdout", stderr="", seconds=0.1, timed_out=False
    )
    assert done.failure_text() == "printed to stdout"
