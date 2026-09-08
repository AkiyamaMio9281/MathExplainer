"""Run untrusted, model-generated Python in a separate process.

Everything this package renders is code an LLM wrote, so it is executed as a
child process with a hard timeout and a scrubbed environment rather than
imported into the host interpreter. Two consequences worth stating plainly:

* The parent process holds an ANTHROPIC_API_KEY. Generated code must never see
  it, so the child's environment is built from an allowlist instead of being
  inherited -- ``os.environ.copy()`` with a couple of ``pop`` calls is the easy
  version of this and it leaks every credential nobody thought to name.
* This is process isolation, not a security boundary. There is no syscall
  filter and no network block, so generated code can still reach the network
  and the parts of the filesystem the user can reach. For genuinely untrusted
  input, run the whole pipeline inside a container. ``SANDBOX_CAVEAT`` below
  says so where users will see it.
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

SANDBOX_CAVEAT = (
    "Generated code runs as a child process with a timeout and a scrubbed "
    "environment. That is isolation, not a security boundary: there is no "
    "syscall filter and no network block. Run inside a container if the input "
    "is untrusted."
)

# Variables the child genuinely needs. Everything else is dropped, so a new
# secret in the parent environment does not silently become readable.
_ENV_ALLOWLIST = (
    "PATH",
    "SYSTEMROOT",      # Windows: python won't start without it
    "WINDIR",
    "TEMP",
    "TMP",
    "TMPDIR",
    "HOME",
    "USERPROFILE",
    # Windows resolves the per-user site-packages directory from APPDATA. Drop
    # it and the child cannot import anything pip installed with --user --
    # manim included -- which surfaces as a baffling ModuleNotFoundError.
    "APPDATA",
    "LOCALAPPDATA",
    "LANG",
    "LC_ALL",
    "PYTHONPATH",
    "COMSPEC",
    "NUMBER_OF_PROCESSORS",
    "PROCESSOR_ARCHITECTURE",
)

# Belt and braces: even if one of the above ever starts carrying a secret, a
# name that looks like a credential never reaches the child.
_SECRET_MARKERS = ("KEY", "TOKEN", "SECRET", "PASSWORD", "CREDENTIAL", "AUTH")


def child_env(extra: Mapping[str, str] | None = None) -> dict[str, str]:
    """Build the child's environment from an allowlist, never by inheritance."""
    env = {
        name: os.environ[name]
        for name in _ENV_ALLOWLIST
        if name in os.environ and not _looks_secret(name)
    }
    # Keep generated code's own output decodable regardless of console codepage.
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUNBUFFERED"] = "1"
    if extra:
        for name, value in extra.items():
            if _looks_secret(name):
                raise ValueError(f"refusing to pass {name!r} into the sandbox")
            env[name] = value
    return env


def _looks_secret(name: str) -> bool:
    upper = name.upper()
    return any(marker in upper for marker in _SECRET_MARKERS)


@dataclass(frozen=True)
class Completed:
    """Outcome of one sandboxed run."""

    returncode: int | None      # None when the run was killed on timeout
    stdout: str
    stderr: str
    seconds: float
    timed_out: bool

    @property
    def ok(self) -> bool:
        return self.returncode == 0 and not self.timed_out

    def failure_text(self, limit: int = 4000) -> str:
        """Stderr if there is any, else stdout -- trimmed, for feeding back to
        the model. The tail carries the traceback, so keep the end."""
        text = (self.stderr or self.stdout or "").strip()
        if self.timed_out:
            text = f"Timed out after {self.seconds:.1f}s.\n{text}"
        if len(text) > limit:
            text = "...\n" + text[-limit:]
        return text


def run_python(
    args: Sequence[str],
    *,
    cwd: Path,
    timeout: float,
    extra_env: Mapping[str, str] | None = None,
) -> Completed:
    """Run ``sys.executable`` with *args* under a timeout, in *cwd*."""
    cwd.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    try:
        proc = subprocess.run(
            [sys.executable, *args],
            cwd=str(cwd),
            env=child_env(extra_env),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
        )
    except subprocess.TimeoutExpired as exc:
        return Completed(
            returncode=None,
            stdout=_as_text(exc.stdout),
            stderr=_as_text(exc.stderr),
            seconds=time.perf_counter() - started,
            timed_out=True,
        )
    return Completed(
        returncode=proc.returncode,
        stdout=proc.stdout or "",
        stderr=proc.stderr or "",
        seconds=time.perf_counter() - started,
        timed_out=False,
    )


def _as_text(raw: object) -> str:
    if raw is None:
        return ""
    if isinstance(raw, bytes):
        return raw.decode("utf-8", "replace")
    return str(raw)
