"""Run a long command detached from any console window, appending its output
and, when it ends, its exit code to a log.

The launcher starts a waiter process (this module with --wait) and returns at
once. The waiter has no console window (Windows: CREATE_NO_WINDOW and a new
process group; POSIX: a new session), so closing a terminal cannot deliver a
window-close or hang-up event to it or to the command it runs. The waiter runs
the command with stdout and stderr appended to the log, waits, and appends
``exit code N``. If the waiter itself is killed, the log has no exit line,
which is the signal that the run did not end on its own.

Usage (V1-5 long runs)::

    PYTHONPATH=src uv run python -m detached --log <log> -- \\
        python -u -m inspector.runtime screen-ablate --parquet <HF test parquet>
    PYTHONPATH=src uv run python -m detached --log <log> -- \\
        python -u -m build_dictionary --examples --example-features <json> --max-rss-gb 4
"""
from __future__ import annotations

import argparse
import datetime
import os
import subprocess
import sys
import time
from pathlib import Path

CREATE_NO_WINDOW = 0x08000000
CREATE_NEW_PROCESS_GROUP = 0x00000200
SRC = Path(__file__).resolve().parent


def _stamp() -> str:
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _append(log: Path, text: str) -> None:
    with open(log, "a", encoding="utf-8") as fh:
        fh.write(text)


def run_logged(cmd: list[str], log: Path, cwd: Path | None = None) -> int:
    """Run cmd with stdout and stderr appended to log; append a start line
    before and an exit-code line after. Returns the exit code (127 if the
    command could not be started)."""
    log = Path(log)
    log.parent.mkdir(parents=True, exist_ok=True)
    _append(log, f"[{_stamp()}] detached start (waiter pid {os.getpid()}): {subprocess.list2cmdline(cmd)}\n")
    t0 = time.time()
    try:
        with open(log, "ab") as fh:
            rc = subprocess.call(cmd, stdout=fh, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, cwd=cwd)
    except OSError as exc:
        _append(log, f"[{_stamp()}] exit code 127: could not start the command ({exc})\n")
        return 127
    hexed = f" (0x{rc & 0xFFFFFFFF:08X})" if rc < 0 or rc > 255 else ""
    _append(log, f"[{_stamp()}] exit code {rc}{hexed} after {(time.time() - t0) / 3600:.2f} h\n")
    return rc


def launch(cmd: list[str], log: Path, cwd: Path | None = None) -> int:
    """Start the waiter for cmd without a console window and return its pid."""
    if not cmd:
        raise ValueError("no command to run")
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join([str(SRC)] + ([env["PYTHONPATH"]] if env.get("PYTHONPATH") else []))
    kwargs: dict = {"stdin": subprocess.DEVNULL, "stdout": subprocess.DEVNULL, "stderr": subprocess.DEVNULL,
                    "cwd": cwd, "env": env, "close_fds": True}
    if os.name == "nt":
        kwargs["creationflags"] = CREATE_NO_WINDOW | CREATE_NEW_PROCESS_GROUP
    else:
        kwargs["start_new_session"] = True
    waiter = [sys.executable, "-m", "detached", "--wait", "--log", str(Path(log).resolve())]
    if cwd is not None:
        waiter += ["--cwd", str(cwd)]
    return subprocess.Popen(waiter + ["--", *cmd], **kwargs).pid


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--log", type=Path, required=True, help="log file; output and the exit code are appended")
    p.add_argument("--cwd", type=Path, default=None, help="working directory of the command")
    p.add_argument("--wait", action="store_true", help=argparse.SUPPRESS)   # the waiter itself
    p.add_argument("cmd", nargs=argparse.REMAINDER, help="-- then the command")
    args = p.parse_args(argv)
    cmd = args.cmd[1:] if args.cmd[:1] == ["--"] else args.cmd
    if not cmd:
        p.error("give the command after --")
    if args.wait:
        return run_logged(cmd, args.log, args.cwd)
    pid = launch(cmd, args.log, args.cwd)
    print(f"detached: waiter pid {pid}; output and exit code -> {args.log}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
