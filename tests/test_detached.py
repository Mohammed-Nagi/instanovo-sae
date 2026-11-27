"""detached: the waiter appends the command's output and its exit code to the log."""
from __future__ import annotations

import sys
import time

import pytest

import detached


def test_run_logged_appends_output_and_exit_code(tmp_path):
    log = tmp_path / "logs" / "run.log"
    log.parent.mkdir()
    log.write_text("earlier line\n", encoding="utf-8")
    rc = detached.run_logged([sys.executable, "-c", "import sys; print('hello'); sys.exit(3)"], log)
    text = log.read_text(encoding="utf-8")
    assert rc == 3
    assert text.startswith("earlier line\n")                      # appended, not truncated
    assert "detached start" in text and "hello" in text
    assert text.rstrip().splitlines()[-1].split("] ", 1)[1].startswith("exit code 3 after")


def test_run_logged_records_a_command_that_cannot_start(tmp_path):
    log = tmp_path / "run.log"
    assert detached.run_logged([str(tmp_path / "no_such_program")], log) == 127
    assert "exit code 127: could not start" in log.read_text(encoding="utf-8")


def test_launch_returns_at_once_and_the_waiter_logs_the_exit_code(tmp_path):
    log = tmp_path / "run.log"
    t0 = time.time()
    pid = detached.launch([sys.executable, "-c", "import time; time.sleep(1); print('done')"], log)
    assert pid > 0 and time.time() - t0 < 1.0                       # did not wait for the command
    deadline = time.time() + 60
    while time.time() < deadline:
        if log.is_file() and "exit code" in log.read_text(encoding="utf-8"):
            break
        time.sleep(0.2)
    text = log.read_text(encoding="utf-8")
    assert "done" in text and "exit code 0 after" in text


def test_main_needs_a_command(tmp_path):
    with pytest.raises(SystemExit):
        detached.main(["--log", str(tmp_path / "x.log")])
