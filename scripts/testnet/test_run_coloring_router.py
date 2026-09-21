"""Check that a failed measurement cannot leave device users behind."""

import os
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path

from run_coloring_router import run_window


class RunnerTests(unittest.TestCase):
    def test_timeout_kills_descendants_that_ignore_termination(self):
        script = (
            "import os,signal,subprocess,sys,time; "
            "child=subprocess.Popen([sys.executable,'-c',"
            "'import signal,time; signal.signal(signal.SIGTERM,signal.SIG_IGN); time.sleep(300)']); "
            "print(os.getpid(),child.pid,flush=True); time.sleep(300)"
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "processes.log"
            with path.open("w") as log, self.assertRaises(subprocess.TimeoutExpired):
                run_window(
                    [sys.executable, "-c", script],
                    Path(directory),
                    dict(os.environ),
                    log,
                    timeout=0.5,
                )
            parent, child = map(int, path.read_text().split())
            deadline = time.monotonic() + 2
            while time.monotonic() < deadline:
                try:
                    os.kill(child, 0)
                except ProcessLookupError:
                    break
                time.sleep(0.01)
            for pid in [parent, child]:
                with self.assertRaises(ProcessLookupError):
                    os.kill(pid, 0)


if __name__ == "__main__":
    unittest.main()
