import os, subprocess, sys, unittest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _run(*args):
    return subprocess.run(
        [sys.executable, "run_sft.py", *args, "--help"],
        cwd=REPO_ROOT, capture_output=True, text=True,
    )


class TestSFTCalibCLI(unittest.TestCase):
    def test_help_mentions_calib_mode(self):
        r = _run()
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn("--calib-mode", r.stdout)
        self.assertIn("--calib-source", r.stdout)

    def test_default_parses_ok_without_calib_flags(self):
        # Use --help-based parsing as a smoke test; full parse is covered by validate tests.
        r = _run()
        self.assertEqual(r.returncode, 0)


if __name__ == "__main__":
    unittest.main()
