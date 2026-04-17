import os, subprocess, sys, unittest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _run():
    return subprocess.run(
        [sys.executable, "run_rl_dapo.py", "--help"],
        cwd=REPO_ROOT, capture_output=True, text=True,
    )


class TestRLDAPOCalibCLI(unittest.TestCase):
    def test_help_mentions_calib_mode(self):
        r = _run()
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn("--calib-mode", r.stdout)


if __name__ == "__main__":
    unittest.main()
