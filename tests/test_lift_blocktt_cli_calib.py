import os, subprocess, sys, unittest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _run():
    return subprocess.run(
        [sys.executable, "ref/LIFT/src/finetune_blocktt.py", "--help"],
        cwd=REPO_ROOT, capture_output=True, text=True,
    )


class TestLIFTBlockTTCalibCLI(unittest.TestCase):
    def test_help_mentions_underscore_calib_mode(self):
        r = _run()
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn("--calib_mode", r.stdout)
        self.assertIn("--calib_source", r.stdout)


if __name__ == "__main__":
    unittest.main()
