import threading
import time
import unittest
from unittest.mock import patch

from main import GlycanPipelineRunner


class StaticParallelismTests(unittest.TestCase):
    def test_static_only_uses_configured_parallel_workers(self):
        runner = GlycanPipelineRunner.__new__(GlycanPipelineRunner)
        runner.force_update = True
        runner.discover_static_only_glycans = lambda: (
            ["glycan-a", "glycan-b", "glycan-c", "glycan-d"],
            {},
        )
        runner.run_database_finalization = lambda: True

        lock = threading.Lock()
        active = 0
        peak_active = 0

        def run_one(_glycan_name):
            nonlocal active, peak_active
            with lock:
                active += 1
                peak_active = max(peak_active, active)
            time.sleep(0.03)
            with lock:
                active -= 1
            return {"glystatic": True, "glymeta": True, "skipped": False}

        runner.run_static_only_glycan = run_one

        with patch("main.config.static_workers", 4):
            results = runner.run_static_only_pipeline()

        self.assertGreater(peak_active, 1)
        self.assertEqual(results["successful_glycans"], 4)
        self.assertEqual(results["failed_glycans"], 0)
        self.assertTrue(results["finalization_success"])


if __name__ == "__main__":
    unittest.main()
