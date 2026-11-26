from contextlib import redirect_stderr
from io import StringIO
import unittest

from progress import progress, timed


class ProgressLogTests(unittest.TestCase):
    def test_redirected_progress_is_line_based_and_throttled(self):
        log = StringIO()
        with redirect_stderr(log), timed("Synthetic stage"):
            self.assertEqual(sum(progress(range(50), label="Subjects", unit="subject")), 1225)

        output = log.getvalue()
        self.assertNotIn("\r", output)
        self.assertIn("Subjects: 50/50", output)
        self.assertIn("subject/s", output)
        self.assertIn("Synthetic stage — done in", output)
        self.assertLessEqual(sum(line.startswith("Subjects:") for line in output.splitlines()), 3)
