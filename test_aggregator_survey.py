import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

from aggregator_survey import ensure_output_directory, probe_identity


class _Messages:
    def __init__(self):
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(content=[SimpleNamespace(type="text", text="Model identity")])


class AggregatorSurveyTests(unittest.TestCase):
    def test_identity_probe_forwards_generation_settings(self):
        messages = _Messages()
        client = SimpleNamespace(messages=messages)

        entry = probe_identity(client, "provider/model", max_tokens=1200, temperature=0.2)

        self.assertEqual(entry["response"], "Model identity")
        self.assertEqual(messages.calls, [{
            "model": "provider/model",
            "max_tokens": 1200,
            "temperature": 0.2,
            "messages": [{"role": "user", "content": entry["probe"]}],
        }])

    def test_identity_probe_omits_unspecified_temperature(self):
        messages = _Messages()
        client = SimpleNamespace(messages=messages)

        probe_identity(client, "provider/model")

        self.assertNotIn("temperature", messages.calls[0])

    def test_output_path_creates_missing_parent_directory(self):
        with TemporaryDirectory() as temp_dir:
            out_path = Path(temp_dir) / "reports" / "nested" / "run.json"

            ensure_output_directory(str(out_path))

            self.assertTrue(out_path.parent.is_dir())

    def test_bare_output_filename_does_not_create_a_directory(self):
        with patch("aggregator_survey.os.makedirs") as makedirs:
            ensure_output_directory("run.json")

        makedirs.assert_not_called()


if __name__ == "__main__":
    unittest.main()
