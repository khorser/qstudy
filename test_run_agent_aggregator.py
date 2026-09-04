import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import run_agent_aggregator


class RunAgentAggregatorCliTests(unittest.TestCase):
    def test_load_credentials_uses_local_file_when_environment_is_unset(self):
        with TemporaryDirectory() as temp_dir:
            credentials_file = Path(temp_dir) / ".aggregator_credentials.json"
            credentials_file.write_text(json.dumps({
                "base_url": "https://provider.example/v1",
                "api_key": "local-key",
            }))
            with (patch.object(run_agent_aggregator, "CREDENTIALS_FILE", str(credentials_file)),
                  patch.dict(os.environ, {}, clear=True)):
                run_agent_aggregator.load_credentials()

                self.assertEqual(os.environ["OPENAI_BASE_URL"], "https://provider.example/v1")
                self.assertEqual(os.environ["OPENAI_API_KEY"], "local-key")

    def test_load_credentials_preserves_environment_overrides(self):
        with TemporaryDirectory() as temp_dir:
            credentials_file = Path(temp_dir) / ".aggregator_credentials.json"
            credentials_file.write_text(json.dumps({
                "base_url": "https://file.example/v1",
                "api_key": "file-key",
            }))
            environment = {
                "OPENAI_BASE_URL": "https://environment.example/v1",
                "OPENAI_API_KEY": "environment-key",
            }
            with (patch.object(run_agent_aggregator, "CREDENTIALS_FILE", str(credentials_file)),
                  patch.dict(os.environ, environment, clear=True)):
                run_agent_aggregator.load_credentials()

                self.assertEqual(os.environ["OPENAI_BASE_URL"], environment["OPENAI_BASE_URL"])
                self.assertEqual(os.environ["OPENAI_API_KEY"], environment["OPENAI_API_KEY"])


if __name__ == "__main__":
    unittest.main()
