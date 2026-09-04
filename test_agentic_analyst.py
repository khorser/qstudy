import unittest
from unittest.mock import patch

import requests

from agentic_analyst import DeutschJozsa, run_agent_aggregator


class _Response:
    def __init__(self, body):
        self.body = body

    def raise_for_status(self):
        pass

    def json(self):
        return self.body


def _tool_call_response(name, arguments="{}", call_id="call_123"):
    return _Response({"choices": [{"message": {
        "content": None,
        "tool_calls": [{
            "id": call_id,
            "type": "function",
            "function": {"name": name, "arguments": arguments},
        }],
    }}]})


class RunAgentAggregatorTests(unittest.TestCase):
    def test_preserves_tool_call_id_and_null_content(self):
        responses = [
            _Response({"choices": [{"message": {
                "content": None,
                "tool_calls": [{
                    "id": "call_123",
                    "type": "function",
                    "function": {"name": "list_oracles", "arguments": "{}"},
                }],
            }}]}),
            _Response({"choices": [{"message": {"content": "Grounded answer."}}]}),
        ]

        with patch("requests.post", side_effect=responses) as post:
            answer, trace = run_agent_aggregator(
                "https://example.test/v1", "test-key", DeutschJozsa(3), "question",
                model="test-model", verbose=False,
            )

        self.assertEqual(answer, "Grounded answer.")
        self.assertEqual(trace[0][0], "list_oracles")
        second_payload = post.call_args_list[1].kwargs["json"]
        assistant_tool_message = next(
            message for message in second_payload["messages"]
            if message.get("tool_calls")
        )
        self.assertIsNone(assistant_tool_message["content"])
        tool_result_message = next(
            message for message in second_payload["messages"]
            if message["role"] == "tool"
        )
        self.assertEqual(tool_result_message["tool_call_id"], "call_123")

    def test_rejects_tool_calls_without_an_id(self):
        response = _Response({"choices": [{"message": {
            "content": None,
            "tool_calls": [{"function": {"name": "list_oracles", "arguments": "{}"}}],
        }}]})

        with patch("requests.post", return_value=response):
            with self.assertRaisesRegex(RuntimeError, "without an id"):
                run_agent_aggregator(
                    "https://example.test/v1", "test-key", DeutschJozsa(3), "question",
                    model="test-model", verbose=False,
                )

    def test_preserves_each_id_when_a_response_contains_multiple_tool_calls(self):
        responses = [
            _Response({"choices": [{"message": {
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_oracles",
                        "type": "function",
                        "function": {"name": "list_oracles", "arguments": "{}"},
                    },
                    {
                        "id": "call_circuit",
                        "type": "function",
                        "function": {"name": "run_circuit", "arguments": '{"oracle": "xor"}'},
                    },
                ],
            }}]}),
            _Response({"choices": [{"message": {"content": "Grounded answer."}}]}),
        ]

        with patch("requests.post", side_effect=responses) as post:
            answer, trace = run_agent_aggregator(
                "https://example.test/v1", "test-key", DeutschJozsa(3), "question",
                model="test-model", verbose=False,
            )

        self.assertEqual(answer, "Grounded answer.")
        self.assertEqual([entry[0] for entry in trace], ["list_oracles", "run_circuit"])
        second_payload = post.call_args_list[1].kwargs["json"]
        tool_results = [message for message in second_payload["messages"]
                        if message["role"] == "tool"]
        self.assertEqual([message["tool_call_id"] for message in tool_results],
                         ["call_oracles", "call_circuit"])

    def test_rejects_empty_or_malformed_provider_responses(self):
        for body in ({}, {"choices": []}, {"choices": [None]}, {"choices": [{}]}, []):
            with self.subTest(body=body), patch("requests.post", return_value=_Response(body)):
                with self.assertRaisesRegex(RuntimeError, "no assistant message"):
                    run_agent_aggregator(
                        "https://example.test/v1", "test-key", DeutschJozsa(3), "question",
                        model="test-model", verbose=False,
                    )

    def test_propagates_http_and_timeout_errors(self):
        for error in (requests.HTTPError("bad gateway"), requests.Timeout("timed out")):
            with self.subTest(error=error), patch("requests.post", side_effect=error):
                with self.assertRaises(type(error)):
                    run_agent_aggregator(
                        "https://example.test/v1", "test-key", DeutschJozsa(3), "question",
                        model="test-model", verbose=False,
                    )

    def test_returns_tool_error_for_invalid_or_non_object_arguments(self):
        for arguments in ("not json", "[1, 2]"):
            responses = [_tool_call_response("list_oracles", arguments),
                         _Response({"choices": [{"message": {"content": "Done."}}]})]
            with self.subTest(arguments=arguments), patch("requests.post", side_effect=responses):
                answer, trace = run_agent_aggregator(
                    "https://example.test/v1", "test-key", DeutschJozsa(3), "question",
                    model="test-model", verbose=False,
                )

            self.assertEqual(answer, "Done.")
            self.assertEqual(trace[0][1], {})
            self.assertIn("invalid arguments", trace[0][2]["error"])

    def test_returns_tool_error_for_an_unknown_tool(self):
        responses = [_tool_call_response("not_a_tool"),
                     _Response({"choices": [{"message": {"content": "Done."}}]})]
        with patch("requests.post", side_effect=responses):
            answer, trace = run_agent_aggregator(
                "https://example.test/v1", "test-key", DeutschJozsa(3), "question",
                model="test-model", verbose=False,
            )

        self.assertEqual(answer, "Done.")
        self.assertEqual(trace[0][2], {"error": "unknown tool not_a_tool"})

    def test_returns_bounded_exit_when_max_steps_is_reached(self):
        with patch("requests.post", return_value=_tool_call_response("list_oracles")):
            answer, trace = run_agent_aggregator(
                "https://example.test/v1", "test-key", DeutschJozsa(3), "question",
                model="test-model", max_steps=1, verbose=False,
            )

        self.assertEqual(answer, "[max_steps reached without a final answer]")
        self.assertEqual(len(trace), 1)


if __name__ == "__main__":
    unittest.main()
