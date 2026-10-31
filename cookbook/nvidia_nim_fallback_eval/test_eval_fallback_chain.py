import io
import json
import os
import subprocess
import sys
import tempfile
import threading
import unittest
import urllib.error
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import eval_fallback_chain as evaluator


def completion(content="ACKNOWLEDGED", finish="stop", tool_calls=None):
    message = {"role": "assistant", "content": content}
    if tool_calls is not None:
        message["tool_calls"] = tool_calls
    return {
        "model": "served-model",
        "choices": [{"finish_reason": finish, "message": message}],
        "usage": {"prompt_tokens": 9, "completion_tokens": 4, "completion_tokens_details": {"reasoning_tokens": 2}},
    }


def tool_call():
    return {
        "id": "call_1",
        "type": "function",
        "function": {"name": "get_stock_price", "arguments": '{"ticker":"NVDA"}'},
    }


class FakeResponse(io.BytesIO):
    def __init__(self, payload):
        super().__init__(json.dumps(payload).encode())


class EvaluatorTests(unittest.TestCase):
    def test_resume_only_schedules_missing_failed_and_incomplete_tool_pairs(self):
        good = evaluator.Result("m", "instruction", True, 200, 1, 1, "")
        failed = evaluator.Result("m", "logic", False, 429, 1, 0, "limited", error_kind="rate_limit")
        tool = evaluator.Result("m", "tool_select", True, 200, 1, 1, "")
        rows = (good, failed, tool)
        by_name = {task.name: task for task in evaluator.TASKS}
        self.assertFalse(evaluator.needs_attempt(by_name["instruction"], "m", 1, rows))
        self.assertTrue(evaluator.needs_attempt(by_name["logic"], "m", 1, rows))
        self.assertTrue(evaluator.needs_attempt(by_name["json_extract"], "m", 1, rows))
        self.assertTrue(evaluator.needs_attempt(by_name["tool_select"], "m", 1, rows))
        continued = evaluator.Result("m", "tool_select_continuation", True, 200, 1, 1, "")
        self.assertFalse(evaluator.needs_attempt(by_name["tool_select"], "m", 1, (*rows, continued)))
        replacement = evaluator.Result("m", "logic", True, 200, 2, 1, "")
        merged, history = evaluator.merge_result(rows, (), replacement)
        self.assertEqual(len(merged), 3)
        self.assertIn(failed, history)
        self.assertIn(replacement, merged)
        refreshed_failure = evaluator.Result("m", "tool_select", False, 429, 1, 0, "limited", error_kind="rate_limit")
        retained, history = evaluator.merge_result(merged, history, refreshed_failure)
        self.assertIn(tool, retained)
        self.assertIn(refreshed_failure, history)

    def test_resume_validates_profile_and_preserves_failed_attempt_history(self):
        metadata = {
            "models": ["m"],
            "repeats": 1,
            "profiles": {"m": {"temperature": 0}},
            "key_route": "one",
            "max_tokens": 42,
            "timeout": 12,
            "workers": 3,
        }
        failed = evaluator.Result("m", "logic", False, 429, 1, 0, "limited", error_kind="rate_limit")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "resume.json"
            evaluator.save_checkpoint(path, (failed,), metadata, complete=False, history=(failed,))
            rows, history = evaluator.load_resume(path, {**metadata, "workers": 1})
            self.assertEqual(rows, (failed,))
            self.assertEqual(history, (failed,))
            with self.assertRaises(ValueError):
                evaluator.load_resume(path, {**metadata, "key_route": "two"})
            with self.assertRaises(ValueError):
                evaluator.load_resume(path, {**metadata, "profiles": {"m": {"temperature": 1}}})

    def test_request_pacer_spaces_starts_using_shared_clock(self):
        now = [0.0]
        sleeps = []

        def sleep(delay):
            sleeps.append(delay)
            now[0] += delay

        pacer = evaluator.RequestPacer(0.25, clock=lambda: now[0], sleeper=sleep)
        pacer.wait()
        pacer.wait()
        pacer.wait()
        self.assertEqual(sleeps, [0.25, 0.25])
        self.assertEqual(now[0], 0.5)

    def test_resume_rejects_corrupted_result_and_history_records(self):
        metadata = {"models": ["m"], "repeats": 1, "profiles": {}, "key_route": "one", "max_tokens": 42, "timeout": 12}
        valid = evaluator.asdict(evaluator.Result("m", "instruction", True, 200, 1.5, 3, "", finish_reason="stop"))
        changes = (
            {"model": "unknown_model"},
            {"task": "unknown_task"},
            {"trial": 999},
            {"trial": True},
            {"passed": "false"},
            {"status": True},
            {"status": -1},
            {"completion_tokens": -1},
            {"prompt_tokens": "10"},
            {"reasoning_tokens": 1.2},
            {"latency_s": float("nan")},
            {"latency_s": float("inf")},
            {"latency_s": -1},
            {"latency_s": True},
            {"status": 429},
            {"finish_reason": "length"},
            {"error_kind": "wrong_answer"},
            {"error": "Failed"},
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "resume.json"
            for collection in ("results", "history"):
                for change in changes:
                    with self.subTest(collection=collection, change=change):
                        payload = {
                            "schema": 2,
                            "complete": False,
                            "metadata": metadata,
                            "results": [valid],
                            "history": [],
                        }
                        payload[collection] = [{**valid, **change}]
                        path.write_text(json.dumps(payload))
                        with self.assertRaises(ValueError):
                            evaluator.load_resume(path, metadata)
            for schema in (1, "2", True, None):
                path.write_text(
                    json.dumps({"schema": schema, "complete": False, "metadata": metadata, "results": [valid]})
                )
                with self.subTest(schema=schema), self.assertRaises(ValueError):
                    evaluator.load_resume(path, metadata)
            path.write_text(json.dumps({"schema": 2, "complete": False, "metadata": metadata, "results": [valid]}))
            rows, history = evaluator.load_resume(path, metadata)
            self.assertEqual(len(rows), 1)
            self.assertEqual(history, ())

    def test_json_requires_only_object_integer_and_unique_keys(self):
        valid = '{"name":"Maria Lopez","age":34,"city":"Lisbon"}'
        self.assertTrue(evaluator.grade_json({"content": valid}))
        for invalid in (
            "Answer: " + valid,
            valid + " done",
            valid.replace("34", "34.0"),
            valid.replace('"age":34', '"age":false'),
            valid.replace('"name":"Maria Lopez"', '"name":"wrong","name":"Maria Lopez"'),
        ):
            with self.subTest(invalid=invalid):
                self.assertFalse(evaluator.grade_json({"content": invalid}))

    def test_exact_retrieval_and_single_valid_tool(self):
        task = next(task for task in evaluator.TASKS if task.name == "long_context")
        self.assertTrue(task.grade({"content": evaluator.NEEDLE}))
        self.assertFalse(task.grade({"content": "Answer: " + evaluator.NEEDLE}))
        self.assertTrue(evaluator.grade_stock({"tool_calls": [tool_call()]}))
        for calls in (
            [None],
            [tool_call(), tool_call()],
            [{"function": None}],
            [{"function": {"name": "get_stock_price", "arguments": []}}],
        ):
            self.assertFalse(evaluator.grade_stock({"tool_calls": calls}))

    def test_truncated_answer_never_passes_and_usage_is_retained(self):
        task = evaluator.TASKS[0]
        result, _ = evaluator.request_task(
            "http://example.invalid/v1",
            "test",
            "candidate",
            task,
            trial=2,
            opener=lambda *args, **kwargs: FakeResponse(completion(finish="length")),
        )
        self.assertFalse(result.passed)
        self.assertEqual(result.error_kind, "truncated")
        self.assertEqual(result.finish_reason, "length")
        self.assertEqual(
            (result.trial, result.returned_model, result.prompt_tokens, result.reasoning_tokens),
            (2, "served-model", 9, 2),
        )

    def test_malformed_completion_is_a_result(self):
        for payload in ({}, {"choices": []}, {"choices": [None]}, {"choices": [{"message": None}]}, []):
            result, _ = evaluator.request_task(
                "http://example.invalid/v1",
                "test",
                "candidate",
                evaluator.TASKS[0],
                opener=lambda *args, **kwargs: FakeResponse(payload),
            )
            self.assertFalse(result.passed)
            self.assertEqual(result.error_kind, "protocol")

    def test_http_and_timeout_classification(self):
        for error, kind in (
            (TimeoutError("slow"), "timeout"),
            (
                urllib.error.HTTPError("http://example.invalid", 429, "limited", {}, io.BytesIO(b"private")),
                "rate_limit",
            ),
        ):

            def fail(*args, **kwargs):
                raise error

            result, _ = evaluator.request_task(
                "http://example.invalid/v1", "test", "candidate", evaluator.TASKS[0], opener=fail
            )
            self.assertEqual(result.error_kind, kind)
            self.assertNotIn("private", result.error)

    def test_profile_overrides_and_real_tool_continuation_are_sent(self):
        task = next(task for task in evaluator.TASKS if task.name == "tool_select")
        requests = []

        def respond(request, **kwargs):
            requests.append((json.loads(request.data), kwargs["timeout"]))
            payload = completion(None, "tool_calls", [tool_call()])
            payload["choices"][0]["message"].update({"reasoning_content": "canonical", "reasoning": "duplicate"})
            return FakeResponse(payload)

        result, message = evaluator.request_task(
            "http://example.invalid/v1",
            "test",
            "candidate",
            task,
            timeout=23,
            max_tokens=8192,
            profile={"chat_template_kwargs": {"enable_thinking": True}},
            opener=respond,
        )
        continuation = evaluator.continuation_task(task, message)
        self.assertTrue(result.passed)
        self.assertIsNotNone(continuation)
        assistant = continuation.messages[-3]
        self.assertEqual(assistant["reasoning_content"], "canonical")
        self.assertNotIn("reasoning", assistant)
        self.assertEqual(continuation.messages[-2]["tool_call_id"], "call_1")
        self.assertIn("TOOL_RESULT_OK", continuation.messages[-2]["content"])
        self.assertEqual(requests[0][0]["chat_template_kwargs"], {"enable_thinking": True})
        self.assertEqual((requests[0][0]["max_tokens"], requests[0][1]), (8192, 23))
        continued, _ = evaluator.request_task(
            "http://example.invalid/v1",
            "test",
            "candidate",
            continuation,
            opener=lambda *args, **kwargs: FakeResponse(completion("TOOL_RESULT_OK")),
        )
        self.assertTrue(continued.passed)

    def test_cli_checkpoints_survive_endpoint_failures(self):
        requests = []

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                requests.append(json.loads(self.rfile.read(int(self.headers["Content-Length"]))))
                self.send_response(429)
                self.end_headers()
                self.wfile.write(b'{"error":"limited"}')

            def log_message(self, *args):
                return

        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "results.json"
                environment = {
                    **os.environ,
                    "EVAL_API_KEY": "test",
                    "EVAL_MODELS": "unavailable",
                    "EVAL_REPEATS": "1",
                    "EVAL_WORKERS": "2",
                    "EVAL_OUT": str(path),
                    "EVAL_BASE_URL": f"http://127.0.0.1:{server.server_port}/v1",
                    "EVAL_PROFILES_JSON": json.dumps({"unavailable": {"reasoning_budget": 4096}}),
                }
                process = subprocess.run(
                    [sys.executable, evaluator.__file__], env=environment, capture_output=True, text=True, timeout=10
                )
                self.assertEqual(process.returncode, 0, process.stderr)
                saved = json.loads(path.read_text())
                self.assertTrue(saved["complete"])
                self.assertEqual(len(saved["results"]), 10)
                self.assertEqual(saved["summaries"][0]["error_counts"], {"rate_limit": 8, "dependency": 2})
                self.assertIsNone(saved["summaries"][0]["conditional_accuracy"])
                self.assertEqual(saved["metadata"]["profiles"], {"unavailable": {"reasoning_budget": 4096}})
                self.assertTrue(all(request["reasoning_budget"] == 4096 for request in requests))
                previous = tuple(evaluator.Result(**row) for row in saved["results"])
                passing = evaluator.Result("unavailable", "instruction", True, 200, 1, 1, "", finish_reason="stop")
                rows, history = evaluator.merge_result(previous, (), passing)
                evaluator.save_checkpoint(path, rows, saved["metadata"], complete=False, history=history)
                resumed = subprocess.run(
                    [sys.executable, evaluator.__file__],
                    env={**environment, "EVAL_RESUME": "1", "EVAL_WORKERS": "1"},
                    capture_output=True,
                    text=True,
                    timeout=10,
                )
                self.assertEqual(resumed.returncode, 0, resumed.stderr)
                self.assertEqual(len(requests), 15)
                self.assertEqual(sum("ACKNOWLEDGED" in request["messages"][0]["content"] for request in requests), 1)
                resumed_output = json.loads(path.read_text())
                self.assertEqual(len(resumed_output["results"]), 10)
                self.assertEqual(len(resumed_output["history"]), 10)
                self.assertEqual(resumed_output["history_error_counts"]["rate_limit"], 8)
                self.assertTrue(
                    next(row for row in resumed_output["results"] if row["task"] == "instruction")["passed"]
                )
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=2)

    def test_cli_rejects_unknown_profile_fields_before_requesting(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "results.json"
            process = subprocess.run(
                [sys.executable, evaluator.__file__],
                check=False,
                env={
                    **os.environ,
                    "EVAL_API_KEY": "test",
                    "EVAL_MODELS": "candidate",
                    "EVAL_OUT": str(path),
                    "EVAL_BASE_URL": "http://example.invalid/v1",
                    "EVAL_PROFILES_JSON": json.dumps({"candidate": {"unsupported_control": 1}}),
                },
                capture_output=True,
                text=True,
                timeout=10,
            )
            self.assertNotEqual(process.returncode, 0)
            self.assertIn("supported generation parameters", process.stderr)
            self.assertFalse(path.exists())

    def test_checkpoint_and_metrics_distinguish_failure_types(self):
        good = evaluator.Result("m", "instruction", True, 200, 1, 4, "")
        wrong = evaluator.Result("m", "logic", False, 200, 2, 4, "wrong", error_kind="wrong_answer")
        timeout = evaluator.Result("m", "json", False, 0, 30, 0, "timeout", error_kind="timeout")
        summary = evaluator.summarize("m", (good, wrong, timeout))
        self.assertEqual(summary["overall_success"], 1 / 3)
        self.assertEqual(summary["availability"], 2 / 3)
        self.assertEqual(summary["conditional_accuracy"], 1 / 2)
        self.assertEqual(summary["all_attempt_p90_s"], 30)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "results.json"
            evaluator.save_checkpoint(path, (good, timeout), {"run": "test"}, complete=False)
            evaluator.save_checkpoint(path, (good, wrong, timeout), {"run": "test"}, complete=True)
            saved = json.loads(path.read_text(), parse_constant=lambda value: self.fail(value))
            self.assertTrue(saved["complete"])
            self.assertEqual(len(saved["results"]), 3)
            self.assertEqual(saved["metadata"], {"run": "test"})


if __name__ == "__main__":
    unittest.main()
