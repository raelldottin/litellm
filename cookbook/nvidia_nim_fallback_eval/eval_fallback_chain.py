"""Score NVIDIA NIM chat models for a LiteLLM fallback chain.

Runs a fixed set of deterministic, auto-graded tasks (instruction following, arithmetic, JSON extraction,
tool calling, tool selection, code tracing, long-context retrieval, logic) against each model through any
OpenAI-compatible endpoint, and prints a ranked table plus per-task failures.

Stdlib only so it runs inside the LiteLLM Docker image as-is:

    docker exec -i -e EVAL_REPEATS=3 litellm python - < eval_fallback_chain.py

Env:
    EVAL_BASE_URL   default https://integrate.api.nvidia.com/v1 (or http://localhost:4000/v1 for the proxy)
    EVAL_API_KEY    default $NVIDIA_NIM_API_KEY
    EVAL_MODELS     comma separated, default is the candidate list below
    EVAL_REPEATS    runs per task, default 3
    EVAL_WORKERS    parallel requests per model, default 3
    EVAL_OUT        path for the raw JSON results, default fallback_eval_results.json
    EVAL_TIMEOUT    request inactivity timeout in seconds, default 180
    EVAL_MAX_TOKENS  output budget, default 4096
    EVAL_PROFILES_JSON  per-model native generation settings as JSON
    EVAL_KEY_ROUTE   non-secret label identifying this credential trial
    EVAL_MIN_REQUEST_INTERVAL  seconds between request starts, default 0
    EVAL_RESUME      set to 1 to resume an existing matching checkpoint
"""

import http.client
import json
import math
import os
import re
import statistics
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.request
from collections import Counter
from collections.abc import Callable
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Final

DEFAULT_MODELS: Final = (
    "z-ai/glm-5.3-flash",
    "deepseek-ai/deepseek-v4.1-flash",
    "minimaxai/minimax-m3.1-flash",
    "meta/muse-glimmer-30b",
    "moonshotai/kimi-k2.6",
    "stepfun-ai/step-5-preview",
    "nvidia/nemotron-3-ultra-550b-a55b",
    "nvidia/nemotron-3.5-super-vl-120b-a12b",
    "nvidia/nemotron-3.5-lightning-30b-a3b",
    "google/gemma-4-31b-it",
    "poolside/laguna-xs-2.1",
)

WEATHER_TOOL: Final = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Current weather for a city",
        "parameters": {
            "type": "object",
            "properties": {
                "city": {"type": "string"},
                "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
            },
            "required": ["city", "unit"],
        },
    },
}
STOCK_TOOL: Final = {
    "type": "function",
    "function": {
        "name": "get_stock_price",
        "description": "Latest stock price for a ticker symbol",
        "parameters": {
            "type": "object",
            "properties": {"ticker": {"type": "string"}},
            "required": ["ticker"],
        },
    },
}

NEEDLE: Final = "PELICAN-4471"
HAYSTACK: Final = "\n".join(
    f"Log line {i}: service heartbeat ok, latency {(i * 37) % 200} ms, region us-east-{i % 4}."
    + (f" The deployment passphrase is {NEEDLE}." if i == 1300 else "")
    for i in range(2000)
)

CODE_SNIPPET: Final = """
def f(xs):
    out = []
    for i, x in enumerate(xs):
        if x % 3 == 0:
            out.append(x * i)
        elif x % 2 == 0:
            out.insert(0, x)
    return out

print(f([3, 4, 6, 7, 8, 9]))
""".strip()


@dataclass(frozen=True, slots=True)
class Task:
    name: str
    messages: tuple[dict[str, object], ...]
    grade: Callable[[dict[str, object]], bool]
    tools: tuple[dict[str, object], ...] = ()


def strip_reasoning(text: str) -> str:
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


def content_of(message: dict[str, object]) -> str:
    content = message.get("content")
    return strip_reasoning(content) if isinstance(content, str) else ""


def first_tool_call(message: dict[str, object]) -> tuple[str, dict[str, object]] | None:
    calls = message.get("tool_calls")
    if not isinstance(calls, list) or len(calls) != 1 or not isinstance(calls[0], dict):
        return None
    fn = calls[0].get("function", {})
    if not isinstance(fn, dict) or not isinstance(fn.get("name"), str) or not isinstance(fn.get("arguments"), str):
        return None
    args = parse_json_object(fn["arguments"])
    return (fn["name"], args) if args is not None else None


def unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    if len({key for key, _ in pairs}) != len(pairs):
        raise ValueError("Duplicate JSON object key")
    return dict(pairs)


def parse_json_object(text: str) -> dict[str, object] | None:
    try:
        parsed = json.loads(text, object_pairs_hook=unique_object)
    except (ValueError, TypeError):
        return None
    return parsed if isinstance(parsed, dict) else None


def user(text: str) -> tuple[dict[str, str], ...]:
    return ({"role": "user", "content": text},)


def grade_json(message: dict[str, object]) -> bool:
    obj = parse_json_object(content_of(message))
    return (
        obj is not None and type(obj.get("age")) is int and obj == {"name": "Maria Lopez", "age": 34, "city": "Lisbon"}
    )


def grade_weather(message: dict[str, object]) -> bool:
    call = first_tool_call(message)
    return (
        call is not None
        and call[0] == "get_weather"
        and str(call[1].get("city", "")).strip().lower() in {"paris", "paris, france", "paris france"}
        and call[1].get("unit") == "celsius"
    )


def grade_stock(message: dict[str, object]) -> bool:
    call = first_tool_call(message)
    return call is not None and call[0] == "get_stock_price" and str(call[1].get("ticker", "")).upper() == "NVDA"


def last_line(message: dict[str, object]) -> str:
    lines = [line.strip() for line in content_of(message).splitlines() if line.strip()]
    return lines[-1] if lines else ""


TASKS: Final = (
    Task(
        "instruction",
        user("Reply with exactly the word ACKNOWLEDGED in uppercase. No punctuation, no other text."),
        lambda m: content_of(m) == "ACKNOWLEDGED",
    ),
    Task(
        "arithmetic",
        user("Compute 17 * 23 + 144 / 12 - 2^5. Put only the final integer on the last line."),
        lambda m: last_line(m).strip("*` .") == "371",
    ),
    Task(
        "json_extract",
        user(
            "Extract name, age (integer) and city from this text as a single JSON object with exactly those keys "
            "and no other text: 'Last spring, Maria Lopez, who had just turned 34, moved from Porto to Lisbon.'"
        ),
        grade_json,
    ),
    Task(
        "tool_call",
        user("What's the weather like in Paris right now? I prefer Celsius."),
        grade_weather,
        (WEATHER_TOOL, STOCK_TOOL),
    ),
    Task(
        "tool_select",
        user("How is NVIDIA's stock doing at the moment?"),
        grade_stock,
        (WEATHER_TOOL, STOCK_TOOL),
    ),
    Task(
        "code_trace",
        user(f"What does this Python program print? Put only the printed output on the last line.\n\n{CODE_SNIPPET}"),
        lambda m: last_line(m).strip("`") == "[8, 4, 0, 12, 45]",
    ),
    Task(
        "long_context",
        user(f"{HAYSTACK}\n\nWhat is the deployment passphrase mentioned in the logs above? Reply with it only."),
        lambda m: content_of(m) == NEEDLE,
    ),
    Task(
        "logic",
        user(
            "Alice, Bob and Carol each own exactly one pet: a cat, a dog or a fish. Alice does not own the dog. "
            "Bob owns neither the dog nor the fish. Who owns the dog? Put only the name on the last line."
        ),
        lambda m: last_line(m).strip("*`. ").lower() == "carol",
    ),
)


@dataclass(frozen=True, slots=True)
class Result:
    model: str
    task: str
    passed: bool
    status: int
    latency_s: float
    completion_tokens: int
    error: str
    trial: int = 1
    finish_reason: str = ""
    returned_model: str = ""
    prompt_tokens: int = 0
    reasoning_tokens: int = 0
    error_kind: str = ""


def token_count(value: object) -> int:
    return value if type(value) is int and value >= 0 else 0


class RequestPacer:
    def __init__(self, interval: float, *, clock: Callable = time.monotonic, sleeper: Callable = time.sleep):
        self.interval = interval
        self.clock = clock
        self.sleeper = sleeper
        self.lock = threading.Lock()
        self.last_started = None

    def wait(self) -> None:
        with self.lock:
            if self.last_started is not None:
                delay = self.interval - (self.clock() - self.last_started)
                if delay > 0:
                    self.sleeper(delay)
            self.last_started = self.clock()


def request_task(
    base_url: str,
    api_key: str,
    model: str,
    task: Task,
    *,
    trial: int = 1,
    timeout: float = 180,
    max_tokens: int = 4096,
    profile: dict[str, object] | None = None,
    opener: Callable = urllib.request.urlopen,
    pacer: RequestPacer | None = None,
) -> tuple[Result, dict[str, object] | None]:
    body = {
        "temperature": 0,
        "max_tokens": max_tokens,
        **(profile or {}),
        "model": model,
        "messages": list(task.messages),
        **({"tools": list(task.tools)} if task.tools else {}),
    }
    request = urllib.request.Request(
        f"{base_url.rstrip('/')}/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
    )
    if pacer is not None:
        pacer.wait()
    start = time.monotonic()
    try:
        with opener(request, timeout=timeout) as response:
            payload = json.load(response)
    except urllib.error.HTTPError as e:
        status = e.code
        e.close()
        kind = "rate_limit" if status == 429 else "http"
        return Result(
            model, task.name, False, status, time.monotonic() - start, 0, f"HTTP {status}", trial=trial, error_kind=kind
        ), None
    except (TimeoutError, urllib.error.URLError, OSError, http.client.HTTPException) as e:
        kind = (
            "timeout"
            if isinstance(e, TimeoutError) or isinstance(getattr(e, "reason", None), TimeoutError)
            else "transport"
        )
        return Result(model, task.name, False, 0, time.monotonic() - start, 0, kind, trial=trial, error_kind=kind), None
    except (ValueError, UnicodeError):
        return Result(
            model,
            task.name,
            False,
            200,
            time.monotonic() - start,
            0,
            "Invalid JSON response",
            trial=trial,
            error_kind="protocol",
        ), None
    elapsed = time.monotonic() - start
    if not isinstance(payload, dict):
        return Result(
            model, task.name, False, 200, elapsed, 0, "Invalid completion envelope", trial=trial, error_kind="protocol"
        ), None
    choices = payload.get("choices")
    choice = choices[0] if isinstance(choices, list) and choices and isinstance(choices[0], dict) else {}
    message = choice.get("message")
    finish = choice.get("finish_reason")
    if not isinstance(message, dict) or not isinstance(finish, str):
        return Result(
            model,
            task.name,
            False,
            200,
            elapsed,
            0,
            "Missing message or finish_reason",
            trial=trial,
            error_kind="protocol",
        ), None
    usage = payload.get("usage") if isinstance(payload.get("usage"), dict) else {}
    details = usage.get("completion_tokens_details") if isinstance(usage.get("completion_tokens_details"), dict) else {}
    try:
        passed = finish in {"stop", "tool_calls"} and bool(task.grade(message))
    except (TypeError, ValueError, AttributeError, KeyError, IndexError):
        return Result(
            model,
            task.name,
            False,
            200,
            elapsed,
            0,
            "Malformed assistant message",
            trial=trial,
            finish_reason=finish,
            error_kind="protocol",
        ), None
    kind = (
        ""
        if passed
        else "truncated"
        if finish == "length"
        else "wrong_answer"
        if finish in {"stop", "tool_calls"}
        else "rejected"
    )
    detail = "" if passed else json.dumps({"content": message.get("content"), "tool_calls": message.get("tool_calls")})
    return Result(
        model,
        task.name,
        passed,
        200,
        elapsed,
        token_count(usage.get("completion_tokens")),
        detail[:300],
        trial=trial,
        finish_reason=finish,
        returned_model=payload.get("model") if isinstance(payload.get("model"), str) else "",
        prompt_tokens=token_count(usage.get("prompt_tokens")),
        reasoning_tokens=token_count(details.get("reasoning_tokens")),
        error_kind=kind,
    ), message


def continuation_task(task: Task, message: dict[str, object] | None) -> Task | None:
    if message is None or first_tool_call(message) is None:
        return None
    tool = message["tool_calls"][0]
    if not isinstance(tool.get("id"), str) or not tool["id"]:
        return None
    assistant = {key: value for key, value in message.items() if key != "reasoning"}
    if "reasoning_content" not in assistant and "reasoning" in message:
        assistant = {**assistant, "reasoning_content": message["reasoning"]}
    messages = (
        *task.messages,
        {**assistant, "role": "assistant"},
        {"role": "tool", "tool_call_id": tool["id"], "content": '{"confirmation":"TOOL_RESULT_OK"}'},
        {"role": "user", "content": "Reply with exactly the confirmation value supplied by the tool. No other text."},
    )
    return Task(
        task.name + "_continuation",
        messages,
        lambda answer: content_of(answer) == "TOOL_RESULT_OK" and not answer.get("tool_calls"),
        task.tools,
    )


def iter_model_results(
    base_url: str,
    api_key: str,
    model: str,
    repeats: int,
    workers: int,
    *,
    timeout: float = 180,
    max_tokens: int = 4096,
    profile: dict[str, object] | None = None,
    previous: tuple[Result, ...] = (),
    pacer: RequestPacer | None = None,
):
    with ThreadPoolExecutor(max_workers=workers) as pool:

        def submit(task, trial):
            return pool.submit(
                request_task,
                base_url,
                api_key,
                model,
                task,
                trial=trial,
                timeout=timeout,
                max_tokens=max_tokens,
                profile=profile,
                pacer=pacer,
            )

        pending = {
            submit(task, trial): (task, trial)
            for task in TASKS
            for trial in range(1, repeats + 1)
            if needs_attempt(task, model, trial, previous)
        }
        while pending:
            done, _ = wait(pending, return_when=FIRST_COMPLETED)
            for future in done:
                task, trial = pending.pop(future)
                result, message = future.result()
                yield result
                if not task.tools or task.name.endswith("_continuation"):
                    continue
                continuation = continuation_task(task, message) if result.passed else None
                if continuation is None:
                    yield Result(
                        model,
                        task.name + "_continuation",
                        False,
                        0,
                        0,
                        0,
                        "Initial tool call did not qualify",
                        trial=trial,
                        error_kind="dependency",
                    )
                else:
                    pending[submit(continuation, trial)] = (continuation, trial)


def percentile90(values: list[float]) -> float | None:
    return values[math.ceil(0.9 * len(values)) - 1] if values else None


def summarize(model: str, results: tuple[Result, ...]) -> dict[str, object]:
    attempted = tuple(r for r in results if r.error_kind != "dependency")
    ok = tuple(r for r in attempted if r.status == 200 and r.error_kind not in {"protocol", "truncated", "rejected"})
    latencies = sorted(r.latency_s for r in ok)
    all_latencies = sorted(r.latency_s for r in attempted)
    overall = sum(r.passed for r in results) / len(results) if results else 0
    return {
        "model": model,
        "score": overall,
        "overall_success": overall,
        "availability": len(ok) / len(attempted) if attempted else 0,
        "conditional_accuracy": sum(r.passed for r in ok) / len(ok) if ok else None,
        "p50_s": statistics.median(latencies) if latencies else None,
        "p90_s": percentile90(latencies),
        "all_attempt_p90_s": percentile90(all_latencies),
        "tok_per_s": sum(r.completion_tokens for r in ok) / max(sum(r.latency_s for r in ok), 1e-9),
        "failed_tasks": sorted({r.task for r in results if not r.passed}),
        "error_counts": dict(Counter(r.error_kind for r in results if r.error_kind)),
        "completed_results": len(results),
    }


def result_key(result: Result) -> tuple[str, str, int]:
    return result.model, result.task, result.trial


def needs_attempt(task: Task, model: str, trial: int, previous: tuple[Result, ...]) -> bool:
    passed = {(row.model, row.task, row.trial) for row in previous if row.passed}
    if (model, task.name, trial) not in passed:
        return True
    return bool(task.tools) and (model, task.name + "_continuation", trial) not in passed


def merge_result(
    results: tuple[Result, ...], history: tuple[Result, ...], result: Result
) -> tuple[tuple[Result, ...], tuple[Result, ...]]:
    old = next((row for row in results if result_key(row) == result_key(result)), None)
    if old is None:
        return (*results, result), history
    if old.passed:
        return results, (*history, result)
    replacement = tuple(result if result_key(row) == result_key(result) else row for row in results)
    return replacement, (*history, old)


def validate_resume_record(row: object, metadata: dict[str, object]) -> Result:
    if not isinstance(row, dict):
        raise ValueError("Resume result must be an object")
    try:
        result = Result(**row)
    except TypeError as error:
        raise ValueError("Invalid resume result fields") from error
    tasks = {task.name for task in TASKS} | {task.name + "_continuation" for task in TASKS if task.tools}
    if not isinstance(result.model, str) or result.model not in metadata["models"]:
        raise ValueError("Resume result model is outside the evaluation scope")
    if not isinstance(result.task, str) or result.task not in tasks:
        raise ValueError("Resume result task is outside the evaluation scope")
    if type(result.trial) is not int or not 1 <= result.trial <= metadata["repeats"]:
        raise ValueError("Resume result trial is outside the evaluation scope")
    if type(result.passed) is not bool:
        raise ValueError("Resume result passed must be a boolean")
    counts = (result.status, result.completion_tokens, result.prompt_tokens, result.reasoning_tokens)
    if any(type(value) is not int or value < 0 for value in counts):
        raise ValueError("Resume result status and token counts must be nonnegative integers")
    if type(result.latency_s) not in {int, float}:
        raise ValueError("Resume result latency must be a finite nonnegative number")
    try:
        valid_latency = math.isfinite(result.latency_s) and result.latency_s >= 0
    except OverflowError:
        valid_latency = False
    if not valid_latency:
        raise ValueError("Resume result latency must be a finite nonnegative number")
    if any(
        not isinstance(value, str)
        for value in (result.error, result.error_kind, result.finish_reason, result.returned_model)
    ):
        raise ValueError("Resume result text fields must be strings")
    if result.error_kind not in {
        "",
        "rate_limit",
        "http",
        "timeout",
        "transport",
        "protocol",
        "truncated",
        "wrong_answer",
        "rejected",
        "dependency",
    }:
        raise ValueError("Unknown resume result error kind")
    if result.passed and (
        result.status != 200 or result.finish_reason not in {"stop", "tool_calls"} or result.error_kind or result.error
    ):
        raise ValueError("Successful resume result has inconsistent completion evidence")
    return result


def load_resume(path: Path, metadata: dict[str, object]) -> tuple[tuple[Result, ...], tuple[Result, ...]]:
    payload = json.loads(path.read_text(), object_pairs_hook=unique_object)
    if not isinstance(payload, dict) or type(payload.get("schema")) is not int or payload["schema"] != 2:
        raise ValueError("Resume checkpoint must use schema 2")
    if type(payload.get("complete")) is not bool or not isinstance(payload.get("metadata"), dict):
        raise ValueError("Invalid resume checkpoint envelope")
    if not isinstance(payload.get("results"), list) or not isinstance(payload.get("history", []), list):
        raise ValueError("Resume results and history must be arrays")
    old_metadata = payload["metadata"]
    for field in ("models", "repeats", "profiles", "key_route", "max_tokens", "timeout"):
        if json.dumps(old_metadata.get(field), sort_keys=True) != json.dumps(metadata.get(field), sort_keys=True):
            raise ValueError("Resume configuration differs for " + field)
    if "base_url" in old_metadata and old_metadata["base_url"] != metadata.get("base_url"):
        raise ValueError("Resume configuration differs for base_url")
    results = tuple(validate_resume_record(row, metadata) for row in payload["results"])
    history = tuple(validate_resume_record(row, metadata) for row in payload.get("history", ()))
    if len({result_key(row) for row in results}) != len(results):
        raise ValueError("Resume checkpoint contains duplicate task/trial results")
    return results, history


def save_checkpoint(
    path: Path,
    results: tuple[Result, ...],
    metadata: dict[str, object],
    *,
    complete: bool,
    history: tuple[Result, ...] = (),
) -> None:
    models = tuple(dict.fromkeys(result.model for result in results))
    summaries = tuple(summarize(model, tuple(row for row in results if row.model == model)) for model in models)
    payload = {
        "schema": 2,
        "complete": complete,
        "metadata": metadata,
        "summaries": summaries,
        "results": tuple(asdict(result) for result in results),
        "history": tuple(asdict(result) for result in history),
        "history_error_counts": dict(Counter(result.error_kind for result in history if result.error_kind)),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, prefix=path.name + ".", delete=False) as output:
        json.dump(payload, output, allow_nan=False, indent=2)
        output.flush()
        os.fsync(output.fileno())
    os.replace(output.name, path)


def main() -> None:
    base_url: Final = os.environ.get("EVAL_BASE_URL", "https://integrate.api.nvidia.com/v1")
    api_key: Final = os.environ.get("EVAL_API_KEY") or os.environ.get("NVIDIA_NIM_API_KEY") or ""
    if not api_key:
        sys.exit("Set EVAL_API_KEY or NVIDIA_NIM_API_KEY")
    models: Final = tuple(m.strip() for m in os.environ.get("EVAL_MODELS", ",".join(DEFAULT_MODELS)).split(","))
    repeats: Final = int(os.environ.get("EVAL_REPEATS", "3"))
    workers: Final = int(os.environ.get("EVAL_WORKERS", "3"))
    timeout: Final = float(os.environ.get("EVAL_TIMEOUT", "180"))
    max_tokens: Final = int(os.environ.get("EVAL_MAX_TOKENS", "4096"))
    min_interval: Final = float(os.environ.get("EVAL_MIN_REQUEST_INTERVAL", "0"))
    profiles: Final = json.loads(os.environ.get("EVAL_PROFILES_JSON", "{}"))
    allowed: Final = {
        "temperature",
        "top_p",
        "seed",
        "max_tokens",
        "max_completion_tokens",
        "reasoning_effort",
        "chat_template_kwargs",
        "extra_body",
    }
    if (
        not models
        or any(not model for model in models)
        or min(repeats, workers, max_tokens) < 1
        or not math.isfinite(timeout)
        or timeout <= 0
        or not math.isfinite(min_interval)
        or min_interval < 0
    ):
        sys.exit("Models, repeats, workers, output budget and timeout must be nonempty and positive")
    if not isinstance(profiles, dict) or any(
        not isinstance(profile, dict) or set(profile) - allowed for profile in profiles.values()
    ):
        sys.exit("EVAL_PROFILES_JSON must map models to supported generation parameters")
    output: Final = Path(os.environ.get("EVAL_OUT", "fallback_eval_results.json"))
    metadata: Final = {
        "started_at": datetime.now(timezone.utc).isoformat(),
        "models": models,
        "repeats": repeats,
        "workers": workers,
        "base_url": base_url,
        "min_request_interval": min_interval,
        "timeout": timeout,
        "max_tokens": max_tokens,
        "profiles": profiles,
        "key_route": os.environ.get("EVAL_KEY_ROUTE", "unspecified"),
        "expected_results_per_model": repeats * (len(TASKS) + sum(bool(task.tools) for task in TASKS)),
    }
    if os.environ.get("EVAL_RESUME", "0") == "1":
        try:
            results, history = load_resume(output, metadata)
        except (OSError, ValueError, TypeError, KeyError) as error:
            sys.exit("Cannot resume checkpoint: " + str(error))
        previous_metadata = json.loads(output.read_text())["metadata"]
        run_metadata = {
            **metadata,
            "started_at": previous_metadata.get("started_at", metadata["started_at"]),
            "resume_history": (
                *previous_metadata.get("resume_history", ()),
                {
                    "resumed_at": metadata["started_at"],
                    "previous_workers": previous_metadata.get("workers"),
                    "previous_interval": previous_metadata.get("min_request_interval", 0),
                    "workers": workers,
                    "min_request_interval": min_interval,
                },
            ),
        }
    else:
        results, history = (), ()
        run_metadata = metadata
    pacer = RequestPacer(min_interval)
    save_checkpoint(output, results, run_metadata, complete=False, history=history)
    for model in models:
        for result in iter_model_results(
            base_url,
            api_key,
            model,
            repeats,
            workers,
            timeout=timeout,
            max_tokens=max_tokens,
            profile=profiles.get(model),
            previous=results,
            pacer=pacer,
        ):
            results, history = merge_result(results, history, result)
            save_checkpoint(output, results, run_metadata, complete=False, history=history)
            sys.stdout.write(
                f"{result.model} trial={result.trial} {result.task} "
                f"{'PASS' if result.passed else 'FAIL ' + result.error_kind} {result.latency_s:.1f}s\n"
            )
            sys.stdout.flush()
    save_checkpoint(output, results, run_metadata, complete=True, history=history)
    summaries = sorted(
        (summarize(model, tuple(row for row in results if row.model == model)) for model in models),
        key=lambda summary: (-summary["overall_success"], summary["p50_s"] or math.inf),
    )
    sys.stdout.write(json.dumps(summaries, indent=2, allow_nan=False) + "\n")
    sys.stdout.flush()


if __name__ == "__main__":
    main()
