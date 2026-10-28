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
"""

import json
import os
import re
import statistics
import sys
import time
import urllib.error
import urllib.request
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from typing import Final

DEFAULT_MODELS: Final = (
    "z-ai/glm-5.3-flash",
    "deepseek-ai/deepseek-v4.1-flash",
    "minimaxai/minimax-m3.1-flash",
    "moonshotai/kimi-k2.6",
    "stepfun-ai/step-5-preview",
    "nvidia/nemotron-3-ultra-550b-a55b",
    "nvidia/nemotron-3-super-120b-a12b",
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
    messages: tuple[dict[str, str], ...]
    grade: Callable[[dict[str, object]], bool]
    tools: tuple[dict[str, object], ...] = ()


def strip_reasoning(text: str) -> str:
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


def content_of(message: dict[str, object]) -> str:
    content = message.get("content")
    return strip_reasoning(content) if isinstance(content, str) else ""


def first_tool_call(message: dict[str, object]) -> tuple[str, dict[str, object]] | None:
    calls = message.get("tool_calls")
    if not isinstance(calls, list) or not calls:
        return None
    fn = calls[0].get("function", {})
    try:
        args = json.loads(fn.get("arguments") or "{}")
    except json.JSONDecodeError:
        return None
    return fn.get("name", ""), args if isinstance(args, dict) else {}


def parse_json_object(text: str) -> dict[str, object] | None:
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not match:
        return None
    try:
        parsed = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None


def user(text: str) -> tuple[dict[str, str], ...]:
    return ({"role": "user", "content": text},)


def grade_json(message: dict[str, object]) -> bool:
    obj = parse_json_object(content_of(message))
    return obj is not None and obj == {"name": "Maria Lopez", "age": 34, "city": "Lisbon"}


def grade_weather(message: dict[str, object]) -> bool:
    call = first_tool_call(message)
    return (
        call is not None
        and call[0] == "get_weather"
        and "paris" in str(call[1].get("city", "")).lower()
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
        lambda m: NEEDLE in content_of(m),
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


def call(base_url: str, api_key: str, model: str, task: Task) -> Result:
    body = {"model": model, "messages": list(task.messages), "temperature": 0, "max_tokens": 4096}
    if task.tools:
        body["tools"] = list(task.tools)
    request = urllib.request.Request(
        f"{base_url.rstrip('/')}/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
    )
    start = time.monotonic()
    try:
        with urllib.request.urlopen(request, timeout=180) as response:
            payload = json.load(response)
    except urllib.error.HTTPError as e:
        return Result(model, task.name, False, e.code, time.monotonic() - start, 0, e.read().decode()[:300])
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as e:
        return Result(model, task.name, False, 0, time.monotonic() - start, 0, str(e)[:300])
    elapsed = time.monotonic() - start
    message = payload["choices"][0]["message"]
    tokens = (payload.get("usage") or {}).get("completion_tokens") or 0
    passed = task.grade(message)
    detail = "" if passed else json.dumps({"content": message.get("content"), "tool_calls": message.get("tool_calls")})
    return Result(model, task.name, passed, 200, elapsed, tokens, detail[:300])


def run_model(base_url: str, api_key: str, model: str, repeats: int, workers: int) -> tuple[Result, ...]:
    jobs = tuple(task for task in TASKS for _ in range(repeats))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        return tuple(pool.map(lambda task: call(base_url, api_key, model, task), jobs))


def summarize(model: str, results: tuple[Result, ...]) -> dict[str, object]:
    ok = tuple(r for r in results if r.status == 200)
    latencies = sorted(r.latency_s for r in ok)
    return {
        "model": model,
        "score": sum(r.passed for r in results) / len(results),
        "availability": len(ok) / len(results),
        "p50_s": statistics.median(latencies) if latencies else float("nan"),
        "p90_s": latencies[int(0.9 * (len(latencies) - 1))] if latencies else float("nan"),
        "tok_per_s": sum(r.completion_tokens for r in ok) / max(sum(r.latency_s for r in ok), 1e-9),
        "failed_tasks": sorted({r.task for r in results if not r.passed}),
    }


def main() -> None:
    base_url: Final = os.environ.get("EVAL_BASE_URL", "https://integrate.api.nvidia.com/v1")
    api_key: Final = os.environ.get("EVAL_API_KEY") or os.environ.get("NVIDIA_NIM_API_KEY") or ""
    if not api_key:
        sys.exit("Set EVAL_API_KEY or NVIDIA_NIM_API_KEY")
    models: Final = tuple(m.strip() for m in os.environ.get("EVAL_MODELS", ",".join(DEFAULT_MODELS)).split(","))
    repeats: Final = int(os.environ.get("EVAL_REPEATS", "3"))
    workers: Final = int(os.environ.get("EVAL_WORKERS", "3"))

    per_model: Final = {m: run_model(base_url, api_key, m, repeats, workers) for m in models}
    summaries: Final = sorted(
        (summarize(m, rs) for m, rs in per_model.items()), key=lambda s: (-s["score"], s["p50_s"])
    )

    print(f"{'model':40} {'score':>6} {'avail':>6} {'p50 s':>7} {'p90 s':>7} {'tok/s':>7}  failed tasks")
    for s in summaries:
        print(
            f"{s['model']:40} {s['score']:6.0%} {s['availability']:6.0%} {s['p50_s']:7.1f} {s['p90_s']:7.1f} "
            f"{s['tok_per_s']:7.0f}  {','.join(s['failed_tasks'])}"
        )
    for r in (r for rs in per_model.values() for r in rs if not r.passed):
        print(f"FAIL {r.model} {r.task} status={r.status} {r.error}", file=sys.stderr)

    with open(os.environ.get("EVAL_OUT", "fallback_eval_results.json"), "w") as f:
        json.dump({"summaries": summaries, "results": [asdict(r) for rs in per_model.values() for r in rs]}, f)


if __name__ == "__main__":
    main()
