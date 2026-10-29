# NVIDIA NIM fallback evaluation

`eval_fallback_chain.py` screens chat models through an OpenAI-compatible endpoint using eight deterministic tasks and two tool-result continuation tasks per trial. It checks exact output, arithmetic, JSON extraction, tool selection, code tracing, long-context retrieval and logic. Generated code is never executed

Use the existing container environment for authentication. Set `EVAL_API_KEY` or `NVIDIA_NIM_API_KEY` without printing its value

```bash
docker exec -i -e EVAL_REPEATS=3 -e EVAL_WORKERS=1 \
  -e EVAL_TIMEOUT=300 -e EVAL_MAX_TOKENS=16384 \
  -e EVAL_OUT=/tmp/fallback_eval_results.json <litellm-container> \
  python - < eval_fallback_chain.py
```

`EVAL_MODELS` selects comma-separated API model IDs. `EVAL_BASE_URL` defaults to `https://integrate.api.nvidia.com/v1`. Use the proxy URL and group names to measure routing, but a group can serve a different underlying model through fallback. Each result therefore records the returned model name, which does not necessarily identify an individual deployment

`EVAL_PROFILES_JSON` maps model IDs to generation overrides, such as `{"candidate":{"chat_template_kwargs":{"enable_thinking":true}}}`. Supported fields are `temperature`, `top_p`, `seed`, `max_tokens`, `max_completion_tokens`, `reasoning_effort`, `chat_template_kwargs` and `extra_body`. For direct NVIDIA calls, put native fields such as `chat_template_kwargs` at the top level. Record profiles alongside results and compare only settings that could actually be deployed. `EVAL_KEY_ROUTE` records a non-secret label for the key used. Repeat the run separately for each key

Every completed request is saved atomically. An interrupted file has `complete: false`; keep it as partial evidence. The output contains trial numbers, finish reasons, returned model names, prompt/completion/reasoning token counts, latency and classified errors. HTTP response bodies are not logged. A truncated response cannot pass. Tool continuations preserve canonical `reasoning_content` and remove the legacy duplicate `reasoning` field

`overall_success` divides passes by all evaluated tasks, including dependent continuations that could not run. `availability` measures completed, usable responses among attempted requests. `conditional_accuracy` measures grading accuracy among those usable responses. A wrong answer counts as an available response, while protocol failures, rejected completions and truncation do not. Separate error counts show timeouts, rate limits and dependency failures. Successful-response latency and all-attempt p90 latency are reported separately. Missing measurements are JSON `null`

Default concurrency is three requests per model. Use one worker for initial quality comparisons, then separately test production concurrency and quota behavior. Three repeats of ten fixed tasks are a small screening sample. They do not measure repository-level implementation quality, establish independent quotas, or prove ordered proxy failover. The draft YAML is a proposal, not a tested deployment or promotion recommendation

Run the local regression suite without API access

```bash
cd cookbook/nvidia_nim_fallback_eval
python3 -m unittest -v test_eval_fallback_chain.py
```
