# NVIDIA NIM fallback evaluation

[fallback_chain.yaml](fallback_chain.yaml) contains five distinct models in this order

| Position | Model | Native reasoning control |
| --- | --- | --- |
| 1 | `nvidia/nemotron-3-ultra-550b-a55b` | Thinking with medium effort |
| 2 | `nvidia/nemotron-3.5-super-vl-120b-a12b` | Thinking enabled |
| 3 | `minimaxai/minimax-m3.1-flash` | Medium effort |
| 4 | `meta/muse-glimmer-30b` | Medium reasoning strength |
| 5 | `Beam-1.0-501B-A23B` | Medium effort on Reflection |

Each NVIDIA model tries the primary key, then the secondary key, before moving to the next model. Beam uses the third credential on a separate provider. These nine physical routes require eight fallback transitions. Explicit model groups establish key order; a shared deployment pool does not

The `Hermes Kanban` alias uses local heuristic routing with every tier pointing at Ultra's first route. It makes no classifier model request. Muse has a 131,072-token context, so its context errors skip its second key and go directly to Beam

Supply `NVIDIA_NIM_API_KEY`, `NVIDIA_NIM_API_KEY_SECONDARY` and `REFLECTION_API_KEY` through the deployment environment. Two NVIDIA keys do not establish independent quotas

On the tested LiteLLM 1.105.1 deployment, NVIDIA dropped `max_retries` unless `allowed_openai_params: [max_retries]` was present. `max_retries: 0` alone still produced three upstream requests for each 429. The example preserves that client setting and sets `RateLimitErrorRetries: 0` to avoid router retries on rate limits

Every route caps output at 16,384 tokens and supplies native reasoning controls. Incoming generic reasoning effort and `max_completion_tokens` are removed. Keep `max_tokens` out of `additional_drop_params` so the configured cap survives transformation

For database-managed routing, also set `ROUTER_MAX_FALLBACKS=8` at gateway startup and restart the gateway. A saved settings response alone did not establish the runtime limit in the tested deployment. Streaming failures after content has already been delivered cannot transparently switch models

Observed on October 29-30, 2026, Ultra, Super VL and Muse completed 60/60 screening tasks across both NVIDIA keys. Beam completed 30/30 on its own key. MiniMax answered completed tasks correctly but encountered rate limits even with pacing, placing it after the Nemotron routes. Additional coding contracts and real tool continuations qualified the selected settings; the included screening script alone does not establish repository-level implementation quality or a universal model ranking

During those tests, original GLM 5.3 returned HTTP 410 on both keys. GLM Flash, DeepSeek Flash and Gemma had endpoint timeouts, Kimi returned 404, and Step timed out. Lightning failed a coding assertion and truncated a concurrency implementation with a 16,384-token budget. Laguna encountered a 503 and a concurrency timeout. These are dated endpoint and profile observations

Live gateway validation forced eight ordinary HTTP 429 responses through isolated test routes, observed one request per route, and received a real Beam response with the expected deployment ID. Native request capture also checked all five profiles with three incoming token-budget variants

[Muse's API reference](https://docs.api.nvidia.com/nim/reference/meta-muse-glimmer-30b-infer) documents its native controls. Its [model card](https://build.nvidia.com/meta/muse-glimmer-30b/modelcard) documents the context limit

`eval_fallback_chain.py` independently screens an OpenAI-compatible endpoint with eight deterministic tasks and two actual tool-result continuations per trial. Generated code is never executed. Candidate defaults support discovery and are separate from the selected chain

Set `EVAL_API_KEY` or `NVIDIA_NIM_API_KEY` in the evaluation process environment. The evaluator does not retrieve credentials from LiteLLM's database. This example assumes a container named `litellm` already has an evaluation key in its environment

```bash
docker exec -i -e EVAL_REPEATS=3 -e EVAL_WORKERS=1 \
  -e EVAL_MIN_REQUEST_INTERVAL=3 -e EVAL_TIMEOUT=300 \
  -e EVAL_MAX_TOKENS=16384 -e EVAL_KEY_ROUTE=primary \
  -e EVAL_MODELS=nvidia/nemotron-3-ultra-550b-a55b \
  -e EVAL_OUT=/tmp/fallback_eval_primary.json litellm \
  python - < eval_fallback_chain.py
```

`EVAL_BASE_URL` defaults to `https://integrate.api.nvidia.com/v1`. A proxy URL and group names measure routing, where a fallback may serve the request. The returned model name alone may not identify the physical deployment

`EVAL_PROFILES_JSON` maps API model IDs to generation overrides. Match the native controls in the configuration when qualifying settings. Supported fields include sampling controls, token limits, `reasoning_effort`, `chat_template_kwargs` and `extra_body`. Direct NVIDIA requests put native `chat_template_kwargs` at the top level. Evaluate each key separately with a different output path and non-secret `EVAL_KEY_ROUTE` label

Every completed request is saved atomically. Interrupted checkpoints have `complete: false`. Results retain trial numbers, finish reasons, returned model names, token counts, latency and error categories. Truncated responses cannot pass. Continuations preserve canonical `reasoning_content` and remove duplicate legacy `reasoning`. HTTP error bodies and credentials are excluded

`EVAL_MIN_REQUEST_INTERVAL` spaces requests across workers, models and continuations in one process. It does not coordinate separate processes. Start with one worker for quality comparisons, then test production concurrency separately. Defaults remain three workers and three repeats

Use `EVAL_RESUME=1` with an existing matching checkpoint to retry missing or failed tasks. Models, repeats, native profiles, key label, budget and timeout must match. Worker count and pacing may change. Successful tasks are retained. An incomplete continuation refreshes its initial tool call to recover assistant context. Superseded attempts remain in `history`, preserving earlier failures

`overall_success` includes all tasks and dependent continuations that could not run. `availability` measures usable responses among actual attempts. `conditional_accuracy` measures correctness among those usable responses. Error counts separate rate limits, timeouts, malformed responses, truncation and dependency failures. Read recovered scores alongside their history. Missing measurements use JSON `null`

Run the local regression suite without an API key

```bash
cd cookbook/nvidia_nim_fallback_eval
python3 -m unittest -v test_eval_fallback_chain.py
```
