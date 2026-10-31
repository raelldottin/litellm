# NVIDIA NIM fallback evaluation

[fallback_chain.yaml](fallback_chain.yaml) defines ten models behind the `Hermes Kanban` alias

| Position | Model | Native controls | Timeout per attempt |
| --- | --- | --- | --- |
| 1 | `nvidia/nemotron-3-ultra-550b-a55b` | Thinking enabled, medium effort | 300 seconds |
| 2 | `nvidia/nemotron-3.5-super-vl-120b-a12b` | Thinking enabled | 300 seconds |
| 3 | `minimaxai/minimax-m3.1-flash` | Medium native effort | 300 seconds |
| 4 | `meta/muse-glimmer-30b` | Medium reasoning strength | 300 seconds |
| 5 | `openai/gpt-oss-20b` | Low effort, temperature zero | 300 seconds |
| 6 | `google/diffusiongemma-26b-a4b-it` | Thinking enabled | 300 seconds |
| 7 | `nvidia/nemotron-3-nano-omni-30b-a3b-reasoning` | Reasoning budget 4096 | 300 seconds |
| 8 | `mistralai/mistral-large-4-1t-a52b` | Low effort | 60 seconds |
| 9 | `nvidia/nemotron-3.5-lightning-30b-a3b` | Thinking enabled, reasoning budget 4096 | 300 seconds |
| 10 | `Beam-1.0-501B-A23B` | Medium effort on Reflection | 300 seconds |

Each NVIDIA model tries the primary key and then the secondary key before advancing. Beam uses the existing Reflection credential. Nineteen physical routes require eighteen fallback transitions. Explicit groups establish credential order. Two credentials do not establish independent NVIDIA quotas

Supply `NVIDIA_NIM_API_KEY`, `NVIDIA_NIM_API_KEY_SECONDARY` and `REFLECTION_API_KEY` through the deployment environment. Choose a single configuration source for these model records, either the YAML example or your database workflow

The entry alias uses local heuristic classification, with every tier pointing to Ultra's first route. Five additional local strategy records share their physical groups' names and pin GPT-OSS low, Mistral low and Beam medium. They make no classifier model requests and are excluded from physical deployment selection

## Reasoning controls and request overrides

LiteLLM 1.105.1 intentionally removes a deployment's `extra_body.reasoning_effort` when a request supplies its own effort. A direct SDK request can pass a wire check while a Router request sends the client's unsupported `ultra` value

The same-name strategy records use `tier_model_configs` to pin the tested effort before the Router selects the physical deployment. Keep these records alongside their physical routes

Routes with top-level native effort allow `reasoning_effort` and drop only `max_completion_tokens`. Strategy pins replace incoming effort. Other routes remove generic effort and supply their native template or reasoning-budget controls. Every physical route caps output at 16384 tokens

NVIDIA's adapter also requires `allowed_openai_params` to retain SDK `max_retries: 0`. Without it, the tested deployment made three upstream attempts on one rate-limited route. Router `RateLimitErrorRetries` and `TimeoutErrorRetries` are both zero

For a database-managed gateway, set `ROUTER_MAX_FALLBACKS=18` at startup as well as the saved router limit. Recreate the gateway with the intended image and preserved database and encryption settings. A saved settings response alone does not establish the running fallback limit

Muse uses a 131072-token context. GPT-OSS uses the conservative NVIDIA-hosted limit of 128000. Their context errors skip subsequent short-context routes and go to the next larger model, DiffusionGemma

## Evaluation scope

These observations were collected on October 31, 2026, using the profiles above and both NVIDIA credentials. The original four NVIDIA models remain first. The added models provide more backup choices and have measured coding limitations

| Added model | Screening | Coding assertions passed / executed | Expected assertions | Completed code generations |
| --- | --- | --- | --- | --- |
| `openai/gpt-oss-20b` | 60/60 | 82/84 | 84 | 6/6 |
| `google/diffusiongemma-26b-a4b-it` | 19/20 | 81/84 | 84 | 6/6 |
| `nvidia/nemotron-3-nano-omni-30b-a3b-reasoning` | 60/60 | 78/84 | 84 | 6/6 |
| `mistralai/mistral-large-4-1t-a52b` | 20/20 | 55/56 | 84 | 4/6 |
| `nvidia/nemotron-3.5-lightning-30b-a3b` | 20/20 | 37/38 | 84 | 6/6 |

GPT-OSS and Omni received three screening trials per task and credential. DiffusionGemma, Lightning and the selected Mistral low profile received one trial per task and credential. Screening includes actual tool-result continuations

The separate coding benchmark used project fixtures for normalization, cancellation/concurrency and CI evidence validation, with 42 expected assertions per credential. The bundled screening script does not execute generated code or include those project fixtures. Unavailable generations reduce the expected-total score and are excluded from the executed-assertion denominator

Mistral's two concurrency generations timed out at 300 seconds. Its served routes use a shorter timeout so the gateway advances sooner. Lightning returned all six generations, but three contained syntax errors and one failed cancellation cleanup. Its executed-assertion denominator excludes the three modules that could not load; the expected denominator retains all 84 assertions. DiffusionGemma missed one retrieval check. These results describe the tested endpoint and profile, rather than a universal ranking

Step initially passed 60/60 screening checks but completed only two of six coding generations. Before activation, its simple live proxy requests timed out at 60 seconds on both credentials in two rounds. The exact captured request also timed out directly against NVIDIA at 75 seconds on both credentials. Lightning's tested thinking profile replaced Step as a late backup. Lightning's separate nonthinking profile scored 34/60 screening checks and was not selected

Mistral Large 4 accepted explicit `low`, while explicit `none` returned HTTP 400 despite an older cached hosted recipe. Omni's hosted schema uses top-level `reasoning_budget`, rather than the `thinking_token_budget` field shown in a self-hosted example. Step's hosted schema does not document a reasoning-off switch

DiffusionGemma's model card recommends final-only historical responses. Test inherited assistant reasoning and tool history through the actual proxy before activation, in addition to each model's own tool continuation. Preserve canonical `reasoning_content` during this check so an incompatible route cannot pass by silently changing the history

The [Mistral hosted reference](https://docs.api.nvidia.com/nim/reference/mistralai-mistral-large-4-1t-a52b-infer), [Omni hosted reference](https://docs.api.nvidia.com/nim/reference/nvidia-nemotron-3-nano-omni-30b-a3b-reasoning-infer), [Step hosted reference](https://docs.api.nvidia.com/nim/reference/stepfun-ai-step-5-preview-infer), and [GPT-OSS model card](https://build.nvidia.com/openai/gpt-oss-20b/modelcard) document the relevant endpoint controls and limits

Before activation, verify physical deployment IDs through the live proxy, replay inherited reasoning/tool history, and force direct-entry and alias-entry rate-limit cascades into the real terminal deployment. Remove all temporary models and strategy records after the check. An immediate rate-limit test does not establish a short deadline for every slow timeout cascade. Streams cannot transparently switch models after content has already been delivered

The selected configuration passed 49/49 live route checks, including both-key tool continuations and inherited reasoning history for all five additions. Direct-entry and alias-entry tests each observed eighteen single rate-limited requests followed by the real Beam deployment. Both client keys passed the activated alias, followed by three turns in one Hermes session and a fresh worker-profile session. These are dated runtime observations; endpoint availability can change

## Run the screening evaluator

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

`EVAL_PROFILES_JSON` maps API model IDs to generation overrides. Match the native controls in the configuration when qualifying settings. Supported fields include sampling controls, token limits, `reasoning_effort`, `reasoning_budget`, `chat_template_kwargs` and `extra_body`. Direct NVIDIA requests put native `chat_template_kwargs` at the top level. Evaluate each key separately with a different output path and non-secret `EVAL_KEY_ROUTE` label

Every completed request is saved atomically. Interrupted checkpoints have `complete: false`. Results retain trial numbers, finish reasons, returned model names, token counts, latency and error categories. Truncated responses cannot pass. Continuations preserve canonical `reasoning_content` and remove duplicate legacy `reasoning`. HTTP error bodies and credentials are excluded

`EVAL_MIN_REQUEST_INTERVAL` spaces requests across workers, models and continuations in one process. It does not coordinate separate processes. Start with one worker for quality comparisons, then test production concurrency separately. Defaults remain three workers and three repeats

Use `EVAL_RESUME=1` with an existing matching checkpoint to retry missing or failed tasks. Models, repeats, native profiles, key label, budget and timeout must match. Worker count and pacing may change. Successful tasks are retained. An incomplete continuation refreshes its initial tool call to recover assistant context. Superseded attempts remain in `history`, preserving earlier failures

`overall_success` includes all tasks and dependent continuations that could not run. `availability` measures usable responses among actual attempts. `conditional_accuracy` measures correctness among those usable responses. Error counts separate rate limits, timeouts, malformed responses, truncation and dependency failures. Read recovered scores alongside their history. Missing measurements use JSON `null`

Run the local regression suite without an API key

```bash
cd cookbook/nvidia_nim_fallback_eval
python3 -m unittest -v test_eval_fallback_chain.py
```
