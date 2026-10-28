# NVIDIA NIM fallback chain eval

`eval_fallback_chain.py` scores NVIDIA NIM chat models on eight auto-graded tasks (instruction following, arithmetic, JSON extraction, tool calling, tool selection, code tracing, ~35k token needle retrieval, logic), repeated `EVAL_REPEATS` times, and ranks them by pass rate, then median latency. It is stdlib only, so it runs inside the LiteLLM Docker image with the key that is already there

```bash
docker exec -i -e EVAL_REPEATS=3 <litellm-container> python - < eval_fallback_chain.py
```

Point `EVAL_BASE_URL` at `http://localhost:4000/v1` and `EVAL_MODELS` at proxy model group names to score the chain through the proxy instead of NVIDIA directly

`fallback_chain.draft.yaml` is the starting order taken from published numbers (GLM-5.3-Flash AA index 57, DeepSeek-V4.1-Flash DeepSWE 74 to 77, MiniMax-M3.1-Flash DeepSWE 73.7, Kimi K2.6 SWE-bench Verified 80.2 vendor reported, Nemotron 3 Ultra SWE-bench Verified 70.7, Nemotron 3.5 Lightning as the fast last resort). Gemma 4 31B and Laguna XS 2.1 start outside the chain because neither has published agentic scores at that level, but both are in the eval and get promoted if they beat a chain member
