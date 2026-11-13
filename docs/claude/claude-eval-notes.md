# Claude as a zero-shot page reader: client measurements

*Status: reference, measured 2026-11-02 before Phase 1 (work package "WP-E1" of the original plan,
[glyphs/PLAN.md](../glyphs/PLAN.md)); spot-checked against the JSON files in `results/claude_checks/` in the
2026-11-12 clean-up. It records how the headless Claude client is called, how images are charged, and where plain
text stops being legible. "Zero-shot" means the model is used as it is, with no training. Two things learnt later
are added in the box below; the rest is as measured.*

> **Learnt later (2026-11-10 to 12, see [GOTCHAS.md](../GOTCHAS.md) and [CLAUDE_QA.md](CLAUDE_QA.md)):**
> * Opus 5.5 reads images up to 1932 x 1932 px (4,764 tokens) without downscaling when the client is called with
>   `hires=True`; the limits in section 2 are those of the default path.
> * Legible is not the same as usable. At 29.4 px² per character a page transcribes at about 3% error, but questions
>   over a long document at that density fail; question answering needs about 117.6 px² per character.
> * One request with more than about 175k image tokens can silently lose its earliest pages.

**Terms:** CER = character error rate (edit distance divided by reference length); px² per character = image area
divided by characters shown (lower is denser); em = font size in pixels; effort = the command line's
`--effort` setting for how much the model reasons.

Model `claude-opus-5-5` (served model = requested in every call), Claude Code CLI 2.1.316.
Raw data: `results/claude_checks/{tokens,tokens_large,legibility,determinism,effort,hidden_calls,cost_model}.json`,
rendered test images in `results/claude_checks/img/`. Reproduce with `python scripts/check_claude_vision.py all`
(about $0.45).

## 1. Invocation (vtc/claude_eval.py)

```
cd <fresh empty temp dir>
env -i PATH HOME <proxy + CA vars> [ANTHROPIC_API_KEY ...] \
    CLAUDE_CODE_DISABLE_ADVISOR_TOOL=1 CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC=1 \
    DISABLE_AUTOUPDATER=1 FORCE_PROMPT_CACHING_5M=1 \
  claude -p --input-format stream-json --output-format stream-json --verbose \
    --model claude-opus-5-5 --system-prompt "<system>" --tools "" --max-turns 1 \
    --strict-mcp-config --setting-sources "" --disable-slash-commands --no-session-persistence \
    [--effort <level>]    < one stream-json user message: [image..., text prompt]
```

The final `type=="result"` event gives text, usage, modelUsage, total_cost_usd. We also record the last
`rate_limit_event` (subscription quota use).

**Why the clean environment matters (hidden calls).** Running `claude -p` from inside a Claude Code session
inherits the parent's variables (`CLAUDE_CODE_SESSION_ID`, `CLAUDE_CODE_REMOTE*`, `CLAUDE_EFFORT`, ...).
In that case the child:
* reuses the **parent's session id** and emits remote-session events (`post_turn_summary`),
* makes an auxiliary **claude-haiku-5-5** call (1 176 in / 13 out, $0.00012; the post-turn summary),
* adds about 830 prompt tokens (1 392 vs 563 fixed overhead) and writes them at the 1-hour cache rate (2x input).

With the whitelisted environment and the flags above, every call is one Opus request, and the fixed prompt
overhead is 563 tokens (CLI-injected scaffolding plus our system prompt). Over 25 recorded check calls,
`modelUsage` listed only `claude-opus-5-5` (`hidden_calls.json`). The advisor tool was not active for this org
anyway; we disable it for safety. Grayscale (`L`) PNGs are accepted, and they cost the same tokens as RGB.

**Caching.** By default the CLI writes the whole prompt to the 1-hour cache (2x input price). With
`FORCE_PROMPT_CACHING_5M=1` it is 1.25x. With `DISABLE_PROMPT_CACHING=1` it is 1.0x, but then shared prefixes
(legend/key images) are never reused. For unique images the difference is under $0.001 per call, so we keep 5m
caching, which helps `*_legend` runs.

**Billing note.** This runner authenticates by OAuth subscription. `total_cost_usd` is the list-price
equivalent (`costBasis: list`); in practice the calls consume the 5-hour and 7-day quota. Implied list prices:
$4/M input, $5/M 5-minute cache write, $20/M output.

## 2. Image token accounting

| image | W·H/750 | measured image tokens | ceil(W/28)·ceil(H/28) |
|---|---|---|---|
| 150x150 | 30 | 39 | 36 |
| 300x300 | 120 | 125 | 121 |
| 600x600 (L and RGB) | 480 | 488 / 487 | 484 |
| 500x1000 | 667 | 651 | 648 |
| 1000x1000 | 1333 | 1300 | 1296 |
| 1072x1072 | 1532 | 1523 | 1521 |
| 1568x733 | 1532 | 1515 | 1512 |
| 1400x1400 (over config limit) | 2613 | 2503 | 2500 |
| 2000x1000 (over) | 2667 | 2594 | 2592 |
| 1568x1568 (over) | 3278 | 3141 | 3136 |
| 2576x1456 | 5001 | 3001 | 4784, **downscaled** |
| 3200x1600 | 6827 | 2640 | 6670, **downscaled** |

Image tokens are measured as input tokens minus the input tokens of a text-only call.

**Observed formula:** `image_tokens = ceil(W/28) * ceil(H/28) + ~3`. This is 784 px²/token on 28-px patches,
so W·H/750 overestimates by about 4.5%. Small images pay for partial patches. Implications:
* For canvas sizes, use multiples of 28 px so that px² and tokens map 1:1. Otherwise report
  `ceil(W/28)*ceil(H/28)` as the token cost. At the 150 to 400 px canvases that matter here, rounding is up to 15%.
* No downscaling happens at or below 1568x1568 (2.46 MP). Downscaling does happen at 3.75 MP. The config limits
  (edge 1568, area 1.15 MP) are therefore conservative and safe. The real area limit lies between 2.46 and 3.75 MP
  (about 3 000 tokens).

## 3. Legibility sanity curve (1 411-char original English paragraph, square canvases, DejaVu Sans `latin` / Archivo Narrow `latin_condensed`)

| scheme | em px | canvas | px²/char | CER | notes |
|---|---|---|---|---|---|
| latin | 4.0 | 129² | 11.8 | 0.84 | refuses ("too small"), quotes fragments |
| latin | 5.0 | 158² | 17.7 | 0.37 (effort=low: 0.31) | first ~3 lines right, then confabulated |
| latin | 6.0 | 188² | 25.0 | 0.008, 0.040, 0.032, 0.011 (low: 0.019) | steep part of the curve, noisy |
| latin | 7.0 | 218² | 33.7 | 0.000 | |
| latin | 8.0 | 248² | 43.6 | 0.000 (4 of 4 runs) | |
| latin | 10 / 12 | 308² / 368² | 67 / 96 | 0.000 | |
| latin_condensed | 4.0 | 109² | 8.4 | 0.67 | fluent fabricated prose |
| latin_condensed | 5.0 | 135² | 12.9 | 0.66 | fabricated |
| latin_condensed | 6.0 | 160² | 18.1 | 0.47 (low: 0.67) | partly fabricated |
| latin_condensed | 7.0 | 185² | 24.3 | 0.008 | |
| latin_condensed | 8.0 | 210² | 31.3 | 0.001 | |
| latin_condensed | 10 / 12 | 261² / 311² | 48 / 69 | 0.000 | |

**Takeaways for the Phase-1 pilot**
* For both fonts the cliff is between about 18 and 25 px²/char. Above about 30 px²/char, CER is 0.
  The condensed font needs a larger em (7 vs 6) but reaches the same CER at slightly fewer px²/char, so its
  advantage is small. At 25 px²/char, Claude reads about 31 chars per vision token.
* The informative range is about 8 to 35 px²/char. A grid of {8, 11, 14, 17, 20, 23, 27, 32, 40} beats the
  planned {4 ... 60}: 4 is pointless, and 60 is saturated for Latin, though it may still be needed for subst/quad.
* Below the cliff, Claude does not degrade gracefully. It **confabulates fluent English** (CER 0.5 to 0.7, with
  outputs roughly the right length) or **refuses**. The prompt should forbid refusals and ask for a best guess.
  CER near 0.6 to 0.8 then means "illegible" whatever the cause. The language prior is strong (risk R7 of the original plan): the opening
  sentences are read right even when the rest is invented.
* With 3 000-char passages, the canvases will be about 210 to 330 px per side. That is small enough that 28-px
  patch rounding matters (see §2).

## 4. Latency, cost, concurrency

* Cost model fitted over 24 fresh calls: `cost ≈ $5.2/M input tokens + $20.5/M output tokens` (intercept ≈ 0).
  A 1 411-char transcription is 461 output tokens (about 3.06 chars/token) and costs **$0.013** per call.
  Input is minor: 563 fixed tokens plus image tokens, about $0.003.
* **Adaptive thinking** turns on for hard (near-illegible) images at the CLI default effort: 300 to 900 thinking
  tokens, which doubles cost to $0.024 to $0.030. With `--effort low` there was no thinking, the cost was
  $0.012 to $0.013, and CER was within run-to-run noise (3 cases). The default effort is whatever the CLI chooses,
  which is not pinned. **Recommend pinning `VBPE_EFFORT` (low, or the default, chosen in the pilot)** so that
  results are reproducible.
* Latency is about 2.2 s plus 14 ms per output token. Sequential calls with ~460 output tokens took 7 to 9 s;
  under 8-way concurrency they took 4 to 23 s. Process start-up is about 0.4 s.
* **Concurrency:** 14 calls with 8 workers finished in 28 s wall time, with 0 errors, 0 retries, and no
  rate-limit events other than `status: allowed`. Each CLI process uses about 200 MB RSS.
* Phase-1 estimate: 3 000-char passages are about 1 000 output tokens, so about $0.025 per call without thinking
  and up to about $0.04 with it. 360 calls ≈ $9 to $15 (list-equivalent).

## 5. Determinism

Each image was sent 3 times with the cache bypassed (the CLI exposes no temperature control):
* em 8 (43.6 px²/char): 3 identical outputs, CER 0.
* em 6 (25 px²/char, on the cliff): all 3 outputs differ. CER vs the reference was 0.040, 0.032 and 0.011, and
  pairwise CER between runs was 0.024 to 0.053.

In the saturated region outputs are deterministic. On the steep part of the curve, single-call CER has an SD of
about ±0.015 at a CER of about 0.02. Use more than one passage or more than one repeat per point there. The
response cache returns the first sample, so repeats need `use_cache=False` or distinct jobs.

## 6. Client API notes

* `transcribe(images, system, prompt, model=DEFAULT_MODEL, max_retries=5, *, timeout=300, use_cache=True,
  effort=VBPE_EFFORT, cache_dir=None)`. It raises `ValueError` if an image exceeds `CLAUDE_MAX_EDGE` or
  `CLAUDE_MAX_AREA`. The extra result keys are `usage`, `thinking_tokens`, `model_usage`, `requested_model`,
  `rate_limit`, `attempts`, `stop_reason` and `cache_key`.
* Retries use exponential backoff with jitter (2 s, 4 s, 8 s ... capped at 60 s, times 0.5 to 1.5). They cover a
  non-zero exit, a missing result event, an `is_error` result, and timeouts. A 4xx result other than 429 is not
  retried. Errors are never cached.
* `transcribe_many(jobs, workers=8, progress=True)` keeps results in input order and passes `meta` through. A
  job that raises (for example an oversize image) becomes an error record and does not stop the pool.
