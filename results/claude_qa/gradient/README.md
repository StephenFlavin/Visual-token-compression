# Graded resolution: long-document Q&A with recent text sharp and old text dense

*Pre-registered 2026-11-13, written and committed before the first scored call. The results section is added
below it after the calls; nothing above "Results" changes after that commit.*

## Question

The DeepSeek-OCR paper proposes, without testing it, a context memory that forgets: older context is re-rendered at
ever lower resolution, so recent text is sharp and expensive and old text is blurry and cheap (noted in
[docs/gpu/COLOUR_GIST.md](../../../docs/gpu/COLOUR_GIST.md), "Prior art"; reference 1 in
[docs/background/prior-art.md](../../../docs/background/prior-art.md)). Does **non-uniform resolution for text**
work on one application that worked uniformly: Opus 5.5 answering questions about a long document given as page
images ([docs/claude/CLAUDE_QA.md](../../../docs/claude/CLAUDE_QA.md), Test 1)? Concretely: does a third of a
document read at density d, inside a document whose other thirds are at other densities, score like the whole
document read uniformly at density d? And does it matter whether the sharp part is the recent or the old part?

## Arms (new)

The document is cut into three equal parts by characters (cuts moved to the next space, and past any fact a
question needs, so no fact is split across two densities). Each third is rendered with the existing
`render_pages` at its own density on 1932-px pages; every page holds text of one third only; pages stay in
document order.

| arm | first (oldest) third | middle third | last (most recent) third |
|---|---|---|---|
| `recent_sharp` | 58.8 px²/char | 88.2 | 117.6 |
| `old_sharp` (control: separates position from density) | 117.6 | 88.2 | 58.8 |

## Held fixed (identical to the v2 uniform arms)

The 3 documents (v2 `qav2_d0/d1/d2`: 700k characters, paragraph offsets 0, 1150, 2480; seeds 0, 1, 2), the 120
questions per document (40 invented facts, 40 numbers in sentences, 40 multi-step; ids, wording and answers checked
equal to `qav2_d*_high_b88/questions.json`), the system prompt and the prompt (`claude_qa_render.SYSTEM`,
`prompt_for("image", ...)` with the total page count), the font and renderer, 1932-px pages, Opus 5.5
(`claude-opus-5-5`), `--effort high`, headless `claude -p` through `vbpe/claude_eval.py`.

**Dry run (free, `check`):** each mixed arm is 18 pages per document (4 + 6 + 8 or 8 + 6 + 4), the same page
count as uniform 88.2, and its image-token formula total is 79,887 / 79,749 / 79,818 for d0 / d1 / d2 against
uniform 88.2's 79,887 / 79,749 / 79,749 (ratio 1.000, 1.000, 1.001). Image tokens per third for `recent_sharp`
d0: 17,745 / 26,652 / 35,490. Measured px²/char per third: 58.9-59.0, 88.3-88.5, 117.7-118.0. Lookups (needle and
number questions) per third, pooled over the 3 documents: 79 / 84 / 77. So both mixed arms cost the same tokens as
uniform 88.2 (2.9x fewer than text) by construction; the measured input tokens are reported per call.

Baselines (existing rows, not re-run, one call per document, paired by question): uniform 117.6, 88.2 and 58.8
(`qav2_d{0,1,2}_high_b{118,88,59}`, image rows) and text (the text row of the same directories).

## Calls

One call per document and arm (6 calls, repeat 0, no cache salt), then, if the cap allows, a second call per
document and arm with `cache_salt="gradient1"` (12 calls in all), so call-to-call variation is visible. Each call's
pages must arrive: input tokens >= 0.97 x the image-token formula, else the call is marked incomplete and excluded.
Calls that refuse, leave questions unanswered or trip the existing engagement check are excluded and listed.
Refusals are reported, never worked around.

## Scores and tests (existing scoring code: `norm_num` exact match, `analyse_claude_qa.wilson`, `paired_diff_ci`, `mcnemar`)

1. **Overall accuracy** of each mixed arm against text, uniform 88.2 (equal tokens), uniform 117.6 and uniform
   58.8: paired by question, Newcombe 95% interval, exact McNemar. Primary: the first call of each document
   (360 pairs). Secondary: every valid call, each paired with the same baseline row (720 pairs; the interval then
   ignores that the baseline is one call, so it is too narrow and only descriptive).
2. **By tier:** lookups (needle and number questions) by the third their fact sits in, for the two mixed arms and
   for the three uniform arms on the same questions, with Wilson intervals.
3. **Multi-step questions by the tiers their facts touch:** the rows' `anchors` are every fact a question needs
   (from `make_complex_v2`), so each question's tiers are where its anchors sit in the document. Split per mixed
   arm: "dense" = at least one fact in the third that arm draws at 58.8; "sharp" = all facts in its 88.2 / 117.6
   thirds. Every arm (incl. uniform ones) is scored on the same split. Counts touch every third almost always, so
   they fall in "dense". From the dry run: 24 sharp and 96 dense multi-step questions per split, pooled.
4. **Cost and input tokens per call** (`total_cost_usd`, list price).

Point estimates for N1-N4 pool every valid call of an arm (mean over calls).

## Predictions (from the brief; not to be changed)

- **N1:** each tier inside a mixed document scores within 3 points of the uniform arm at the same density on the
  same lookups (no penalty for mixing).
- **N2:** `recent_sharp` and `old_sharp` have the same overall accuracy within 3 points (density matters, position
  does not).
- **N3:** overall, both mixed arms land between uniform 88.2 and uniform 58.8, i.e. below the equal-token uniform
  arm: grading resolution is a trade (accuracy moved to where it is wanted), not a free gain. (Judged as: uniform
  58.8 <= mixed < uniform 88.2, for both arms.)
- **N4:** multi-step questions with a fact in the 58.8 tier lose at least 10 points against the same questions in
  the uniform 117.6 arm; those with all facts in sharper tiers do not. (Judged per mixed arm on its own split:
  dense loss >= 10 points and sharp loss < 10 points, for both arms.)

Mine (the runner's, labelled as such):

- **R1:** the dense-tier multi-step loss in each mixed arm is smaller than uniform 58.8's on the same questions
  (uniform 58.8: 76.0% / 74.0% on the two dense splits), because only part of each chain is dense.
- **R2:** the lookup tiers barely separate: uniform 58.8 already gets 97-99% of lookups in every third, so N1 holds
  mostly because lookups are near the ceiling, not because mixing is shown to be free.

## What this sample can and cannot show

With ~80 lookups per third per call and lookups at 97-100% at every density, a third's accuracy is 2 questions
from a 2.5-point difference: N1's 3-point margin is inside call-to-call noise and the Wilson interval of 80/80 is
95.4-100%. Only a penalty of roughly 8+ points in one third would stand out. Overall, 360 paired questions give a
Newcombe interval of about +-2-3 points near 92-100% accuracy, so differences of ~3 points between arms are at the
edge of what is detectable. The "sharp" multi-step group is small (24 questions per split): a 10-point loss there
is 2-3 questions. The test can show a large tier effect (a mixed third scoring like a different density, or the
dense third dragging chains down by 10-20 points); it cannot show small (<5 point) effects.

## Spend

Hard cap **$12 at list price** (`total_cost_usd`), tracked in `spend.jsonl`. A call starts only if spent + calls in
flight + this call's upper estimate (max of $1.00 and 1.3x the dearest call so far) stays within the cap. Calls
draw on the user's subscription quota; on a usage-limit error the run stops, rows are committed and what remains
is reported.

## Reproduce

```bash
python3 experiments/claude_qa/claude_qa_gradient.py check            # free: pages, tokens per tier
python3 experiments/claude_qa/claude_qa_gradient.py run --yes        # resumable; rows appended per call
python3 experiments/claude_qa/claude_qa_gradient.py analyse          # summary.md / summary.json from whatever rows exist
```

Files: `runs.jsonl` (one row per call, per-question outcomes with each fact's tier), `spend.jsonl`, `dry_run.json`,
`question_tiers.json`, `summary.md`, `summary.json`.

## Results

*(added after the calls)*
