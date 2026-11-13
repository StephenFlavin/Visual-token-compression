"""Context compaction with a page-image archive (user request, 2026-11-11).

When Claude Code compacts a long session, the summary keeps the gist and drops details. This tests keeping the full
pre-compaction transcript as an archive the model can open with Read after compaction, as page images (2.9x) or as
text, against normal compaction (summary only) and no compaction.

Per corpus (the 3 deep-research corpora of claude_qa_research.py: 34-50 real Wikipedia articles with invented facts,
120 exact-answer questions):

1. **Reading** (shared by all arms): a headless Claude Code session reads every source file with Read (low effort).
2. **D: no compaction:** a fork of that session answers the questions.
3. **Compaction:** `/compact` on the original session (one shared summary).
4. Forks of the compacted session answer the same questions, with Read only (the source files are gone):
   * **A: normal compaction:** the summary only;
   * **B: + page archive:** the transcript's sources as 88.2 px²/char pages (`archive_pages/page_NN.png`) and an index;
   * **C: + text archive:** the same chunks as text files (`archive_text/part_NN.txt`) and the same index.
   In practice the pointer to the archive comes from a SessionStart hook with matcher `compact` after a PreCompact hook
   renders the transcript (both verified headless); here it is put in the question prompt so every arm forks from the
   same compacted session.

Metrics: accuracy (all, lookups, synthesis) and the list-price cost of answering after compaction; shared costs
(reading, compaction) reported separately.

    python3 experiments/compaction/run_compact.py --corpora 0 1 2
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import shutil
import subprocess
import sys
import textwrap
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "experiments" / "claude_qa"), str(ROOT / "experiments" / "readhook")]
import claude_qa_research as R  # noqa: E402
import claude_qa_render as Q  # noqa: E402

WORK = Path("/tmp/claude-0/-home-user-visual-token-compaction/9e274458-7352-5bb1-b191-52e96364b3f6/scratchpad/compaction")
OUT = ROOT / "results" / "compaction"
MODEL = "claude-opus-5-5"
BUDGET = 88.2
PART_CHARS = 50_000        # Read refuses files above ~25k tokens


def claude(args: list[str], prompt: str, cwd: Path, tools: str, effort: str, timeout: int = 3600) -> dict:
    from vtc import claude_eval as CE
    cmd = [CE.CLAUDE_BIN, "-p", "--output-format", "stream-json", "--verbose", "--model", MODEL, "--effort", effort,
           "--tools", tools, "--allowedTools", tools or "Read", "--setting-sources", "project,local",
           "--strict-mcp-config", "--max-turns", "300", *args, prompt]
    t0 = time.time()
    p = subprocess.run(cmd, capture_output=True, text=True, env=CE.build_env(), cwd=cwd, timeout=timeout,
                       stdin=subprocess.DEVNULL)
    ev = [json.loads(l) for l in p.stdout.splitlines() if l.strip().startswith("{")]
    res = next((e for e in ev if e.get("type") == "result"), {})
    reads = [c for e in ev if e.get("type") == "assistant" for c in e["message"]["content"]
             if c.get("type") == "tool_use" and c["name"] == "Read"]
    # res["usage"] covers this invocation only; total_cost_usd and modelUsage of a resumed or forked session include
    # the whole session history, so the cost of this step is computed from usage (Opus 5.5 list prices, fitted exactly
    # from the CLI's own costs: input $4, output $20, cache write $5, cache read $0.20 per million).
    uu = res.get("usage") or {}
    u = {"outputTokens": uu.get("output_tokens", 0), "cacheReadInputTokens": uu.get("cache_read_input_tokens", 0),
         "cacheCreationInputTokens": uu.get("cache_creation_input_tokens", 0), "inputTokens": uu.get("input_tokens", 0)}
    step_cost = (u["inputTokens"] * 4 + u["outputTokens"] * 20 + u["cacheCreationInputTokens"] * 5
                 + u["cacheReadInputTokens"] * 0.2) / 1e6
    return {"session_id": res.get("session_id"), "result": res.get("result", ""), "cost": round(step_cost, 4),
            "cli_session_total": res.get("total_cost_usd"),
            "turns": res.get("num_turns"), "is_error": res.get("is_error"), "wall_s": round(time.time() - t0),
            "reads": len(reads), "read_paths": [r["input"].get("file_path", "") for r in reads],
            "output_tokens": u.get("outputTokens", 0), "cache_read": u.get("cacheReadInputTokens", 0),
            "cache_write": u.get("cacheCreationInputTokens", 0), "uncached": u.get("inputTokens", 0),
            "stderr": p.stderr[-500:], "stream": p.stdout}


def write_sources(srcs_txt: list[str], d: Path) -> list[str]:
    """One file per source (big ones in parts), lines wrapped at ~1000 chars for Read."""
    d.mkdir(parents=True, exist_ok=True)
    names = []
    for s, txt in enumerate(srcs_txt, 1):
        parts = [txt[i:i + PART_CHARS] for i in range(0, len(txt), PART_CHARS)]
        for j, part in enumerate(parts, 1):
            name = f"source_{s:02d}.txt" if len(parts) == 1 else f"source_{s:02d}_part{j}.txt"
            (d / name).write_text("\n".join(textwrap.wrap(part, 1000, break_long_words=False, break_on_hyphens=False)) + "\n")
            names.append(name)
    return names


def transcript(sid: str) -> list[dict]:
    p = glob.glob(os.path.expanduser(f"~/.claude/projects/*/{sid}.jsonl"))
    return [json.loads(l) for l in open(p[0]) if l.strip()] if p else []


def archive_flow(tr: list[dict]) -> list[str]:
    """The sources as the reading session received them (Read results in order), each under a `▶ <file>` header,
    whitespace collapsed (prose), as one flow per Read."""
    calls, out = {}, []
    for e in tr:
        m = e.get("message") or {}
        if not isinstance(m.get("content"), list):
            continue
        for c in m["content"]:
            if c.get("type") == "tool_use" and c.get("name") == "Read":
                calls[c["id"]] = os.path.basename(c["input"].get("file_path", ""))
            if c.get("type") == "tool_result" and c.get("tool_use_id") in calls:
                body = c.get("content")
                text = body if isinstance(body, str) else " ".join(x.get("text", "") for x in body or [] if isinstance(x, dict))
                text = re.sub(r"(?m)^\s*\d+\t", "", text)
                text = re.sub(r"<system-reminder>.*?</system-reminder>", "", text, flags=re.S)
                out.append(f"▶ {calls[c['tool_use_id']]} " + " ".join(text.split()))
    return out


def build_archive(flows: list[str], d: Path) -> dict:
    """Pages at BUDGET (d/archive_pages) and text parts with the same chunk boundaries (d/archive_text); one index."""
    from vtchook import render
    chunks = render.chunk(" ".join(flows), render.page_chars(BUDGET))
    for sub in ("archive_pages", "archive_text"):
        (d / sub).mkdir(parents=True, exist_ok=True)
    lines, tok_img = [], 0
    for k, c in enumerate(chunks, 1):
        pg = render.render(c, BUDGET)[0]
        pg.image.save(d / "archive_pages" / f"page_{k:02d}.png")
        (d / "archive_text" / f"part_{k:02d}.txt").write_text("\n".join(textwrap.wrap(c, 1000, break_long_words=False)) + "\n")
        tok_img += pg.tokens
        files = [f.replace("source_", "source ") for f in re.findall(r"▶ (source_\d+(?:_part\d+)?)\.txt", c)]
        parts = ([] if c.startswith("▶") else ["continues the previous source"]) + (["starts " + ", ".join(files)] if files else [])
        lines.append(f"{k:02d}: " + "; ".join(parts))
    index = ("Index of the archive: what each numbered file holds. Each source begins with ▶ <file name>; text before the "
             "first ▶ in a file continues the previous file's last source.\n\n" + "\n".join(lines) + "\n")
    for sub in ("archive_pages", "archive_text"):
        (d / sub / "INDEX.txt").write_text(index)
    return {"files": len(chunks), "image_tokens": tok_img, "chars": sum(map(len, chunks))}


READ_PROMPT = """You are a research assistant collecting sources for a report. Read each of these files in full with the
Read tool, one after another, in this order:
{files}
After each file, note in one short line what it covers. When you have read all of them, reply DONE."""

Q_PROMPT = """Answer the questions below about the sources you read earlier in this session. Answer every question with
the exact value only (a number), one line per question in the form 'ID: answer'. If you cannot find the answer, give
your best guess.{archive}

Questions:
{questions}"""

ARCHIVE = {"B": ("\n\nThe complete text of every source you read before this conversation was compacted is archived as "
                 "page images in {dir}/ (page_01.png ... page_{n:02d}.png); {dir}/INDEX.txt says which sources each page "
                 "holds. Open pages with the Read tool to look things up."),
           "C": ("\n\nThe complete text of every source you read before this conversation was compacted is archived as "
                 "text files in {dir}/ (part_01.txt ... part_{n:02d}.txt); {dir}/INDEX.txt says which sources each file "
                 "holds. Open files with the Read tool to look things up.")}


def score(qs, text: str) -> dict:
    ans = R.parse(text)
    per = [{"id": q["id"], "kind": q["kind"], "answer": q["answer"], "given": ans.get(q["id"]),
            "correct": ans.get(q["id"]) is not None and Q.norm_num(ans[q["id"]]) == q["answer"]} for q in qs]
    acc = lambda ks: sum(p["correct"] for p in per if p["kind"] in ks) / max(1, sum(p["kind"] in ks for p in per))  # noqa: E731
    return {"acc_all": sum(p["correct"] for p in per) / len(per), "acc_lookup": acc(R.LOOKUP), "acc_synth": acc(R.SYNTH),
            "answered": sum(p["given"] is not None for p in per), "per_question": per}


def run_corpus(k: int, srcs, a) -> list[dict]:
    d = WORK / f"corpus{k}"
    if d.exists():
        shutil.rmtree(d)
    proj = d / "proj"
    doc, src_txt, qs = R.build(srcs, seed=k)
    names = write_sources(src_txt, proj / "sources")
    rows = []

    def rec(arm, r, extra=None):
        row = {"corpus": k, "arm": arm, **{x: r[x] for x in ("cost", "turns", "wall_s", "reads", "output_tokens",
                                                              "cache_read", "cache_write", "uncached", "is_error")},
               **(extra or {})}
        (d / f"stream_{arm}.jsonl").write_text(r["stream"])
        rows.append(row)
        with open(OUT / "runs.jsonl", "a") as f:
            f.write(json.dumps(row) + "\n")
        print(f"[compact] corpus {k} {arm}: " + ", ".join(f"{x}={row[x]}" for x in row if x not in ("per_question",)
                                                       and not x.startswith("corpus")), flush=True)

    # 1. reading
    r = claude([], READ_PROMPT.format(files="\n".join(f"sources/{n}" for n in names)), proj, "Read", "low", timeout=7200)
    sid = r["session_id"]
    read_ok = sorted({os.path.basename(p) for p in r["read_paths"]}) == sorted(names)
    rec("read", r, {"session_id": sid, "files": len(names), "all_read": read_ok})
    tr = transcript(sid)
    info = build_archive(archive_flow(tr), d / "store")
    (d / "archive_info.json").write_text(json.dumps(info))
    shutil.rmtree(proj / "sources")                     # after reading, only the transcript (and archives) remain
    qtext = "\n".join(f"{q['id']}: {q['question']}" for q in qs)
    # 2. no compaction (fork before compacting)
    r = claude(["--resume", sid, "--fork-session"], Q_PROMPT.format(archive="", questions=qtext), proj, "Read", a.effort)
    rec("D_no_compaction", r, score(qs, r["result"]))
    # 3. compaction
    r = claude(["--resume", sid], "/compact", proj, "", a.effort)
    rec("compaction", r)
    # 4. after compaction
    for arm, name in (("A", "A_normal_compaction"), ("B", "B_page_archive"), ("C", "C_text_archive")):
        sub = {"B": "archive_pages", "C": "archive_text"}.get(arm)
        if sub:                                          # only this arm's archive is in the project
            shutil.copytree(d / "store" / sub, proj / sub)
        arch = ARCHIVE[arm].format(dir=sub, n=info["files"]) if sub else ""
        r = claude(["--resume", sid, "--fork-session"], Q_PROMPT.format(archive=arch, questions=qtext), proj, "Read", a.effort)
        rec(name, r, {**score(qs, r["result"]), "archive_reads": sum("archive" in p for p in r["read_paths"])})
        if sub:
            shutil.rmtree(proj / sub)
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpora", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--effort", default="high")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    corp = R.split_corpora(R.articles())
    for k in a.corpora:
        run_corpus(k, corp[k], a)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
