"""local_qa.py against a mock OpenAI-compatible server: no model, no network, no data/ needed."""
import json
import re
import sys
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments" / "claude_qa"))
import local_qa as LQ  # noqa: E402

DOC = " ".join(f"Sentence {i} of the test document has the value {i * 7} in it." for i in range(400))
QS = [{"id": f"N{i + 1:02d}", "kind": "needle" if i < 6 else "complex", "sub": None if i < 6 else "sum",
       "question": f"What is value {i}?", "answer": str(100 + i), "pos": i / 10} for i in range(10)]
GOLD = {q["id"]: q["answer"] for q in QS}


class Mock(BaseHTTPRequestHandler):
    log, prompt_tokens, wrong_when_images = [], 50_000, set()

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        parts = body["messages"][1]["content"]
        n_img = sum(p["type"] == "image_url" for p in parts)
        ids = re.findall(r"^(N\d\d):", parts[-1]["text"], re.M)
        Mock.log.append({"n_img": n_img, "ids": ids, "text": parts[-1]["text"], "body": body})
        ans = "\n".join(f"{i}: {'7' if (n_img and i in Mock.wrong_when_images) else GOLD[i]}" for i in ids)
        out = {"choices": [{"message": {"content": "<think>hmm N01: 999</think>\n" + ans}, "finish_reason": "stop"}],
               "usage": {"prompt_tokens": Mock.prompt_tokens, "completion_tokens": 40},
               "timings": {"prompt_n": 900, "cache_n": 0, "prompt_per_second": 1000.0, "predicted_n": 40,
                           "predicted_per_second": 30.0}}
        data = json.dumps(out).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *a):
        pass


@pytest.fixture()
def server(monkeypatch, tmp_path):
    srv = HTTPServer(("127.0.0.1", 0), Mock)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    Mock.log, Mock.prompt_tokens, Mock.wrong_when_images = [], 50_000, set()
    monkeypatch.setattr(LQ, "build", lambda d, chars, seed: (DOC, QS))
    monkeypatch.setattr(LQ.config, "RESULTS", tmp_path)
    yield f"http://127.0.0.1:{srv.server_address[1]}"
    srv.shutdown()


def test_pages_are_on_the_32_px_token_grid_and_hold_the_whole_document():
    pages, info = LQ.render_pages(DOC, 117.6, 640)
    assert len(pages) >= 2 and all(im.size[0] == 640 and im.size[1] % 32 == 0 for im in pages)
    assert all(p["vision_tokens"] == (p["W"] // 32) * (p["H"] // 32) for p in info)
    assert sum(p["chars"] for p in info) >= len(LQ.metrics.normalize(DOC)) - len(pages)
    dense = sum(p["vision_tokens"] for p in LQ.render_pages(DOC, 58.8, 640)[1])
    assert dense < 0.62 * sum(p["vision_tokens"] for p in info)


def test_document_prefix_is_identical_across_question_batches():
    a = LQ.user_content("text", DOC, [], QS[:5])[0]["text"]
    b = LQ.user_content("text", DOC, [], QS[5:])[0]["text"]
    cut = a.index("Questions:\n") + len("Questions:\n")
    assert a[:cut] == b[:cut] and a[cut:] != b[cut:]
    pa, pb = (LQ.user_content("page118", DOC, ["data:x", "data:y"], q) for q in (QS[:5], QS[5:]))
    assert pa[:2] == pb[:2] and pa[2]["text"].startswith("The document is the 2 page images above")
    assert "document" not in LQ.user_content("none", DOC, [], QS[:5])[0]["text"]


def test_run_resume_and_analyse(server):
    Mock.wrong_when_images = {"N07", "N08", "N09", "N10"}
    common = ["--run-name", "t", "--base-url", server]
    run = ["run", *common, "--page-px", "640", "--questions-per-call", "4", "--arms", "text", "page118",
           "--extra", '{"chat_template_kwargs": {"enable_thinking": false}}']
    assert LQ.main(run) == 0
    n_calls = len(Mock.log)
    assert n_calls == 6 and [len(c["ids"]) for c in Mock.log[:3]] == [4, 4, 2]
    assert Mock.log[0]["body"]["chat_template_kwargs"] == {"enable_thinking": False}
    assert Mock.log[0]["body"]["cache_prompt"] is True and Mock.log[0]["body"]["temperature"] == 0
    assert Mock.log[0]["n_img"] == 0 and Mock.log[3]["n_img"] >= 2
    assert LQ.main(run) == 0 and len(Mock.log) == n_calls            # both arms done: nothing is called again
    rows = [json.loads(x) for x in (LQ.config.RESULTS / "local_qa/t/runs.jsonl").read_text().splitlines()]
    text, page = rows
    assert text["acc_all"] == 1.0 and page["acc_all"] == 0.6           # the <think> block is not parsed as answers
    assert page["acc"] == {"complex": 0.0, "needle": 1.0} and not page["incomplete"]
    assert text["gen_tokens_per_s"] == 30.0 and text["prefill_tokens_per_s"] == 1000.0
    assert LQ.main(["speed", *common, "--label", "plain", "--repeats", "2"]) == 0
    assert LQ.main(["analyse", *common]) == 0
    md = (LQ.config.RESULTS / "local_qa/t/summary.md").read_text()
    assert "| page118 | 1 |" in md and "-40.0" in md and "| loss |" in md and "generation, plain: 30.0" in md


def test_too_few_prompt_tokens_marks_the_arm_invalid(server):
    Mock.prompt_tokens = 20
    assert LQ.main(["run", "--run-name", "u", "--base-url", server, "--page-px", "640", "--arms", "text",
                    "page118"]) == 0
    rows = [json.loads(x) for x in (LQ.config.RESULTS / "local_qa/u/runs.jsonl").read_text().splitlines()]
    assert rows[1]["incomplete"] and not rows[0]["incomplete"]
    assert "INVALID" in LQ.summarise(rows, [])


def test_server_down_is_an_error_not_a_row(server):
    assert LQ.main(["run", "--run-name", "v", "--base-url", "http://127.0.0.1:9", "--arms", "none",
                    "--timeout", "5"]) == 1
    assert not (LQ.config.RESULTS / "local_qa/v/runs.jsonl").exists()
