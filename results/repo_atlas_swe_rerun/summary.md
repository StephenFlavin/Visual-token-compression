Tasks validated: 16 of 16; with both runs of both arms: 9; rows: 54

| arm | task-runs resolved | tasks resolved in both / one / neither | median turns (runs) | mean cost per run | median cost per run | median seconds | median input tokens | pages dropped | first test (1 run): resolved of 16, median turns, mean cost |
|---|---|---|---|---|---|---|---|---|---|
| plain | 10 of 18 | 5 / 0 / 4 | 11 | $0.23 | $0.21 | 64 | 150,734 | 0 | 4 of 9, 13, $0.26 |
| atlas | 8 of 18 | 4 / 0 / 5 | 9 | $0.57 | $0.43 | 82 | 502,673 | 0 | 6 of 9, 10, $0.54 |

| task | plain r1 / r2 (turns, cost) | atlas r1 / r2 (turns, cost) | first test plain / atlas |
|---|---|---|---|
| python-wheel-build__fromager-1124 | no (14, $0.35) / no (14, $0.28) | no (14, $0.85) / - | no / no |
| nesquena__hermes-webui-330_interface | no (14, $0.30) / no (11, $0.31) | no (10, $0.96) / no (13, $0.96) | no / no |
| tox-dev__tox-3904 | yes (15, $0.44) / yes (21, $0.57) | yes (19, $1.06) / yes (17, $1.02) | yes / yes |
| pytest-dev__pyfakefs-1286 | yes (16, $0.32) / yes (17, $0.29) | yes (29, $1.20) / yes (12, $0.61) | yes / yes |
| pyjanitor-devs__pyjanitor-1593 | yes (12, $0.40) / - | yes (12, $3.61) / - | yes / yes |
| scientific-python__docstub-123 | no (11, $0.20) / no (8, $0.17) | no (7, $0.26) / no (5, $0.22) | no / no |
| jsonpickle__jsonpickle-588 | yes (19, $0.43) / - | yes (9, $0.39) / - | yes / yes |
| jsonpickle__jsonpickle-587 | yes (11, $0.19) / yes (14, $0.23) | no (8, $0.29) / no (13, $0.38) | no / yes |
| niconoe__pyinaturalist-643 | yes (4, $0.07) / yes (5, $0.08) | yes (6, $0.34) / yes (4, $0.32) | yes / yes |
| microsoft__fabric-cicd-670 | no (11, $0.16) / no (9, $0.13) | no (6, $0.34) / no (12, $0.47) | no / no |
| tobymao__sqlglot-6067 | yes (9, $0.15) / - | yes (9, $1.12) / yes (12, $2.00) | yes / yes |
| agronholm__anyio-975 | yes (13, $0.34) / no (11, $0.24) | yes (13, $0.62) / - | yes / yes |
| stanfordnlp__dspy-8607 | no (13, $0.22) / no (15, $0.28) | no (15, $0.92) / no (6, $0.66) | no / yes |
| pypa__virtualenv-2940 | no (8, $0.15) / no (9, $0.17) | no (7, $0.33) / - | no / no |
| MAIF__arta-50 | yes (5, $0.07) / yes (5, $0.08) | yes (8, $0.14) / yes (7, $0.14) | yes / yes |
| tobymao__sqlglot-5359 | yes (9, $0.25) / - | yes (11, $1.11) / - | no / yes |

Paired tests over 9 tasks (exact):
* resolved runs per task (0-2), atlas vs plain: atlas higher on 0, plain higher on 1, tied 8; sign test p = 1.000
* resolved in both runs: atlas only 0, plain only 1; exact McNemar p = 1.000
* difference in resolved task-runs (atlas - plain): -2 of 18 (-11.1 points); bootstrap 95% interval over tasks -6 .. +0
* mean turns per task, atlas vs plain: atlas lower on 5 of 8 untied, sign test p = 0.7266; medians 10.5 vs 12.5
* mean cost per task: atlas lower on 0 of 9, p = 0.0039; mean per run $0.57 vs $0.23 (ratio 2.50)
* mean seconds per task: atlas lower on 3 of 9, p = 0.508

Run-to-run agreement (tasks whose run 1 and run 2 differ in resolved):
* plain: 0 of 9 flip (none)
* atlas: 0 of 9 flip (none)
* each run 1 alone, atlas vs plain: run 1: atlas 4 vs plain 5, run 2: atlas 4 vs plain 5

Predictions:
* A1 atlas resolves at least as many task-runs as plain: FAIL (8 vs 10)
* A2 gap under 19 points (atlas - plain < 6 of 32 task-runs): PASS (-2, -11.1 points)
* A3 fewer turns with the atlas (sign test p < 0.05): FAIL (lower on 5 of 8, p = 0.7266)
* A4 atlas costs at least twice plain: PASS (2.50x)

Leak suspects (commands or paths reaching the web, a release, git history, the scratch dirs, a memory store): 4 runs
  * agronholm__anyio-975 plain r1: ["cat > /tmp/claude-0/t_reason.py <<'EOF'\nimport anyio, asyncio\nfrom anyio import CancelScope, get_cancelled_exc_class\nfrom anyio.lowlevel import checkpoint\nasync def main():\n    with CancelScope() as s", 'until [ -s /tmp/claude-0/before.txt ] && ! pgrep -f "pytest tests" >/dev/null; do sleep 5; done; git status --short; diff /tmp/claude-0/before.txt /tmp/claude-0/after.txt; tail -1 /tmp/claude-0/after.', 'pkill -f "until \\[ -s" ; git status --short; git stash list; diff /tmp/claude-0/before.txt /tmp/claude-0/after.txt && echo SAME; tail -1 /tmp/claude-0/after.txt']
  * jsonpickle__jsonpickle-588 plain r1: ['cd /tmp/claude-0/-home-user-visual-bpe/588efa4c-1231-5348-b908-e0ab20b8aad7/scratchpad/swe_rerun/runs/f749a88f4d/jsonpickle; grep -v ruff /tmp/claude-0/-home-user-visual-bpe/588efa4c-1231-5348-b908-e0']
  * agronholm__anyio-975 plain r2: ['cd /tmp/claude-0/-home-user-visual-bpe/588efa4c-1231-5348-b908-e0ab20b8aad7/scratchpad/swe_rerun/runs/f4d487bf78/anyio; V=/tmp/claude-0/-home-user-visual-bpe/588efa4c-1231-5348-b908-e0ab20b8aad7/scrat']
  * jsonpickle__jsonpickle-587 atlas r2: ['cat > /tmp/claude-0/t_slots.py <<\'EOF\'\nimport jsonpickle\nclass Foo:\n    __slots__ = ("bar", "secret")\n    _jsonpickle_exclude = {"secret"}\n    def __init__(self, bar, secret):\n        self.bar = bar\n ', '../venv/bin/python -m pytest tests -q -p no:ruff 2>&1 | grep FAILED > /tmp/claude-0/after.txt; git stash -q; ../venv/bin/python -m pytest tests -q -p no:ruff 2>&1 | grep FAILED > /tmp/claude-0/before.']
Memory: runs with a memory file before or after: 0 of 54; runs whose project store held any file after the run: 2
  * stanfordnlp__dspy-8607 atlas r1: ['c558cf5e-e821-404f-b627-5a18652a6e08/tool-results/bphuhlyg2.txt', 'c558cf5e-e821-404f-b627-5a18652a6e08/tool-results/btxhxa9zy.txt']
  * jsonpickle__jsonpickle-587 plain r2: ['4a02614b-2d02-45dc-8212-031109772b12/tool-results/b0npke5hl.txt']
Runs with more than one result event: 1; timeouts (rc -9): 0; errors: 0

Spent (list price, all agent calls): $27.19
