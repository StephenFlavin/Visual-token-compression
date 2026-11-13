## Gate

* claude-opus-5-5 at 88.2 px²/char: 20 of 20 (100%), 2 atlases, all pages arrived: True
* claude-sonnet-5-5 at 88.2 px²/char: 20 of 20 (100%), 2 atlases, all pages arrived: True

## Runs (Sonnet 5.5 high): tasks validated 16; rows 64

| arm | task-runs solved | tasks both / one / neither | run 1 | run 2 | median turns | mean cost per run | median cost | median seconds |
|---|---|---|---|---|---|---|---|---|
| Sonnet plain | 14 of 32 | 6 / 2 / 8 (16 tasks) | 6 of 16 | 8 of 16 | 7.5 | $0.07 | $0.07 | 41 |
| Sonnet atlas | 13 of 32 | 6 / 1 / 9 (16 tasks) | 7 of 16 | 6 of 16 | 6 | $0.26 | $0.19 | 42 |
| Q1 Opus plain | 19 of 32 | 9 / 1 / 6 (16 tasks) | 10 of 16 | 9 of 16 | 11 | $0.27 | $0.24 | 82 |
| Q1 Opus atlas | 17 of 32 | 8 / 1 / 7 (16 tasks) | 9 of 16 | 8 of 16 | 11.5 | $0.83 | $0.61 | 103 |
| plan-quality Sonnet medium + atlas brief (1 run) | 8 of 16 | - | - | - | 6 | $0.05 | $0.04 | 29 |

| task | Sonnet plain r1 / r2 (turns, $) | Sonnet atlas r1 / r2 (turns, $) | Q1 Opus plain r1/r2 | Q1 Opus atlas r1/r2 | PQ Sonnet brief |
|---|---|---|---|---|---|
| python-wheel-build__fromager-1124 | no (10, $0.09) / no (7, $0.07) | no (6, $0.33) / no (7, $0.33) | no / no | no / no | yes |
| nesquena__hermes-webui-330_interface | no (9, $0.10) / no (9, $0.08) | no (9, $0.38) / no (6, $0.32) | no / no | no / no | no |
| tox-dev__tox-3904 | no (9, $0.15) / yes (10, $0.16) | no (7, $0.35) / no (6, $0.33) | yes / yes | yes / yes | yes |
| pytest-dev__pyfakefs-1286 | yes (11, $0.11) / yes (10, $0.09) | yes (4, $0.17) / yes (5, $0.18) | yes / yes | yes / yes | yes |
| pyjanitor-devs__pyjanitor-1593 | no (11, $0.12) / yes (8, $0.13) | yes (10, $0.58) / no (8, $0.51) | yes / yes | yes / yes | yes |
| scientific-python__docstub-123 | no (5, $0.04) / no (5, $0.05) | no (4, $0.09) / no (5, $0.09) | no / no | no / no | no |
| jsonpickle__jsonpickle-588 | yes (10, $0.08) / yes (11, $0.09) | yes (9, $0.15) / yes (10, $0.15) | yes / yes | yes / yes | yes |
| jsonpickle__jsonpickle-587 | no (7, $0.05) / no (10, $0.06) | no (6, $0.11) / no (6, $0.11) | yes / yes | no / no | no |
| niconoe__pyinaturalist-643 | yes (3, $0.03) / yes (3, $0.02) | yes (4, $0.14) / yes (4, $0.14) | yes / yes | yes / yes | yes |
| microsoft__fabric-cicd-670 | no (4, $0.02) / no (4, $0.02) | no (9, $0.18) / no (6, $0.16) | no / no | no / no | no |
| tobymao__sqlglot-6067 | yes (7, $0.05) / yes (5, $0.04) | yes (5, $0.47) / yes (6, $0.48) | yes / yes | yes / yes | yes |
| agronholm__anyio-975 | no (10, $0.09) / no (6, $0.08) | no (5, $0.20) / no (6, $0.19) | yes / no | yes / yes | no |
| stanfordnlp__dspy-8607 | no (10, $0.07) / no (12, $0.10) | no (7, $0.31) / no (5, $0.28) | no / no | no / no | no |
| pypa__virtualenv-2940 | no (6, $0.05) / no (5, $0.05) | no (6, $0.14) / no (5, $0.14) | no / no | no / no | no |
| MAIF__arta-50 | yes (4, $0.02) / yes (4, $0.03) | yes (4, $0.04) / yes (4, $0.04) | yes / yes | yes / yes | yes |
| tobymao__sqlglot-5359 | yes (7, $0.07) / yes (9, $0.08) | yes (11, $0.87) / yes (8, $0.49) | yes / yes | yes / no | no |

## Paired tests over 16 tasks with both runs of both arms (exact)
* resolved runs per task (0-2): atlas higher on 0, plain higher on 1, tied 15; sign test p = 1.000
* resolved in both runs: atlas only 0, plain only 0; exact McNemar p = 1.000
* atlas - plain resolved task-runs: -1 of 32; bootstrap 95% interval over tasks -3 .. +0
* mean turns per task: atlas lower on 11 of 14 untied, sign test p = 0.0574; medians 6 vs 8.25
* mean cost per task: atlas lower on 0 of 16, p = 0.0000; mean per run $0.26 vs $0.07 (ratio 3.72)
* mean seconds per task: atlas lower on 7 of 16, p = 0.804

Flips between run 1 and run 2:
* plain: 2 of 16 (tox-dev__tox-3904, pyjanitor-devs__pyjanitor-1593)
* atlas: 1 of 16 (pyjanitor-devs__pyjanitor-1593)
* atlas vs plain within each run: run 1: atlas 7 vs plain 6, run 2: atlas 6 vs plain 8

## Predictions
* S1 the gate passes without re-rendering: PASS
* S2 Sonnet atlas solves no more than plain (difference under 3 of 32): PASS (-1 of 32)
* S3 Sonnet atlas stays below Opus plain (Q1) on the 16 tasks complete in both: PASS (13 vs 19 of 32)
* S4 the atlas multiplies Sonnet's cost per run by at least 3: PASS (3.72x)

## Checks
Leak suspects: 2 runs
  * microsoft__fabric-cicd-670 atlas r1: ["sed -i '36,39d' README.md; sed -i '35a\\\n' README.md; sed -n 28,42p README.md; sed -i 's/^pip install fabric-cicd$/&/' docs/index.md; sed -n 15,28p docs/index.md"]
  * tobymao__sqlglot-6067 plain r1: ['cd /tmp/claude-0/*/*/scratchpad/swe_sonnet/runs/abfa9aa131/sqlglot 2>/dev/null || true; V=/tmp/claude-0/-home-user-visual-token-compaction/588efa4c-1231-5348-b908-e0ab20b8aad7/scratchpad/swe_sonnet/runs/abfa9aa131']
Memory: runs with a memory file before or after: 0 of 64; init events naming an auto-memory path: 0
Pages dropped (atlas runs): 0; models served: ['claude-sonnet-5-5']
Runs with more than one result event: 1; timeouts: 0; errors: 0

Spent (list price, gate + agent runs): $11.38
