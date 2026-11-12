## Gate

* claude-opus-5-5 at 88.2 px²/char: 20 of 20 (100%), 2 atlases, all pages arrived: True
* claude-sonnet-5-5 at 88.2 px²/char: 20 of 20 (100%), 2 atlases, all pages arrived: True

## Runs (Sonnet 5.5 high): tasks validated 16; rows 31

| arm | task-runs solved | tasks both / one / neither | run 1 | run 2 | median turns | mean cost per run | median cost | median seconds |
|---|---|---|---|---|---|---|---|---|
| Sonnet plain | 6 of 16 | 0 / 0 / 0 (0 tasks) | 6 of 16 | 0 of 0 | 8 | $0.07 | $0.07 | 44 |
| Sonnet atlas | 6 of 15 | 0 / 0 / 0 (0 tasks) | 6 of 15 | 0 of 0 | 6 | $0.24 | $0.18 | 50 |
| Q1 Opus plain | 19 of 32 | 9 / 1 / 6 (16 tasks) | 10 of 16 | 9 of 16 | 11 | $0.27 | $0.24 | 82 |
| Q1 Opus atlas | 17 of 32 | 8 / 1 / 7 (16 tasks) | 9 of 16 | 8 of 16 | 11.5 | $0.83 | $0.61 | 103 |
| plan-quality Sonnet medium + atlas brief (1 run) | 8 of 16 | - | - | - | 6 | $0.05 | $0.04 | 29 |

| task | Sonnet plain r1 / r2 (turns, $) | Sonnet atlas r1 / r2 (turns, $) | Q1 Opus plain r1/r2 | Q1 Opus atlas r1/r2 | PQ Sonnet brief |
|---|---|---|---|---|---|
| python-wheel-build__fromager-1124 | no (10, $0.09) / - | no (6, $0.33) / - | no / no | no / no | yes |
| nesquena__hermes-webui-330_interface | no (9, $0.10) / - | no (9, $0.38) / - | no / no | no / no | no |
| tox-dev__tox-3904 | no (9, $0.15) / - | no (7, $0.35) / - | yes / yes | yes / yes | yes |
| pytest-dev__pyfakefs-1286 | yes (11, $0.11) / - | yes (4, $0.17) / - | yes / yes | yes / yes | yes |
| pyjanitor-devs__pyjanitor-1593 | no (11, $0.12) / - | yes (10, $0.58) / - | yes / yes | yes / yes | yes |
| scientific-python__docstub-123 | no (5, $0.04) / - | no (4, $0.09) / - | no / no | no / no | no |
| jsonpickle__jsonpickle-588 | yes (10, $0.08) / - | yes (9, $0.15) / - | yes / yes | yes / yes | yes |
| jsonpickle__jsonpickle-587 | no (7, $0.05) / - | no (6, $0.11) / - | yes / yes | no / no | no |
| niconoe__pyinaturalist-643 | yes (3, $0.03) / - | yes (4, $0.14) / - | yes / yes | yes / yes | yes |
| microsoft__fabric-cicd-670 | no (4, $0.02) / - | no (9, $0.18) / - | no / no | no / no | no |
| tobymao__sqlglot-6067 | yes (7, $0.05) / - | yes (5, $0.47) / - | yes / yes | yes / yes | yes |
| agronholm__anyio-975 | no (10, $0.09) / - | no (5, $0.20) / - | yes / no | yes / yes | no |
| stanfordnlp__dspy-8607 | no (10, $0.07) / - | no (7, $0.31) / - | no / no | no / no | no |
| pypa__virtualenv-2940 | no (6, $0.05) / - | no (6, $0.14) / - | no / no | no / no | no |
| MAIF__arta-50 | yes (4, $0.02) / - | yes (4, $0.04) / - | yes / yes | yes / yes | yes |
| tobymao__sqlglot-5359 | yes (7, $0.07) / - | - / - | yes / yes | yes / no | no |

## Checks
Leak suspects: 2 runs
  * microsoft__fabric-cicd-670 atlas r1: ["sed -i '36,39d' README.md; sed -i '35a\\\n' README.md; sed -n 28,42p README.md; sed -i 's/^pip install fabric-cicd$/&/' docs/index.md; sed -n 15,28p docs/index.md"]
  * tobymao__sqlglot-6067 plain r1: ['cd /tmp/claude-0/*/*/scratchpad/swe_sonnet/runs/abfa9aa131/sqlglot 2>/dev/null || true; V=/tmp/claude-0/-home-user-visual-bpe/588efa4c-1231-5348-b908-e0ab20b8aad7/scratchpad/swe_sonnet/runs/abfa9aa131']
Memory: runs with a memory file before or after: 0 of 31; init events naming an auto-memory path: 0
Pages dropped (atlas runs): 0; models served: ['claude-sonnet-5-5']
Runs with more than one result event: 0; timeouts: 0; errors: 0

Spent (list price, gate + agent runs): $5.41
