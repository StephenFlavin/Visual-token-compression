## Layer 1: 33 tickets with all four briefs

| planner | briefs | fix files named (mean) | named source files not in the fix | first file in the fix | fix functions named | invented paths + symbols (of all named) | format problems | judge score (0-6) | judge OUTCOME yes | mean cost | mean seconds | mean turns |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| tree | 33 | 48.1% | 38.2% | 29 of 33 | 44.6% | 6 of 239 (2.5%) | 14 | 5.48 | 28 of 33 | $0.049 | 15 | 1.0 |
| map | 33 | 47.6% | 35.4% | 29 of 33 | 49.2% | 1 of 283 (0.4%) | 13 | 5.42 | 25 of 33 | $0.157 | 14 | 1.0 |
| atlas | 33 | 50.1% | 24.7% | 29 of 33 | 50.3% | 1 of 246 (0.4%) | 16 | 5.67 | 27 of 33 | $0.414 | 22 | 1.0 |
| scout | 33 | 49.8% | 12.5% | 29 of 33 | 50.0% | 2 of 255 (0.8%) | 20 | 5.73 | 30 of 33 | $0.153 | 31 | 9.5 |

Paired (layer 1):
* fix files named, atlas vs tree: mean difference +2.0 points over 33; atlas higher on 6, lower on 6, sign test p = 1.000
* fix files named, scout vs atlas: mean difference -0.3 points over 33; scout higher on 3, lower on 4, sign test p = 1.000
* fix files named, map vs tree: mean difference -0.4 points over 33; map higher on 4, lower on 6, sign test p = 0.754
* fix functions named, atlas vs tree: mean difference +5.6 points over 33; atlas higher on 8, lower on 5, sign test p = 0.581
* judge score / 6, atlas vs tree: mean difference +3.0 points over 33; atlas higher on 7, lower on 4, sign test p = 0.549
* judge score / 6, scout vs atlas: mean difference +1.0 points over 33; scout higher on 3, lower on 2, sign test p = 1.000

Invented paths and symbols (named, absent from the repository, not marked new):
* python-wheel-build__fromager-1124 tree: paths [] symbols ['Bootstrapper.resolve_version']
* nesquena__hermes-webui-330_interface tree: paths [] symbols ['locate']
* tox-dev__tox-3904 scout: paths ['tests/config/loader/toml/test_toml_loader.py'] symbols []
* pyjanitor-devs__pyjanitor-1593 tree: paths [] symbols ['_computations_expand']
* agronholm__anyio-975 atlas: paths [] symbols ['_is_anyio_cancellation']
* pypa__virtualenv-2940 scout: paths ['tests/unit/activation/test_activation.py'] symbols []
* PyPSA__PyPSA-1498 tree: paths [] symbols ['set_from_frame']
* Spenhouet__confluence-markdown-exporter-92 tree: paths [] symbols ['get_authenticated_confluence', 'get_authenticated_jira']
* tobymao__sqlglot-5352 map: paths [] symbols ['Generator.decodecase_sql']

### Sizing (layer 1)

| planner | rank correlation SIZE vs real size (composite) | vs files | vs functions | vs lines | SIZE S / M / L | real class (design's own rule) M / L | SIZE = real class |
|---|---|---|---|---|---|---|---|
| tree | 0.32 | 0.34 | 0.30 | 0.25 | 6 / 22 / 5 | 16 / 17 | 17 of 33 |
| map | 0.27 | 0.29 | 0.23 | 0.22 | 6 / 20 / 7 | 16 / 17 | 17 of 33 |
| atlas | 0.24 | 0.39 | 0.14 | 0.14 | 7 / 19 / 7 | 16 / 17 | 15 of 33 |
| scout | 0.38 | 0.46 | 0.29 | 0.31 | 6 / 20 / 7 | 16 / 17 | 18 of 33 |

Confusion, atlas planner (rows: SIZE; columns: real class by the fix's source files):

| SIZE | real S | real M | real L |
|---|---|---|---|
| S | 0 | 6 | 1 |
| M | 0 | 9 | 10 |
| L | 0 | 1 | 6 |

## Layer 2: tickets that collide

Not evaluable: 1 usable group(s) of three or more tickets (tobymao/sqlglot 5).

## Layer 3: workers

Tickets with all four worker runs: 16 of 16.

| arm | tickets | resolved | median turns | mean cost | median seconds | gold files touched |
|---|---|---|---|---|---|---|
| opus, no brief (repo_atlas_swe plain) | 16 | 8 | 12.5 | $0.26 | 100 | 0.58 |
| opus_atlas | 16 | 11 | 12.5 | $0.32 | 90 | 0.54 |
| opus_tree | 16 | 10 | 15 | $0.42 | 173 | 0.59 |
| sonnet_atlas | 16 | 8 | 6 | $0.05 | 29 | 0.52 |
| haiku_atlas | 16 | 8 | 8.5 | $0.01 | 54 | 0.49 |

Paired (layer 3):
* resolved, opus+atlas brief vs no brief: brief only 3, no brief only 0, p = 0.250
* resolved, opus+atlas brief vs opus+tree brief: atlas only 1, tree only 0, p = 1.000
* turns, opus+atlas brief vs no brief: median 12.5 vs 12.5; brief lower on 5 of 14 untied, p = 0.424
* turns, opus+atlas brief vs opus+tree brief: atlas lower on 9 of 16 untied, p = 0.804
* cost, opus+atlas brief vs no brief: brief lower on 7 of 16, p = 0.804

### Right-sizing (atlas planner's WORKER against the measured tier)

| ticket | planner SIZE | planner WORKER | haiku low | sonnet medium | opus high | measured tier | verdict |
|---|---|---|---|---|---|---|---|
| python-wheel-build__fromager-1124 | L | sonnet / medium | yes $0.01 | yes $0.05 | yes $0.45 | haiku | too large |
| nesquena__hermes-webui-330_interface | M | sonnet / medium | no $0.01 | no $0.04 | no $0.53 | none | unresolvable |
| tox-dev__tox-3904 | M | sonnet / medium | yes $0.01 | yes $0.12 | yes $0.47 | haiku | too large |
| pytest-dev__pyfakefs-1286 | L | sonnet / medium | yes $0.01 | yes $0.06 | yes $0.25 | haiku | too large |
| pyjanitor-devs__pyjanitor-1593 | M | sonnet / medium | yes $0.01 | yes $0.07 | yes $0.97 | haiku | too large |
| scientific-python__docstub-123 | M | sonnet / medium | no $0.00 | no $0.04 | no $0.20 | none | unresolvable |
| jsonpickle__jsonpickle-588 | M | sonnet / medium | yes $0.01 | yes $0.06 | yes $0.68 | haiku | too large |
| jsonpickle__jsonpickle-587 | M | sonnet / medium | no $0.00 | no $0.04 | yes $0.26 | opus | too small |
| niconoe__pyinaturalist-643 | S | haiku / low | yes $0.00 | yes $0.02 | yes $0.09 | haiku | right |
| microsoft__fabric-cicd-670 | M | haiku / low | no $0.00 | no $0.02 | no $0.11 | none | unresolvable |
| tobymao__sqlglot-6067 | M | sonnet / medium | yes $0.00 | yes $0.04 | yes $0.19 | haiku | too large |
| agronholm__anyio-975 | L | sonnet / medium | no $0.01 | no $0.07 | yes $0.34 | opus | too small |
| stanfordnlp__dspy-8607 | L | sonnet / medium | no $0.00 | no $0.07 | yes $0.16 | opus | too small |
| pypa__virtualenv-2940 | M | sonnet / medium | no $0.00 | no $0.02 | no $0.17 | none | unresolvable |
| MAIF__arta-50 | M | haiku / medium | yes $0.00 | yes $0.04 | yes $0.17 | haiku | right |
| tobymao__sqlglot-5359 | S | sonnet / medium | no $0.00 | no $0.04 | no $0.13 | none | unresolvable |

Verdicts: right 2, too small 3, too large 6, unresolvable 5, no choice 0

| policy | tickets | resolved | total worker cost |
|---|---|---|---|
| always opus high | 16 | 11 | $5.17 |
| always haiku low | 16 | 8 | $0.10 |
| always sonnet medium | 16 | 8 | $0.81 |
| planner's choice | 16 | 8 | $0.73 |

Leak suspects in worker runs: 0
Worker runs that wrote auto-memory files: 0

## Predictions

* P1: FAIL (atlas 50.1% vs tree 48.1% fix files named)
* P2: PASS (invented: atlas 0.4%, scout 0.8%)
* P3: PASS (scout 49.8% vs atlas 50.1%)
* P4: not evaluable (collisions None)
* P5: FAIL (atlas SIZE rank correlation 0.2394549701192504)
* P6: FAIL (policies {'always opus high': {'tickets': 16, 'resolved': 11, 'cost_usd': 5.169}, 'always haiku low': {'tickets': 16, 'resolved': 8, 'cost_usd': 0.0956}, 'always sonnet medium': {'tickets': 16, 'resolved': 8, 'cost_usd': 0.8092}, "planner's choice": {'tickets': 16, 'resolved': 8, 'cost_usd': 0.7321}})
* P7: FAIL (layer 3 {'n': 16, 'atlas_res': 11, 'plain_res': 8, 'atlas_turns': 12.5, 'plain_turns': 12.5, 'turns_test': {'n': 14, 'a_lower': 5, 'b_lower': 9, 'p': 0.4239501953125}})

## Spend (list price)

| what | $ |
|---|---|
| L1 atlas | 14.12 |
| L1 map | 5.38 |
| L1 scout | 5.19 |
| L1 tree | 1.71 |
| L3 haiku_atlas | 0.11 |
| L3 opus_atlas | 5.17 |
| L3 opus_tree | 6.76 |
| L3 sonnet_atlas | 0.81 |
| judge | 4.51 |
| screen | 0.50 |
| smoke | 0.01 |
| **total** | **44.25** |
