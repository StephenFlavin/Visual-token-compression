## Layer 1: 16 tickets with all four briefs

| planner | briefs | fix files named (mean) | named source files not in the fix | first file in the fix | fix functions named | invented paths + symbols (of all named) | format problems | judge score (0-6) | judge OUTCOME yes | mean cost | mean seconds | mean turns |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| tree | 16 | 45.2% | 37.0% | 15 of 16 | 51.5% | 3 of 111 (2.7%) | 8 | 5.62 | 14 of 16 | $0.052 | 15 | 1.0 |
| map | 16 | 46.1% | 35.6% | 15 of 16 | 48.1% | 0 of 134 (0.0%) | 5 | 5.44 | 11 of 16 | $0.155 | 15 | 1.0 |
| atlas | 16 | 51.9% | 25.6% | 15 of 16 | 50.0% | 1 of 130 (0.8%) | 8 | 5.75 | 14 of 16 | $0.382 | 23 | 1.0 |
| scout | 16 | 57.0% | 2.8% | 15 of 16 | 52.8% | 2 of 126 (1.6%) | 9 | 5.88 | 15 of 16 | $0.154 | 30 | 9.6 |

Paired (layer 1):
* fix files named, atlas vs tree: mean difference +6.7 points over 16; atlas higher on 4, lower on 2, sign test p = 0.688
* fix files named, scout vs atlas: mean difference +5.1 points over 16; scout higher on 3, lower on 1, sign test p = 0.625
* fix files named, map vs tree: mean difference +0.9 points over 16; map higher on 2, lower on 3, sign test p = 1.000
* fix functions named, atlas vs tree: mean difference -1.5 points over 16; atlas higher on 2, lower on 3, sign test p = 1.000
* judge score / 6, atlas vs tree: mean difference +2.1 points over 16; atlas higher on 4, lower on 2, sign test p = 0.688
* judge score / 6, scout vs atlas: mean difference +2.1 points over 16; scout higher on 2, lower on 1, sign test p = 1.000

Invented paths and symbols (named, absent from the repository, not marked new):
* python-wheel-build__fromager-1124 tree: paths [] symbols ['Bootstrapper.resolve_version']
* nesquena__hermes-webui-330_interface tree: paths [] symbols ['locate']
* tox-dev__tox-3904 scout: paths ['tests/config/loader/toml/test_toml_loader.py'] symbols []
* pyjanitor-devs__pyjanitor-1593 tree: paths [] symbols ['_computations_expand']
* agronholm__anyio-975 atlas: paths [] symbols ['_is_anyio_cancellation']
* pypa__virtualenv-2940 scout: paths ['tests/unit/activation/test_activation.py'] symbols []

### Sizing (layer 1)

| planner | rank correlation SIZE vs real size (composite) | vs files | vs functions | vs lines | SIZE S / M / L | real class (design's own rule) M / L | SIZE = real class |
|---|---|---|---|---|---|---|---|
| tree | 0.33 | 0.39 | 0.37 | 0.35 | 2 / 12 / 2 | 8 / 8 | 9 of 16 |
| map | 0.31 | 0.29 | 0.23 | 0.39 | 2 / 10 / 4 | 8 / 8 | 9 of 16 |
| atlas | 0.38 | 0.42 | 0.12 | 0.35 | 2 / 10 / 4 | 8 / 8 | 8 of 16 |
| scout | 0.55 | 0.63 | 0.34 | 0.60 | 1 / 12 / 3 | 8 / 8 | 10 of 16 |

Confusion, atlas planner (rows: SIZE; columns: real class by the fix's source files):

| SIZE | real S | real M | real L |
|---|---|---|---|
| S | 0 | 2 | 0 |
| M | 0 | 5 | 5 |
| L | 0 | 1 | 3 |

## Layer 2: tickets that collide

Not evaluable: 1 usable group(s) of three or more tickets (tobymao/sqlglot 5).

## Layer 3: workers

Tickets with all four worker runs: 3 of 16.

| arm | tickets | resolved | median turns | mean cost | median seconds | gold files touched |
|---|---|---|---|---|---|---|
| opus, no brief (repo_atlas_swe plain) | 3 | 1 | 14 | $0.43 | 117 | 0.67 |
| opus_atlas | 3 | 2 | 19 | $0.48 | 163 | 0.67 |
| opus_tree | 3 | 2 | 25 | $0.63 | 155 | 0.67 |
| sonnet_atlas | 3 | 2 | 6 | $0.07 | 25 | 0.67 |
| haiku_atlas | 3 | 2 | 11 | $0.01 | 55 | 0.67 |

Paired (layer 3):
* resolved, opus+atlas brief vs no brief: brief only 1, no brief only 0, p = 1.000
* resolved, opus+atlas brief vs opus+tree brief: atlas only 0, tree only 0, p = 1.000
* turns, opus+atlas brief vs no brief: median 19 vs 14; brief lower on 1 of 3 untied, p = 1.000
* turns, opus+atlas brief vs opus+tree brief: atlas lower on 2 of 3 untied, p = 1.000
* cost, opus+atlas brief vs no brief: brief lower on 1 of 3, p = 1.000

### Right-sizing (atlas planner's WORKER against the measured tier)

| ticket | planner SIZE | planner WORKER | haiku low | sonnet medium | opus high | measured tier | verdict |
|---|---|---|---|---|---|---|---|
| python-wheel-build__fromager-1124 | L | sonnet / medium | yes $0.01 | yes $0.05 | yes $0.45 | haiku | too large |
| nesquena__hermes-webui-330_interface | M | sonnet / medium | no $0.01 | no $0.04 | no $0.53 | none | unresolvable |
| tox-dev__tox-3904 | M | sonnet / medium | yes $0.01 | yes $0.12 | yes $0.47 | haiku | too large |

Verdicts: right 0, too small 0, too large 2, unresolvable 1, no choice 0

| policy | tickets | resolved | total worker cost |
|---|---|---|---|
| always opus high | 3 | 2 | $1.45 |
| always haiku low | 3 | 2 | $0.03 |
| always sonnet medium | 3 | 2 | $0.21 |
| planner's choice | 3 | 2 | $0.21 |

Leak suspects in worker runs: 11
* ('python-wheel-build__fromager-1124', 'opus_atlas', ['V=/tmp/claude-0/-home-user-visual-bpe/588efa4c-1231-5348-b908-e0ab20b8aad7/scratchpad/pq/runs/6647f5a616/venv/bin/python\n$V -m pytest -q tests/test_bootstrapper.py::test_cache_lookup_resolver_exceptio', '/tmp/claude-0/-home-user-visual-bpe/588efa4c-1231-5348-b908-e0ab20b8aad7/scratchpad/pq/runs/6647f5a616/venv/bin/python -m pytest -q tests/test_cooldown_exact_pin.py tests/test_cooldown.py 2>&1 | tail '])
* ('python-wheel-build__fromager-1124', 'sonnet_atlas', ['cd /tmp/claude-0/-home-user-visual-bpe/588efa4c-1231-5348-b908-e0ab20b8aad7/scratchpad/pq/runs/a1cb37016d/fromager; git diff --stat; ../venv/bin/python -m pytest tests -q -x 2>&1 | tail -4'])
* ('python-wheel-build__fromager-1124', 'opus_tree', ['cd /tmp/claude-0/-home-user-visual-bpe/588efa4c-1231-5348-b908-e0ab20b8aad7/scratchpad/pq/runs/d0b68ebd38/fromager; grep -rn "cooldown\\|min_release_age" src/ | grep -v "^src/fromager/packagesettings/_', 'cd /tmp/claude-0/-home-user-visual-bpe/588efa4c-1231-5348-b908-e0ab20b8aad7/scratchpad/pq/runs/d0b68ebd38/fromager; sed -n 130,175p src/fromager/sources.py; sed -n 520,570p src/fromager/wheels.py; cat', 'cd /tmp/claude-0/-home-user-visual-bpe/588efa4c-1231-5348-b908-e0ab20b8aad7/scratchpad/pq/runs/d0b68ebd38/fromager; grep -rn "resolve_package_cooldown\\|_resolved_requirements\\|cache_resolution" tests/', 'cd /tmp/claude-0/-home-user-visual-bpe/588efa4c-1231-5348-b908-e0ab20b8aad7/scratchpad/pq/runs/d0b68ebd38/fromager; grep -rn "resolve(\\|resolve_source(\\|resolve_prebuilt\\|resolve_all_prebuilt\\|get_sou', 'cd /tmp/claude-0/-home-user-visual-bpe/588efa4c-1231-5348-b908-e0ab20b8aad7/scratchpad/pq/runs/d0b68ebd38/fromager; sed -n 150,175p src/fromager/bootstrapper.py; sed -n 220,300p src/fromager/bootstrap', "cd /tmp/claude-0/-home-user-visual-bpe/588efa4c-1231-5348-b908-e0ab20b8aad7/scratchpad/pq/runs/d0b68ebd38/fromager; sed -i 's/provider.cooldown = resolver.resolve_package_cooldown(ctx, req)$/provider.", "cd /tmp/claude-0/-home-user-visual-bpe/588efa4c-1231-5348-b908-e0ab20b8aad7/scratchpad/pq/runs/d0b68ebd38/fromager; python3 - <<'EOF'\np='src/fromager/bootstrap_requirement_resolver.py'\ns=open(p).read(", 'cd /tmp/claude-0/-home-user-visual-bpe/588efa4c-1231-5348-b908-e0ab20b8aad7/scratchpad/pq/runs/d0b68ebd38/fromager; grep -rln "min.release.age\\|cooldown" docs e2e | head; ', 'cd /tmp/claude-0/-home-user-visual-bpe/588efa4c-1231-5348-b908-e0ab20b8aad7/scratchpad/pq/runs/d0b68ebd38/fromager; cat docs/how-tos/release-age-cooldown.rst; cat e2e/test_bootstrap_cooldown.sh; cat e', 'cd /tmp/claude-0/-home-user-visual-bpe/588efa4c-1231-5348-b908-e0ab20b8aad7/scratchpad/pq/runs/d0b68ebd38/fromager; python3 - <<\'EOF\'\np=\'e2e/test_bootstrap_cooldown.sh\'\ns=open(p).read()\ns=s.replace(""', 'cd /tmp/claude-0/-home-user-visual-bpe/588efa4c-1231-5348-b908-e0ab20b8aad7/scratchpad/pq/runs/d0b68ebd38/fromager; grep -rn "new toplevel\\|new {\\|resolves to" src/fromager | head', 'cd /tmp/claude-0/-home-user-visual-bpe/588efa4c-1231-5348-b908-e0ab20b8aad7/scratchpad/pq/runs/d0b68ebd38/fromager; sed -n 1,60p tests/test_cooldown.py; sed -n 340,400p tests/test_cooldown.py; sed -n ', 'cd /tmp/claude-0/-home-user-visual-bpe/588efa4c-1231-5348-b908-e0ab20b8aad7/scratchpad/pq/runs/d0b68ebd38/fromager; sed -n 60,340p tests/test_cooldown.py | grep -n "def \\|requests_mock\\|resolve_source', 'cd /tmp/claude-0/-home-user-visual-bpe/588efa4c-1231-5348-b908-e0ab20b8aad7/scratchpad/pq/runs/d0b68ebd38/fromager; sed -n 80,280p tests/test_cooldown.py', 'cd /tmp/claude-0/-home-user-visual-bpe/588efa4c-1231-5348-b908-e0ab20b8aad7/scratchpad/pq/runs/d0b68ebd38/fromager; V=/tmp/claude-0/-home-user-visual-bpe/588efa4c-1231-5348-b908-e0ab20b8aad7/scratchpa', 'cd /tmp/claude-0/-home-user-visual-bpe/588efa4c-1231-5348-b908-e0ab20b8aad7/scratchpad/pq/runs/d0b68ebd38/fromager; V=/tmp/claude-0/-home-user-visual-bpe/588efa4c-1231-5348-b908-e0ab20b8aad7/scratchpa', 'cd /tmp/claude-0/-home-user-visual-bpe/588efa4c-1231-5348-b908-e0ab20b8aad7/scratchpad/pq/runs/d0b68ebd38/fromager; V=/tmp/claude-0/-home-user-visual-bpe/588efa4c-1231-5348-b908-e0ab20b8aad7/scratchpa', 'cd /tmp/claude-0/-home-user-visual-bpe/588efa4c-1231-5348-b908-e0ab20b8aad7/scratchpad/pq/runs/d0b68ebd38/fromager; V=/tmp/claude-0/-home-user-visual-bpe/588efa4c-1231-5348-b908-e0ab20b8aad7/scratchpa', 'cd /tmp/claude-0/-home-user-visual-bpe/588efa4c-1231-5348-b908-e0ab20b8aad7/scratchpad/pq/runs/d0b68ebd38/fromager; bash -n e2e/test_bootstrap_cooldown.sh && echo ok'])
* ('nesquena__hermes-webui-330_interface', 'opus_tree', ['/tmp/claude-0/-home-user-visual-bpe/588efa4c-1231-5348-b908-e0ab20b8aad7/scratchpad/pq/runs/c563647f02/venv/bin/python -m pytest tests/test_onboarding_mvp.py tests/test_onboarding_static.py tests/test', 'grep -n "HERMES_WEBUI_[A-Z_]* *|" README.md | head; /tmp/claude-0/-home-user-visual-bpe/588efa4c-1231-5348-b908-e0ab20b8aad7/scratchpad/pq/runs/c563647f02/venv/bin/python -m pytest tests -q -x 2>&1 | '])
* ('nesquena__hermes-webui-330_interface', 'sonnet_atlas', ['cd /tmp/claude-0/-home-user-visual-bpe/588efa4c-1231-5348-b908-e0ab20b8aad7/scratchpad/pq/runs/5c30cac7e5/hermes-webui; sed -n 60,112p api/onboarding.py; grep -n "chat_ready" api/onboarding.py; grep -', '/tmp/claude-0/-home-user-visual-bpe/588efa4c-1231-5348-b908-e0ab20b8aad7/scratchpad/pq/runs/5c30cac7e5/venv/bin/python -m pytest tests -q -k onboarding 2>&1 | tail -8'])
* ('nesquena__hermes-webui-330_interface', 'haiku_atlas', ['grep -rn "resolve_runtime_provider\\|lru_cache\\|_cache" api/*.py | grep -v "^api/onboarding" | head; /tmp/claude-0/-home-user-visual-bpe/588efa4c-1231-5348-b908-e0ab20b8aad7/scratchpad/pq/runs/0893e56b', '/tmp/claude-0/-home-user-visual-bpe/588efa4c-1231-5348-b908-e0ab20b8aad7/scratchpad/pq/runs/0893e56b79/venv/bin/python -m pytest tests/test_onboarding_ready.py tests/test_onboarding_static.py -q 2>&1 ', 'timeout 500 /tmp/claude-0/-home-user-visual-bpe/588efa4c-1231-5348-b908-e0ab20b8aad7/scratchpad/pq/runs/0893e56b79/venv/bin/python -m pytest tests/test_onboarding_mvp.py tests/test_sprint34.py tests/t', 'timeout 550 /tmp/claude-0/-home-user-visual-bpe/588efa4c-1231-5348-b908-e0ab20b8aad7/scratchpad/pq/runs/0893e56b79/venv/bin/python -m pytest tests -q -x 2>&1 | tail -15'])
* ('nesquena__hermes-webui-330_interface', 'opus_atlas', ['V=/tmp/claude-0/-home-user-visual-bpe/588efa4c-1231-5348-b908-e0ab20b8aad7/scratchpad/pq/runs/8793990039/venv/bin/python; $V -c "import hermes_cli" 2>&1; sed -n 625,700p api/config.py; grep -n "onboar', 'cat > /tmp/claude-0/-home-user-visual-bpe/588efa4c-1231-5348-b908-e0ab20b8aad7/scratchpad/patch1.py <<\'EOF\'\nimport pathlib\np = pathlib.Path("api/onboarding.py")\ns = p.read_text()\n\nold = \'\'\'def get_onb', 'cat > /tmp/claude-0/-home-user-visual-bpe/588efa4c-1231-5348-b908-e0ab20b8aad7/scratchpad/patch2.py <<\'EOF\'\nimport pathlib\np = pathlib.Path("static/onboarding.js")\ns = p.read_text()\nreps = [\n(\'\'\'    O', '/tmp/claude-0/-home-user-visual-bpe/588efa4c-1231-5348-b908-e0ab20b8aad7/scratchpad/pq/runs/8793990039/venv/bin/python -m pytest tests/test_onboarding_ready.py tests/test_onboarding_mvp.py tests/test_', '/tmp/claude-0/-home-user-visual-bpe/588efa4c-1231-5348-b908-e0ab20b8aad7/scratchpad/pq/runs/8793990039/venv/bin/python -m pytest tests -q -x 2>&1 | tail -5; git stash -q && /tmp/claude-0/-home-user-vi', 'cd /tmp/claude-0/-home-user-visual-bpe/588efa4c-1231-5348-b908-e0ab20b8aad7/scratchpad/pq/runs/8793990039/hermes-webui; P=/tmp/claude-0/-home-user-visual-bpe/588efa4c-1231-5348-b908-e0ab20b8aad7/scrat', 'cd /tmp/claude-0/-home-user-visual-bpe/588efa4c-1231-5348-b908-e0ab20b8aad7/scratchpad/pq/runs/8793990039/hermes-webui; P=/tmp/claude-0/-home-user-visual-bpe/588efa4c-1231-5348-b908-e0ab20b8aad7/scrat', 'cd /tmp/claude-0/-home-user-visual-bpe/588efa4c-1231-5348-b908-e0ab20b8aad7/scratchpad/pq/runs/8793990039/hermes-webui; git diff --stat'])
* ('tox-dev__tox-3904', 'opus_tree', ['V=/tmp/claude-0/-home-user-visual-bpe/588efa4c-1231-5348-b908-e0ab20b8aad7/scratchpad/pq/runs/72ac66036a/venv; ls $V/lib/python*/site-packages | grep -i -E "discovery|virtualenv"; grep -rn -i "aarch64', 'V=/tmp/claude-0/-home-user-visual-bpe/588efa4c-1231-5348-b908-e0ab20b8aad7/scratchpad/pq/runs/72ac66036a/venv; sed -n 655,700p $V/lib/python3.13/site-packages/python_discovery/_py_info.py; grep -n "py', "sed -i '/^import sysconfig$/d; /^import sys$/d' src/tox/config/loader/toml/_replace.py\nV=/tmp/claude-0/-home-user-visual-bpe/588efa4c-1231-5348-b908-e0ab20b8aad7/scratchpad/pq/runs/72ac66036a/venv/bin", 'V=/tmp/claude-0/-home-user-visual-bpe/588efa4c-1231-5348-b908-e0ab20b8aad7/scratchpad/pq/runs/72ac66036a/venv/bin/python\n$V -m pytest tests/config/loader/ini/test_factor.py -q -k "py39-arm64-linux-x86', 'V=/tmp/claude-0/-home-user-visual-bpe/588efa4c-1231-5348-b908-e0ab20b8aad7/scratchpad/pq/runs/72ac66036a/venv/bin/python\ngit stash -q -- src && (cd /tmp/dbg3902 && $V -m tox c -e py39-arm64 -k command', 'V=/tmp/claude-0/-home-user-visual-bpe/588efa4c-1231-5348-b908-e0ab20b8aad7/scratchpad/pq/runs/72ac66036a/venv/bin/python\n$V -m pytest tests/config tests/session/cmd/test_sequential.py tests/tox_env -q', 'V=/tmp/claude-0/-home-user-visual-bpe/588efa4c-1231-5348-b908-e0ab20b8aad7/scratchpad/pq/runs/72ac66036a/venv/bin/python\ngit stash -q; $V -m pytest tests/config/source/test_toml_pyproject.py::test_con'])
* ('tox-dev__tox-3904', 'opus_atlas', ['cd /tmp/claude-0/-home-user-visual-bpe/588efa4c-1231-5348-b908-e0ab20b8aad7/scratchpad/pq/runs/faa0ed94b0/tox; cat src/tox/config/loader/ini/factor.py; grep -n "factor\\|platform\\|sysconfig" src/tox/co', 'cd /tmp/claude-0/-home-user-visual-bpe/588efa4c-1231-5348-b908-e0ab20b8aad7/scratchpad/pq/runs/faa0ed94b0/tox; grep -rn "platform\\|sysconfig" tests/config/loader/ini/test_factor.py | head -40; grep -r', 'cd /tmp/claude-0/-home-user-visual-bpe/588efa4c-1231-5348-b908-e0ab20b8aad7/scratchpad/pq/runs/faa0ed94b0/tox; sed -n 440,530p tests/config/loader/ini/test_factor.py; sed -n 40,60p src/tox/config/load', 'cd /tmp/claude-0/-home-user-visual-bpe/588efa4c-1231-5348-b908-e0ab20b8aad7/scratchpad/pq/runs/faa0ed94b0/tox; grep -rln "get_platform\\|sys.platform" tests | head; git log --oneline -3 -- src/tox/conf', "cd /tmp/claude-0/-home-user-visual-bpe/588efa4c-1231-5348-b908-e0ab20b8aad7/scratchpad/pq/runs/faa0ed94b0/tox; python3 - <<'EOF'\np='src/tox/config/loader/ini/factor.py'\ns=open(p).read()\nold='''def fil", "cd /tmp/claude-0/-home-user-visual-bpe/588efa4c-1231-5348-b908-e0ab20b8aad7/scratchpad/pq/runs/faa0ed94b0/tox; sed -i '/^import sys$/d;/^import sysconfig$/d;/^from itertools import chain$/d' src/tox/c", "cd /tmp/claude-0/-home-user-visual-bpe/588efa4c-1231-5348-b908-e0ab20b8aad7/scratchpad/pq/runs/faa0ed94b0/tox; sed -i 's/ - by :user:`gaborbernat`\\././' docs/changelog/3902.bugfix.rst; cat docs/change", "cd /tmp/claude-0/-home-user-visual-bpe/588efa4c-1231-5348-b908-e0ab20b8aad7/scratchpad/pq/runs/faa0ed94b0/tox; sed -i 's/^import sys$/import sys\\nimport sysconfig/; s/^from tox.config.loader.ini impor", 'cd /tmp/claude-0/-home-user-visual-bpe/588efa4c-1231-5348-b908-e0ab20b8aad7/scratchpad/pq/runs/faa0ed94b0/tox; /tmp/claude-0/-home-user-visual-bpe/588efa4c-1231-5348-b908-e0ab20b8aad7/scratchpad/pq/ru', "cd /tmp/claude-0/-home-user-visual-bpe/588efa4c-1231-5348-b908-e0ab20b8aad7/scratchpad/pq/runs/faa0ed94b0/tox; sed -i 's/^        \\[testenv:py39-{arm64,x86_64}\\]$/        [testenv]/' tests/config/load", 'cd /tmp/claude-0/-home-user-visual-bpe/588efa4c-1231-5348-b908-e0ab20b8aad7/scratchpad/pq/runs/faa0ed94b0/tox; V=/tmp/claude-0/-home-user-visual-bpe/588efa4c-1231-5348-b908-e0ab20b8aad7/scratchpad/pq/', 'cd /tmp/claude-0/-home-user-visual-bpe/588efa4c-1231-5348-b908-e0ab20b8aad7/scratchpad/pq/runs/faa0ed94b0/tox; V=/tmp/claude-0/-home-user-visual-bpe/588efa4c-1231-5348-b908-e0ab20b8aad7/scratchpad/pq/', 'cd /tmp/claude-0/-home-user-visual-bpe/588efa4c-1231-5348-b908-e0ab20b8aad7/scratchpad/pq/runs/faa0ed94b0/tox; V=/tmp/claude-0/-home-user-visual-bpe/588efa4c-1231-5348-b908-e0ab20b8aad7/scratchpad/pq/'])
* ('tox-dev__tox-3904', 'sonnet_atlas', ['cd /tmp/claude-0/-home-user-visual-bpe/588efa4c-1231-5348-b908-e0ab20b8aad7/scratchpad/pq/runs/1e1a8ca324/tox; grep -n "sysconfig\\|sys.platform" -B8 -A8 src/tox/config/loader/toml/_replace.py; ls docs', "cd /tmp/claude-0/-home-user-visual-bpe/588efa4c-1231-5348-b908-e0ab20b8aad7/scratchpad/pq/runs/1e1a8ca324/tox\npython3 - <<'E'\np='src/tox/config/loader/ini/factor.py'\ns=open(p).read()\ns=s.replace('''  ", 'cd /tmp/claude-0/-home-user-visual-bpe/588efa4c-1231-5348-b908-e0ab20b8aad7/scratchpad/pq/runs/1e1a8ca324/tox\ngrep -n "sys\\b\\|sysconfig" src/tox/config/loader/toml/_replace.py', "cd /tmp/claude-0/-home-user-visual-bpe/588efa4c-1231-5348-b908-e0ab20b8aad7/scratchpad/pq/runs/1e1a8ca324/tox\nsed -i '7,8d' src/tox/config/loader/toml/_replace.py\nV=/tmp/claude-0/-home-user-visual-bpe"])
* ('tox-dev__tox-3904', 'haiku_atlas', ['PY=/tmp/claude-0/-home-user-visual-bpe/588efa4c-1231-5348-b908-e0ab20b8aad7/scratchpad/pq/runs/4f7ed0100e/venv/bin/python; $PY -m pytest tests/config -q -p no:cacheprovider 2>&1 | grep FAILED | sort >'])
Worker runs that wrote auto-memory files: 0

## Predictions

* P1: PASS (atlas 51.9% vs tree 45.2% fix files named)
* P2: PASS (invented: atlas 0.8%, scout 1.6%)
* P3: PASS (scout 57.0% vs atlas 51.9%)
* P4: not evaluable (collisions None)
* P5: PASS (atlas SIZE rank correlation 0.3794921695550902)
* P6: PASS (policies {'always opus high': {'tickets': 3, 'resolved': 2, 'cost_usd': 1.4498}, 'always haiku low': {'tickets': 3, 'resolved': 2, 'cost_usd': 0.0281}, 'always sonnet medium': {'tickets': 3, 'resolved': 2, 'cost_usd': 0.21}, "planner's choice": {'tickets': 3, 'resolved': 2, 'cost_usd': 0.21}})
* P7: FAIL (layer 3 {'n': 3, 'atlas_res': 2, 'plain_res': 1, 'atlas_turns': 19, 'plain_turns': 14, 'turns_test': {'n': 3, 'a_lower': 1, 'b_lower': 2, 'p': 1.0}})

## Spend (list price)

| what | $ |
|---|---|
| L1 atlas | 6.11 |
| L1 map | 2.48 |
| L1 scout | 2.60 |
| L1 tree | 0.83 |
| L3 haiku_atlas | 0.03 |
| L3 opus_atlas | 1.70 |
| L3 opus_tree | 1.90 |
| L3 sonnet_atlas | 0.21 |
| judge | 2.11 |
| screen | 0.50 |
| smoke | 0.01 |
| **total** | **18.47** |
