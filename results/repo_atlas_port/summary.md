# Repo atlas, Q2 (gitleaks Go -> Rust): summary

*Written by `experiments/repo_atlas_port.py analyse`. One run per arm: an anecdote, not a measurement.*

| Arm | Tests passed | Builds | Concurrent | thread/rayon/async | Mutex / RwLock / Arc / channels | `unsafe` | Rust lines | Turns | CLI turns | API calls | Cost | Hours | Output tokens | Cut |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| plain | 35 / 35 (100.0%) | yes | yes | 1 / 0 / 0 | 8 / 0 / 12 / 0 | 3 | 9,431 | 102 | 4 | 82 | $11.32 | 0.69 | 234,160 | no |
| atlas | 35 / 35 (100.0%) | yes | no | 0 / 0 / 0 | 0 / 0 / 20 / 0 | 2 | 8,902 | 99 | 6 | 97 | $11.20 | 0.59 | 201,480 | no |

**Predictions:**
* B1 (neither arm passes every test): **FAIL** (plain 35, atlas 35 of 35)
* B2 (test shares differ by < 15 points): **PASS** (0.0 points)
* B3 (atlas uses fewer turns): **PASS** (plain 102, atlas 99 turns)

Spend: $23.14 of $50.0 (spend.jsonl).
