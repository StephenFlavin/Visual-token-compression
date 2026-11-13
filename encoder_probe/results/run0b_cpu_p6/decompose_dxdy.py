"""Post-hoc dx / dy decomposition of a Run 0b alignment matrix (interpretation only, not the verdict).
  python3 encoder_probe/results/run0b_cpu_p6/decompose_dxdy.py encoder_probe/results/run0b_cpu_p6 latin_p6"""
import sys, numpy as np
sys.path.insert(0, '/home/user/visual-token-compaction'); sys.path.insert(0, '/home/user/visual-token-compaction/encoder_probe/experiments')
import capacity_curve as C
from pathlib import Path
S = C.load_jsonl(Path(sys.argv[1]) / "samples.jsonl")
G, shifts = C.alignment_matrix(S, sys.argv[2], "natural")
P, K = G.shape
vals = sorted({s[0] for s in shifts})
A = np.zeros((P, len(vals), len(vals)))      # [page, dy, dx]
for k, (dx, dy) in enumerate(shifts):
    A[:, vals.index(dy), vals.index(dx)] = G[:, k]
rng = np.random.default_rng(0)
def bm(x): return C.boot_mean(x, 10000, 0)
def f(t): return f"{t[0]:.2f} ({t[1]:.2f} to {t[2]:.2f})"
row = A.mean(2); col = A.mean(1)
print("dy means:", {v: f(bm(row[:, i])) for i, v in enumerate(vals)})
print("dx means:", {v: f(bm(col[:, i])) for i, v in enumerate(vals)})
def perm_spread(M, reps=5000):
    obs = np.ptp(M.mean(0)); c = 0
    for _ in range(reps):
        Mp = np.take_along_axis(M, np.argsort(rng.random(M.shape), 1), 1)
        c += np.ptp(Mp.mean(0)) >= obs
    return obs, (c + 1) / (reps + 1)
print("dy main effect spread, perm p:", perm_spread(row))
print("dx main effect spread, perm p:", perm_spread(col))
# dx within each dy row (labels permuted within page x row)
obs = np.mean([np.ptp(A[:, i, :].mean(0)) for i in range(len(vals))]); c = 0
for _ in range(5000):
    Ap = np.take_along_axis(A, np.argsort(rng.random(A.shape), 2), 2)
    c += np.mean([np.ptp(Ap[:, i, :].mean(0)) for i in range(len(vals))]) >= obs
print("dx within dy rows: mean spread", obs, "perm p", (c + 1) / 5001)
iy = {v: i for i, v in enumerate(vals)}
for a, b in [(21, 7), (28, 14), (21, 28), (7, 14), (21, 14)]:
    if a in iy and b in iy:
        print(f"dy={a} - dy={b}:", f(bm(row[:, iy[a]] - row[:, iy[b]])))
if 28 in iy:
    print("dy=28 row - mean other rows:", f(bm(row[:, iy[28]] - row[:, [i for v, i in iy.items() if v != 28]].mean(1))))
