"""Compare scikit-reduce with the reference MATLAB code run in Octave."""
import pathlib
import numpy as np, scipy.io as sio
from skreduce import NIS, BLDOD

import sys
inp, ref = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2])
alphas, lambdas = [0.3, 0.5, 1, 2], [0.5, 1, 2]

nis_total = nis_equal = 0
for f in sorted(inp.glob("nis_*.mat")):
    X = sio.loadmat(f)["X"]; R = sio.loadmat(ref / f.name)
    for a, alpha in enumerate(alphas, 1):
        expected = np.sort(np.ravel(R[f"idx_{a}"]).astype(int) - 1)
        got = NIS(alpha=alpha).fit(X).sample_indices_
        nis_total += 1
        if got.shape == expected.shape and np.array_equal(got, expected):
            nis_equal += 1
        else:
            print(f"NIS MISMATCH {f.stem} alpha={alpha}: python {got.size} vs octave {expected.size}")
print(f"NIS: {nis_equal}/{nis_total} (dataset, alpha) pairs select exactly the same instances")

worst = {}
bl_total = bl_equal = 0
for f in sorted(inp.glob("bldod_*.mat")):
    X = sio.loadmat(f)["X"]; R = sio.loadmat(ref / f.name)
    for a, lam in enumerate(lambdas, 1):
        expected = np.ravel(R[f"score_{a}"])
        for algo in ["brute", "kd_tree", "ball_tree", "auto"]:
            got = BLDOD(std_coef=lam, algorithm=algo).fit(X).outlier_factor_
            diff = float(np.max(np.abs(got - expected)))
            worst[algo] = max(worst.get(algo, 0.0), diff)
            bl_total += 1
            bl_equal += diff < 1e-12
            if diff >= 1e-12:
                print(f"BLDOD DIFF {f.stem} lambda={lam} {algo}: max abs diff {diff:.3g}")
print(f"BLDOD: {bl_equal}/{bl_total} (dataset, lambda, algorithm) runs match to 1e-12")
print("BLDOD worst abs diff per algorithm:", {k: f"{v:.2e}" for k, v in worst.items()})
