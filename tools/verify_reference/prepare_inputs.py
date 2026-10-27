"""Write feature matrices as plain .mat files that Octave can load."""
import pathlib, warnings
import numpy as np, pandas as pd, scipy.io as sio
warnings.filterwarnings("ignore")
import sys
csv_dir, bldod_dir, out = map(pathlib.Path, sys.argv[1:4])
out.mkdir(parents=True, exist_ok=True)
for p in sorted(csv_dir.glob("*.csv")):
    X = pd.read_csv(p, low_memory=False).iloc[:, :-1].to_numpy(dtype=float)
    sio.savemat(out / f"nis_{p.stem}.mat", {"X": X})
for p in sorted(bldod_dir.glob("*.mat")):
    m = sio.loadmat(p)
    sio.savemat(out / f"bldod_{p.stem}.mat", {"X": np.asarray(m["X"], dtype=float)})
print(len(list(out.glob("*.mat"))), "files")
