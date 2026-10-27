"""Convert the NIS repository's MATLAB-table .mat files to CSV for verification."""
import sys, pathlib, warnings
import numpy as np, pandas as pd
from matio import load_from_mat
warnings.filterwarnings("ignore")

def unwrap(x):
    while isinstance(x, np.ndarray) and x.dtype == object and x.size == 1:
        x = x.item()
    return x

def to_frame(v):
    if isinstance(v, pd.DataFrame):
        return v
    props = v.properties
    data = unwrap(props["data"]) if not isinstance(props["data"], (list, tuple)) else props["data"]
    names = [str(unwrap(n)) for n in np.ravel(props["varnames"])]
    cols = list(np.ravel(data)) if isinstance(data, np.ndarray) else list(data)
    out = {}
    for name, col in zip(names, cols):
        col = unwrap(col)
        if hasattr(col, "properties") or not isinstance(col, np.ndarray):
            # categorical or other object column: keep codes via str()
            col = np.asarray(getattr(col, "codes", col), dtype=object).ravel() if not isinstance(col, np.ndarray) else col
        arr = np.asarray(col)
        if arr.ndim == 2 and arr.shape[1] > 1:   # multi-column variable
            for j in range(arr.shape[1]):
                out[f"{name}_{j}"] = arr[:, j]
        else:
            out[name] = arr.ravel()
    return pd.DataFrame(out)

src = pathlib.Path(sys.argv[1]); out = pathlib.Path(sys.argv[2]); out.mkdir(exist_ok=True)
ok = 0
for p in sorted(src.glob("*.mat")):
    try:
        d = load_from_mat(str(p))
        key = [k for k in d if not k.startswith("__")][0]
        df = to_frame(d[key])
        df.to_csv(out / (p.stem + ".csv"), index=False)
        ok += 1
    except Exception as e:
        print("FAIL", p.name, type(e).__name__, str(e)[:150])
print("converted", ok)
