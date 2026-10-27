# Verification against the reference MATLAB code

These scripts run the original `NIS.m` and `BLDOD.m` in GNU Octave on the data sets of the
[NIS](https://github.com/fatihaydin1/NIS) and [BLDOD](https://github.com/fatihaydin1/BLDOD)
repositories and compare the results with scikit-reduce.

```bash
pip install scipy pandas mat-io   # plus scikit-reduce itself
tools/verify_reference/run.sh /tmp/skreduce-verify
```

Octave lacks a few MATLAB options used by the originals, so `octave/` holds copies with
three mechanical changes and nothing else:

- `min(X,[],1,'omitnan')` becomes `min(X,[],1)`; Octave's `min` already ignores NaN.
- `std(X,'omitnan')` becomes `omitnan_std(X)`, a column-wise sample standard deviation that ignores NaN.
- `round(x, 0)` becomes `round(x)`, which is the same operation.
