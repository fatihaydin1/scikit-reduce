#!/usr/bin/env bash
# Compare scikit-reduce with the reference MATLAB code, run in GNU Octave.
#
# Usage: tools/verify_reference/run.sh WORKDIR
# Requires: git, GNU Octave (>= 8), Python with scikit-reduce, scipy, pandas and mat-io.
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
WORK="${1:?usage: run.sh WORKDIR}"
mkdir -p "$WORK" && cd "$WORK"

[ -d NIS ] || git clone --depth 1 https://github.com/fatihaydin1/NIS.git
[ -d BLDOD ] || git clone --depth 1 https://github.com/fatihaydin1/BLDOD.git

# The NIS data sets are MATLAB tables, which SciPy cannot read directly.
python "$HERE/convert_nis_tables.py" NIS/datasets nis_csv
python "$HERE/prepare_inputs.py" nis_csv "BLDOD/outlier datasets" oct_in
mkdir -p oct_out
octave --no-gui --quiet --path "$HERE/octave" "$HERE/octave/run_reference.m" oct_in oct_out
python "$HERE/compare.py" oct_in oct_out
