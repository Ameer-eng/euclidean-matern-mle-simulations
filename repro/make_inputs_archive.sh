#!/bin/bash
# Pack the saved per-fit inputs read by repro/regenerate.sh and the audits into one
# deterministic, checksummed archive (LOCAL; not for publication):
#   repro/make_inputs_archive.sh [DATE]          default DATE = 2026-11-16
# Writes repro/inputs_<DATE>/
#   matern_inputs_<DATE>.tar.gz      results/{fits,kappa,hess0,theory}, audit/out/score0,
#                                    audit/out/score0_local (*.npz only; post-refit files;
#                                    results/*_prerefit is excluded) + results/kappa/fit_hosts.csv
#   matern_inputs_<DATE>.files.txt   the member list, in archive order
#   inputs_SHA256SUMS.txt.gz         sha256 of every member (same list as the bundle's)
#   SHA256SUMS                       sha256 of the three files above
# Restore from the project root with:  tar -xzf matern_inputs_<DATE>.tar.gz
# and verify with:                    zcat inputs_SHA256SUMS.txt.gz | sha256sum -c --quiet
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
D=${1:-2026-11-16}
A=repro/inputs_$D
mkdir -p "$A"
L=$A/matern_inputs_$D.files.txt
{ find results/fits results/kappa results/hess0 results/theory audit/out/score0 audit/out/score0_local \
       -type f -name '*.npz'; echo results/kappa/fit_hosts.csv; } | LC_ALL=C sort > "$L"
tar --sort=none --mtime="$D 00:00Z" --owner=0 --group=0 --numeric-owner -cf - -T "$L" \
    | gzip -n -6 > "$A/matern_inputs_$D.tar.gz"
xargs -d '\n' sha256sum < "$L" | gzip -n > "$A/inputs_SHA256SUMS.txt.gz"
( cd "$A" && sha256sum "matern_inputs_$D.tar.gz" "matern_inputs_$D.files.txt" inputs_SHA256SUMS.txt.gz > SHA256SUMS )
echo "$A: $(wc -l < "$L") files; $(du -h "$A/matern_inputs_$D.tar.gz" | cut -f1)"
cat "$A/SHA256SUMS"
