#!/bin/sh
# Clones the code instances used by the related-work papers into ./external (not tracked).
set -e
cd "$(dirname "$0")"
mkdir -p external
[ -d external/qtanner ] || git clone --depth 1 https://github.com/RebKatRad/qTanner external/qtanner
[ -d external/check-weight-constrained-quantum-codes ] || \
    git clone --depth 1 https://github.com/lilyxy/Check-weight-constrained-quantum-codes \
        external/check-weight-constrained-quantum-codes
