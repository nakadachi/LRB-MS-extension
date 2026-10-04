#!/bin/sh
# Low-noise (error-floor) comparison on [[250,10,15]] with the published SOGRAND+XZ points.
cd "$(dirname "$0")"
mkdir -p logs
for spec in "0.0296 1000000" "0.0224 2000000" "0.0169 4000000"; do
    set -- $spec
    python compare.py --code qt250 --decoders decoders_qt250_low.json --ps "$1" --max-shots "$2" \
        --max-fails 40 --chunk 2500 --workers 4 --seed-salt test --out results_qt250_low.json \
        >> logs/qt250_low.log 2>&1
done
