#!/bin/sh
# MBP4 + LRB-MS with relay ladders, on the same samples as run_all.sh (seed salt "test").
cd "$(dirname "$0")"
mkdir -p logs
run() {
    python compare.py --code "$1" --decoders "decoders_$1.json" --ps "$2" --max-shots 200000 \
        --max-fails 50 --chunk 100 --workers 4 --seed-salt test --out "results_$1.json" \
        --only "MBP4+LRB-MS-8+ladder (ours)" >> "logs/$1.log" 2>&1
}
run qt250 0.0392,0.0518,0.0685,0.0906
run qt432 0.06,0.07,0.08,0.09,0.10
run qtC16 0.07,0.08,0.09,0.10,0.11
