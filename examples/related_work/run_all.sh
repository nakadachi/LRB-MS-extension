#!/bin/sh
# Final paired comparison (fresh samples: seed salt "test"); results_<code>.json and logs/.
cd "$(dirname "$0")"
mkdir -p logs
run() {
    python compare.py --code "$1" --decoders "decoders_$1.json" --ps "$2" --max-shots 100000 \
        --max-fails 50 --chunk 100 --workers 4 --seed-salt test --out "results_$1.json" \
        >> "logs/$1.log" 2>&1
}
run qt250 0.0392,0.0518,0.0685,0.0906
run qt432 0.06,0.07,0.08,0.09,0.10
run qtC16 0.07,0.08,0.09,0.10,0.11
run bb144 0.04,0.05,0.06
# LEAD with the strong regularisation alpha = 0.01 the paper uses against local overconfidence
python compare.py --code qt250 --decoders decoders_qt250.json --ps 0.0392,0.0518,0.0685,0.0906 --max-shots 100000 --max-fails 50 --chunk 100 --workers 4 --seed-salt test --out results_qt250.json --only "LEAD α=0.01 [Xiao et al.]" >> logs/qt250.log 2>&1
