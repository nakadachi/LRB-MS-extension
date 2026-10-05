#!/bin/sh
# p = 0.0097 on [[250,10,15]] (the published SOGRAND+XZ point nearest p = 0.01).
cd "$(dirname "$0")"
. /tmp/claude-0/venv/bin/activate 2>/dev/null
python compare.py --code qt250 --decoders decoders_qt250_low.json --ps 0.0097 \
    --max-shots 24000000 --max-fails 30 --chunk 20000 --workers 4 --seed-salt test \
    --out results_qt250_low.json >> logs/qt250_low.log 2>&1
echo DONE >> logs/qt250_low.log
