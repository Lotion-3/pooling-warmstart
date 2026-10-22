#!/usr/bin/env bash
# Runs the rest of the overnight compute once Phase 4 has finished. Logs to logs/pipeline.log.
cd "$(dirname "$0")/.."
export PYTHONIOENCODING=utf-8
log() { echo "$(date '+%Y-%m-%d %H:%M:%S') $*" | tee -a logs/pipeline.log; }
until grep -q "finished\|deadline reached" logs/phase4.log 2>/dev/null; do sleep 30; done
log "phase4 finished; starting fixed-network solve"
python -P -W ignore src/runner.py --instances data/instances/instances_fixed.parquet --out data/solved/parts_fixed \
  --workers 19 --time-limit 60 --starts 20 --batch 250 --log logs/phase4_fixed.log --deadline 06:45 >> logs/phase4_fixed_stdout.txt 2>&1
log "fixed solve done; consolidating"
python -P -W ignore scripts/consolidate.py >> logs/pipeline.log 2>&1
for ds in varied fixed; do
  log "phase5 $ds"
  python -P -W ignore scripts/phase5.py --dataset $ds --workers 19 --threads 16 >> logs/phase5_$ds.txt 2>&1
  log "phase5 $ds exit=$?"
done
python -P -W ignore scripts/report.py >> logs/pipeline.log 2>&1
log "report exit=$?; pipeline done"
