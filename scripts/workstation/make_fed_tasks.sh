#!/usr/bin/env bash
# Writes the federated grid as one command per line (results/fed_tasks.txt), for fed_grid.sbatch. Owner: Kareem.
# Tasks whose result file already exists are left out, so a re-run only does what is missing.
#
#   bash scripts/workstation/make_fed_tasks.sh                           # everything below, defaults
#   PROTOCOLS="B" FOLDS="0 1" bash scripts/workstation/make_fed_tasks.sh
#   LR=0.003 MODELS_FED2="" MODELS_FED6="" MODELS_FED4="mlp,vqc" bash scripts/workstation/make_fed_tasks.sh
#                                   # FED-4 only, VQC with a tuned learning rate (an empty list means: none)
#
# Run from the repository root. Inputs come from experiments.fed6_zero_attacker.prepare_inputs: DATA-6 when
# fqml/data/preprocess.py is in the checkout, the stand-in otherwise (file names get "_d6" with DATA-6).
set -euo pipefail

PROTOCOLS=${PROTOCOLS:-"A B"}
FOLDS=${FOLDS:-"0 1 2 3 4"}
SEEDS=${SEEDS:-"0 1 2"}
MODELS_FED2=${MODELS_FED2-"mlp_p3000"}          # attack grid: models that learn the rare families
MODELS_FED6=${MODELS_FED6-"mlp mlp_p3000"}      # zero-attacker control
MODELS_FED4=${MODELS_FED4-"mlp_p3000,mlp"}      # federated vs local vs centralised (comma list: one job)
LR=${LR:-0.01}
OUT=${OUT:-results/fed_tasks.txt}

d6=$([ -f fqml/data/preprocess.py ] && echo "_d6" || echo "")
lrtag=$(awk -v l="$LR" 'BEGIN { if (l + 0 != 0.01) printf "_lr%g", l }')     # same tag as run_tag() in fed6
tag="_xNumber${lrtag}${d6}"
: > "$OUT"
add() { [ -f "$1" ] && echo "skip (done): $1" >&2 || echo "$2" >> "$OUT"; }

for p in $PROTOCOLS; do
  for f in $FOLDS; do
    for s in $SEEDS; do
      st=$([ "$s" = 0 ] && echo "" || echo "_s$s")
      for m in $MODELS_FED2; do
        add "results/fed2/attacks_${p}_f${f}_${m}_s${s}${tag}.json" \
            "python -m experiments.fed2_attacks --protocol $p --fold $f --seed $s --model $m --lr $LR"
      done
      for m in $MODELS_FED6; do
        add "results/fed6/zero_attacker_${p}_f${f}_${m}_r100${st}${tag}.json" \
            "python -m experiments.fed6_zero_attacker --protocol $p --fold $f --seed $s --model $m --rounds 100 --lr $LR"
      done
    done
    [ -z "$MODELS_FED4" ] && continue
    mt=$([ "$MODELS_FED4" = "mlp,vqc" ] && echo "" || echo "_${MODELS_FED4//,/-}")
    add "results/fed4/trial_${p}_f${f}_r100${mt}${tag}.json" \
        "python -m experiments.fed4_trial --protocol $p --fold $f --rounds 100 --models $MODELS_FED4 --lr $LR"
  done
done
echo "$(wc -l < "$OUT") tasks in $OUT; submit with: sbatch --array=0-$(( $(wc -l < "$OUT") - 1 ))%8 scripts/workstation/fed_grid.sbatch"
