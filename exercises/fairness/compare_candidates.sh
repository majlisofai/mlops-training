#!/usr/bin/env bash
# Put several candidates through the lesson 04 gates and the lesson 08 gates, without promoting any.
# Each candidate is trained and registered as @candidate, then both gate sets check @candidate.
# Prints one line per candidate. Nothing moves @champion.
#
# Usage:
#   bash exercises/fairness/compare_candidates.sh
#   bash exercises/fairness/compare_candidates.sh exercises/quality-gates/examples/logreg.yaml
#   FAIRNESS_GATES=exercises/fairness/solution/test_fairness.py bash exercises/fairness/compare_candidates.sh
set -euo pipefail

FAIRNESS_GATES=${FAIRNESS_GATES:-exercises/fairness/test_fairness.py}
export MLFLOW_TRACKING_URI=${MLFLOW_TRACKING_URI:-http://127.0.0.1:5000}
if [ $# -eq 0 ]; then
  set -- exercises/quality-gates/candidate.yaml \
         exercises/quality-gates/examples/logreg.yaml \
         exercises/quality-gates/examples/stumps.yaml \
         exercises/quality-gates/examples/weak.yaml \
         exercises/quality-gates/examples/leaky.yaml \
         exercises/fairness/examples/big_leaves.yaml
fi
OUT=$(mktemp -d)

for candidate in "$@"; do
  name=$(basename "$candidate" .yaml)
  python exercises/quality-gates/train_candidate.py --candidate "$candidate" > "$OUT/$name.train.log" 2>&1
  GATE_REPORT="$OUT/$name.lab04.json" python -m pytest exercises/quality-gates/solution/test_gates.py \
    -q -p no:cacheprovider > /dev/null || true
  FAIRNESS_GATE_REPORT="$OUT/$name.fair.json" MODEL_URI=models:/f1-points-finish@candidate \
    python -m pytest "$FAIRNESS_GATES" exercises/fairness/test_privacy.py -q -p no:cacheprovider -p no:warnings \
    > /dev/null || true
done

python - "$OUT" "$@" <<'EOF'
import json, sys
from pathlib import Path
out = Path(sys.argv[1])
print(f"{'candidate':<14}{'version':>8}{'lesson 04':>11}{'fairness':>10}{'fpr_gap':>9}{'fnr_gap':>9}  failed gates")
for path in sys.argv[2:]:
    name = Path(path).stem
    lab04 = json.loads((out / f"{name}.lab04.json").read_text())
    fair = json.loads((out / f"{name}.fair.json").read_text())
    measured = {k: v for g in fair["gates"] for k, v in g["measured"].items()}
    failed = [g["gate"] for g in lab04["gates"] + fair["gates"] if g["outcome"] in ("failed", "error")]
    verdict = lambda r: "pass" if r["all_passed"] else "BLOCKED"
    print(f"{name:<14}{lab04['version']:>8}{verdict(lab04):>11}{verdict(fair):>10}"
          f"{measured.get('fpr_gap', '-'):>9}{measured.get('fnr_gap', '-'):>9}  {', '.join(failed) or '-'}")
EOF
