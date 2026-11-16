# Drift Detective: answer key for the trainer

Do not show this page to the detectives until they have named the fault.

`inject.py` sends one window to the API under the name `incident`, so the monitor output does not give the fault away. Four windows carry one fault each. The fifth, `normal_weekend`, carries none: it checks that the detectives do not invent an incident from normal variation.

All values below were measured with the reference solution (`solution/monitor.py`), the `monitoring.yaml` thresholds and the lesson 04 champion (version 1: 300 trees, depth 8). The starter `monitor.py` with TODO 1 to TODO 3 completed gives the same results.

## How to run it

In your own codespace, or let pairs play against each other: one trainee injects, the partner is the detective.

```bash
python exercises/monitoring/inject.py --list            # the five cases
python exercises/monitoring/inject.py stuck_feature     # one case
python exercises/monitoring/inject.py --random          # a secret case, written to logs/incident_answer.txt
```

The API must run with `PREDICTION_LOG=logs/predictions.jsonl` (step 3 of the lesson). The detective then runs:

```bash
python exercises/monitoring/monitor.py
```

You can also run a case on GitHub Actions: **Actions → monitor → Run workflow**, choose the case under **scenario** and the solution under **monitor**.

## The five cases

| Case | What was injected | Checks that alert | The evidence that names it |
|---|---|---|---|
| `units_change` | The timing feed "fixed" the 422 errors of step 7 by sending whole seconds: `quali_gap_ms` = round(ms / 1000). Every value is a valid integer, so the API contract lets it through. | input drift, training-serving skew | `quali_gap_ms: spread x0.00106 of the reference`; mean 0.95 against 965.4. Skew: 76 of 80 values differ, `served / recomputed: median x0.000939`. A ratio of about 1/1000 is a unit change: seconds, not milliseconds. The 4 matching values are the pole-sitters (gap 0). |
| `stuck_feature` | The form job stopped and the feature store fills `driver_form` with its default, 10.0. | input drift, training-serving skew | `driver_form: spread x0 of the reference`, mean 10 against 10.63 (shift only 0.14 sd: the mean hides it). Skew: 78 of 80 differ, `served is always 10`. One value for every car is a stuck feature. |
| `category_renamed` | The circuit database renamed `high_speed` to `high-speed`. | service health, training-serving skew | `20 of 80 entries rejected (25.0%)`, `20 x circuit_type: Input should be 'street', 'permanent' or 'high_speed'`. Skew: `served 'high-speed' where recomputed is 'high_speed' (20 entries)`. Input drift passes: the 60 entries that reached the model are fine, and the race on a high-speed circuit is just missing (Evidently flags `circuit_type` and `rain_prob` for that reason). |
| `label_delay` | The 2026 window, with labels in `logs/labels_latest.csv`: the stewards' feed stalled after race 3, so races 4 to 8 have no final classification. | delayed labels | `labels late: 60 of 160 predictions have a final label (38%, need 90%)`. It also says `accuracy 0.733 on the final labels`: that is the 2026 accuracy drop of step 8, measured on only three races. Service health, input drift and skew pass. |
| `normal_weekend` | No fault: 2025 rounds 17 to 20, four ordinary races. | none: `MONITOR OK` | The largest move is `rain_prob` with a shift of 1.03 sd, under the threshold of 2.0. Evidently flags `team_dnf_rate` and `rain_prob`: with four races there are only four rain forecasts, and the K-S test calls any new set of four different. The right answer is "no incident". |

A good answer names three things: **what** changed (feature and how), **where** it most likely broke (feed, feature store, reference data, label pipeline), and **what to do** (for example: roll back the feed change, page the data team, or wait for the labels before judging the model).

## What to discuss after each case

- `units_change` versus the `serving_skew` window of step 7: the same bug, two outcomes. Floats in seconds broke the contract and every request failed loudly. Rounded seconds pass the contract and the model answers silently with wrong inputs. Which one would you rather have in production?
- `stuck_feature`: why does the mean barely move? A check on the mean alone would miss it. The spread check exists for this case.
- `category_renamed`: the API contract did its job. Who should own the list of circuit types, and how should a rename reach the model team before it reaches production?
- `label_delay`: the monitor cannot say whether the model got worse until the labels arrive. What decision can you take with 60 labels, and what must wait?
- `normal_weekend`: Evidently flags two columns. Is that a false alarm or a weak reference window? What would change with a reference of a whole season?

## Break the monitor (blind spots, measured)

Trainers who want a harder round can edit `inject.py`. These faults were measured against the same reference window, with `monitoring.yaml` unchanged:

| Fault | Input drift | Skew | Evidently |
|---|---|---|---|
| `rain_prob` stuck at 0.15 for every race | passes: shift 0.50 sd, and `rain_prob` has no spread check | alerts: 60 of 80 differ, `served is always 0.15` | flags `rain_prob` |
| `quali_gap_ms` multiplied by 0.62 (a miles-for-kilometres style factor) | passes: shift 0.60 sd, spread x0.62 (allowed x0.5 to x2.0) | alerts: 76 of 80 differ, `median x0.62` | flags `quali_gap_ms` |
| `team_dnf_rate` set to 0 for every team | alerts: spread x0 | alerts: 64 of 80 differ | flags `team_dnf_rate` |

The first two slip past the per-feature drift check and are caught only by the skew check. The skew check needs features recomputed by the training pipeline for the same requests. In production that is a batch job that runs hours later, so for those hours nothing catches the fault. Ask the detectives what they would add: a tighter spread band costs false alarms (the normal range measured in the lesson is x0.76 to x1.27), a range check per feature, or Evidently as a second opinion.
