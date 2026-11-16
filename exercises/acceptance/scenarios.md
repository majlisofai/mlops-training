# Scenario check: release decisions

Individual check at the end of lesson 09. Work alone, without your codespace: about 2 minutes per card.

For each card, write down:

1. **Decision:** ship, or do not ship.
2. **Evidence:** which gate or which line of the dossier decides it.
3. **Next action:** what you do now, and who you tell (use the owners in `release.yaml`).

The answer key is in [`solution/scenarios.md`](solution/scenarios.md). Do not open it before you have written your answers.

---

## Card 1: the reload endpoint is public

A colleague deployed the API on Friday evening. On Monday you run the acceptance suite against it. The dossier says:

```
| 6 | Access tests | 09 | FAIL | 4 passed, 4 failed, 0 not written; status 200: /v1/predict answers without a key;
                                 status 200: /admin/reload answers without a key; ... |
| 8 | Dependency audit | 09 | FAIL | pip-audit, image lab09-unsecured:1: 57 packages, 6 open findings |
```

All the other gates pass, and the container log shows three `POST /admin/reload` calls over the weekend, from addresses you do not know.

---

## Card 2: the model file hash changed

Nobody retrained or promoted anything since the sign-off. The dossier says:

```
| 2 | Integrity: signed SHA-256 manifest | 09 | FAIL | model/model.skops: CHANGED |
| 3 | Quality gates | 04 | PASS | 7 of 7 gates passed on version 1: accuracy 0.842, roc_auc 0.905, ... |
| 7 | Load test and SLO | 05 | PASS | SLO MET: requests 254, p95 latency 33 ms, failure rate 0.00% |
```

At the sign-off, ROC AUC was 0.907 and the p95 latency was 75 ms. A team member says: "Every other gate passed and it is even faster now. Let us sign the manifest again and ship."

---

## Card 3: p95 doubled after promotion

Version 3 replaced version 2 yesterday. Version 3 is the same forest, retrained on the same data with a newer scikit-learn. Every gate passed. The SLO line in the dossier moved like this:

```
version 2:  SLO MET: requests 247, p95 latency  75 ms, failure rate 0.00%
version 3:  SLO MET: requests 245, p95 latency 150 ms, failure rate 0.00%
```

The objective in `slo.yaml` is a p95 latency of at most 500 ms.

---

## Card 4: the data md5 changed

The dossier says:

```
| 1 | Data validation and lineage | 02, 04 | FAIL | 0 failure cases; md5 1885fafd92e1 but the model was trained on 199d7bda0a7b |
| 2 | Integrity: signed SHA-256 manifest | 09 | FAIL | f1/data/races.csv: CHANGED |
```

`git diff f1/data/races.csv` shows one changed line: `driver_form` of one 2024 entry went from `8.0` to `8.01`. The data owner says it is a correction from the timing provider.

---

## Card 5: everything passed, but two lines say EVIDENCE MISSING

```
| 9  | Model card | 08, 09 | EVIDENCE MISSING | no model card at exercises/acceptance/evidence/model_card.md |
| 10 | Subgroup and fairness evidence | 08 | EVIDENCE MISSING | no fairness report at exercises/acceptance/evidence/fairness_report.json |
```

The other eight gates pass. The product owner wants the release today: "the model is the same as in the last release, which had a model card".

---

## Card 6: a key in the repository

During a review you see that a commit from last week added this line to `exercises/serve-model/slo.yaml`:

```
# api_key: 5kyj...4E   (for the load test)
```

The acceptance suite says `Access tests: FAIL`, with `a key is in a file that git tracks: ['exercises/serve-model/slo.yaml']`. The developer offers to delete the line in a new commit.

---

## Card 7: the vulnerability that "does not matter"

The dependency audit fails on the serving image:

```
| 8 | Dependency audit | 09 | FAIL | pip-audit, image lab09-api:1: 57 packages, 6 open findings; pip PYSEC-2026-1795; ... |
```

All six findings are in `pip`, which came with the Python base image. The service owner says: "The API never runs pip. Add the six findings to `accepted` in `release.yaml` and ship."

---

## Card 8: approved by the author

Every gate passed. `decide.py` set the decision to **ship**, and the dossier says:

```
| Set by | Sam |
| Reason | every gate passed; I reviewed the dossier (WARNING: the approver is also the model owner: no separation of duties) |
```

Sam trained the model, wrote the model card and is the only person in `approvers`.
