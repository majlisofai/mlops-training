# Evidence for the release dossier

`accept.py` looks here for the evidence that other lessons produce. The paths are set in [`../release.yaml`](../release.yaml) (`evidence`). A missing file is **EVIDENCE MISSING**, and the decision is **do not ship**.

| File | Where it comes from | What accept.py checks |
|---|---|---|
| `model_card.md` | You write it from [`../model_card_template.md`](../model_card_template.md) (step 7), or you copy your lesson 08 model card | No `TODO` left, and a line `**Model version:** N` with the version under acceptance |
| `fairness_report.json` | `python exercises/acceptance/subgroup_check.py` (step 7), or the lesson 08 fairness report | JSON with `model_version` (the version under acceptance) and `passed` (true or false); `checks` is shown in the dossier |

Commit the model card: it is part of the release. `fairness_report.json` is made again for each release, so git ignores it.
