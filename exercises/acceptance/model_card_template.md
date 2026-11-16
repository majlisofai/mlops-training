# Model card: f1-points-finish

<!-- Copy this file to exercises/acceptance/evidence/model_card.md and replace every TODO.
     accept.py refuses a card that still says TODO, or that describes another model version.
     If you made a model card in lesson 08, use that one instead, and add the two lines
     "Model name" and "Model version" below to it. -->

**Model name:** f1-points-finish
**Model version:** TODO (the version behind @champion, from `curl -s http://127.0.0.1:8000/health`)
**Owner:** TODO (release.yaml, owners.model)
**Date:** TODO

## What the model does

TODO: one or two sentences. What goes in, what comes out, and when it is called.

## Intended use and users

TODO: who uses the predictions, and for what decision.

## Out of scope

TODO: at least two uses the model must not be used for. (Read f1/data/README.md, "What the data is not".)

## Training and evaluation data

TODO: which file, which seasons for training and for the holdout, and the data version (md5).

## Performance

TODO: accuracy, ROC AUC and the margin over the grid rule on 2025, from the lesson 04 gate report.

## Subgroups

TODO: the results of subgroup_check.py, and which subgroup the model serves worst.

## Limitations and risks

TODO: at least two. Think of rookies, wet races, 2026's new regulations, and the serving check of lesson 05.

## Security and integrity

TODO: how the API is protected (lesson 09), and where the signed manifest of this version is.
