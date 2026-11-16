# Model card: f1-points-finish

<!-- Reference solution for exercises/acceptance/model_card_template.md. The numbers are for
     version 1, trained on the course data in a new codespace (lessons 04 and 05). -->

**Model name:** f1-points-finish
**Model version:** 1
**Owner:** the trainee who proposed it (release.yaml, owners.model)
**Date:** the day of lesson 09

## What the model does

Before each race it predicts, for each of the 20 cars, whether the car finishes in the top ten and scores points (1) or not (0), with a probability. It is a scikit-learn pipeline: one-hot encoding of `circuit_type`, then a random forest of 300 trees, depth 8, at least 5 rows per leaf. It is served by the API in `exercises/acceptance/app.py` at `POST /v1/predict`.

## Intended use and users

Race engineers in the training exercise use the predictions on a dashboard to see which cars are likely to score, before the race starts. The prediction informs a person; it does not decide anything by itself.

## Out of scope

- Betting, team or driver decisions, or any claim about a real person, team or season. The data is synthetic (f1/data/README.md).
- Predictions after the race has started, or with any after-race input. The API refuses those inputs.
- The 2026 season without retraining and monitoring: the new regulations changed the DNF rate from 10.9 % to 27.5 %.

## Training and evaluation data

`f1/data/races.csv`, data version (md5) `199d7bda0a7b641e0e68ccee5236a3a7`. Training: seasons 2021 to 2024 (1,920 entries). Holdout: season 2025 (480 entries), never seen in training. Inputs: `grid`, `quali_gap_ms`, `driver_form`, `team_form`, `team_dnf_rate`, `rain_prob`, `circuit_type`, all known before the race.

## Performance

On the 2025 holdout, measured again by the lesson 04 gates: accuracy 0.842, ROC AUC 0.907, 0.037 above the grid rule (0.804). All seven gates in `criteria.yaml` passed.

## Subgroups

From `subgroup_check.py` on 2025: rookies 0.646 accuracy (48 entries), non-rookies 0.863; midfield 0.714, top teams 0.875, backmarkers 0.979. The model serves rookies worst: it predicts points for 88 % of rookie entries, where 56 % scored. Backmarker accuracy is high because the model never predicts points for them; it found none of the 3 backmarker entries that scored.

## Limitations and risks

- Rookies: 48 entries in 2025 are too few to measure the rookie accuracy precisely, and it is 0.22 below the other drivers.
- Wet races and other conditions that are rare in 2021 to 2025 are not checked separately.
- Capacity: the service-level objective (`slo.yaml`) holds for this model; a model with ten times the trees broke it in lesson 05.

## Security and integrity

The API needs a client key for predictions and an admin key for reloads (lesson 09). The keys come from the environment, never from the repository. The model files, the data and the release criteria are fingerprinted with SHA-256 in `integrity/manifest-v1.json` in the model's MLflow run, and the model version carries the tag `integrity.manifest_sha256`.
