# Lesson 08: fairness, transparency and the model card

In lessons 04 and 05 the champion passed seven quality gates, a contract test and a load test. Nobody has yet asked **for whom** it works. In this lesson you measure the champion per subgroup, write fairness and privacy requirements as gates, and watch the champion fail one of them. You try to mitigate the gap and measure what that costs, choose a different model, and document the decision in a model card that is generated from MLflow. At the end the same check runs on GitHub Actions in your fork: one run that the fairness gate blocks, and one that passes.

Time: about 100 minutes.

| Part | What you do | What it shows |
|---|---|---|
| Fairness diagnostics | Complete the fairness report and read it | Overall accuracy hides who gets the errors |
| Fairness and privacy gates | Run the gates on the champion, complete one gate | The champion passes lesson 04 but is blocked here |
| Mitigation test | Try sample weighting and per-group thresholds | Every fix has a cost, and some fixes are not allowed on the current API |
| Model choice | Put six candidates through both gate sets | An equally good model can be fairer |
| Transparency | Generate the model card from MLflow and sign it | The release decision is written down, with the evidence |
| Privacy controls | Make a prediction log, check it, try to leak a name | Data minimisation is something you can test |
| GitHub Actions | Run the check in your fork, blocked and passing | The fairness gate stops a release like any other gate |

Run every command from the repository root, `/workspaces/mlops-training`.

## What you will have at the end

- A fairness report for the champion: a table and a chart with accuracy, selection rate, false positive rate and false negative rate per subgroup, logged to the model's MLflow run.
- Six gates in `test_fairness.py` and `test_privacy.py`, one of them written by you, and a gate report.
- A mitigation test with eight runs in MLflow, and the trade-off between accuracy and the rookie gap.
- A new champion that passes the lesson 04 gates and the lesson 08 gates.
- A model card for that champion, with an owner, an approver and a decision.
- A prediction log that the privacy gate checked, and a GitHub Actions run where the fairness gate blocked a release.

## Subgroups, not protected characteristics

The data has three subgroup columns: `rookie` (first season of the driver), `team_tier` (top, midfield, backmarker) and `circuit_type`. [`f1/data/README.md`](../../f1/data/README.md) says it plainly: they exist to teach subgroup measurement; **they are not protected characteristics.** We use them to practise the method you would use for sex, age or nationality in a real system, on data where nobody is harmed.

Three rules hold for the whole lesson:

- The subgroup columns are **not model inputs**. They are used to measure the model, not to decide.
- `driver_code` and `driver_name` are synthetic, but we treat them as **personal data**. They are never model inputs and never in a prediction log.
- A **group of one is a person.** In 2025 there are two rookies, one in a top team and one in a midfield team. A result for "midfield rookies" is a scorecard of one driver, so the report suppresses it.

## The fairness and privacy requirements

The rules are in [`fairness.yaml`](fairness.yaml). A **gap** is the rookie rate minus the rate of everyone else, so a positive gap is worse for rookies. Like the lesson 04 criteria, the numbers were written before the candidates were compared, and you do not change them to make a model pass.

| Requirement | Threshold | Grid rule | Champion (forest) | Why this number |
|---|---:|---:|---:|---|
| Rookie group is large enough | ≥ 30 entries | 48 | 48 | Below 30 entries a gap means nothing; the gate fails, because it cannot check the rule. |
| Rookie false positive rate gap | ≤ 0.55 | +0.568 | **+0.620** | A false positive for a rookie is an expectation the rookie did not meet. The grid rule, which anyone can apply without a model, has a gap of 0.568. A model must not inflate rookie expectations more than that. 0.55 is that level, rounded down. |
| Rookie false negative rate gap | ≤ 0.10 | +0.113 | −0.094 | A false negative writes off a rookie who then scores. The same reasoning: no worse than the grid rule (0.113), rounded down. |
| Warnings (do not block) | < 5 drivers, or a rate on < 30 entries | | 2 drivers; FPR on 21, FNR on 27 entries | The gate still decides, but the report says the number is shaky. |
| No personal identifiers as inputs | none of `driver_code`, `driver_name` | | 7 inputs, none | Data minimisation: the model needs only what predicts a race. |
| No reporting-only columns as inputs | none of `rookie`, `team_tier` | | none | Purpose limitation: the subgroups are collected to measure, not to decide. |
| No personal data in the prediction log | no identifier as a field or a value | | 0 in 24 records | The log is kept for monitoring; it must not become a list of drivers. |

Why not an accuracy gap? Lesson 04 already checks rookie accuracy ≥ 0.60. Non-rookie accuracy is about 0.86 for every honest model, so an accuracy-gap gate of about 0.25 would be the same gate again. What lesson 04 cannot see is **which errors** rookies get. Equal false negative rates are what Mitchell et al. call *equality of opportunity*; equal false positive and false negative rates are *equalized odds* (section 4.4.1 of the model cards paper).

## The files

| File | What it does |
|---|---|
| [`fairness.yaml`](fairness.yaml) | The fairness and privacy requirements |
| [`fairness_report.py`](fairness_report.py) | Measures the champion per subgroup with fairlearn's `MetricFrame`. **You complete TODO 1 to TODO 3.** |
| [`fairness_metrics.py`](fairness_metrics.py) | Shared helpers: the four metrics, the rookie gaps, the grid rule. You do not need to change it. |
| [`test_fairness.py`](test_fairness.py) | The fairness gates. **You complete TODO 4.** |
| [`test_privacy.py`](test_privacy.py) | The privacy gates: inputs and prediction log |
| [`conftest.py`](conftest.py) | Loads the model behind `MODEL_URI` (default `@champion`) and writes the gate report `fairness_gate_report.json` and `.md` |
| [`logged_api.py`](logged_api.py) | The lesson 05 API, unchanged, plus a prediction log with an allowlist of fields |
| [`send_season.py`](send_season.py) | Sends the 24 races of 2025 to the API, like a race dashboard |
| [`mitigate.py`](mitigate.py) | The mitigation test: sample weighting and per-group thresholds, logged to MLflow |
| [`compare_candidates.sh`](compare_candidates.sh) | Puts several candidates through the lesson 04 and lesson 08 gates, without promoting any |
| [`model_card.py`](model_card.py), [`model_card.yaml`](model_card.yaml) | Generates the model card from MLflow; the yaml holds what people write and sign |
| [`examples/`](examples/) | `big_leaves.yaml`, a candidate; `prediction_log_v0.jsonl`, a log that leaks personal data |
| [`.github/workflows/fairness.yml`](../../.github/workflows/fairness.yml) | Runs the whole check on GitHub Actions |

**Why fairlearn?** `MetricFrame` computes any scikit-learn-style metric per group, the differences and ratios between groups, and bootstrap intervals, in a few lines. `fairlearn==0.15.0` installs next to the existing pins without changing any of them. Plain pandas `groupby` would do the per-group numbers too; you would write the intervals and the between-group summaries yourself.

## Step 1: start the MLflow server

Open a terminal and run the same command as in lessons 02 to 05:

```bash
mlflow server \
  --host 127.0.0.1 --port 5000 \
  --backend-store-uri sqlite:///mlflow.db \
  --artifacts-destination ./mlartifacts \
  --allowed-hosts "*.app.github.dev,localhost:*,127.0.0.1:*" \
  --cors-allowed-origins "https://*.app.github.dev"
```

Leave this terminal running. Open the **Ports** tab and click the globe icon next to **MLflow UI**. Open a second terminal (**Terminal → New Terminal**) for the other commands.

## Step 2: make sure there is a champion

This codespace is new, so its registry is empty. Run the lesson 04 pipeline with the good candidate and the reference gates:

```bash
GATES=exercises/quality-gates/solution/test_gates.py bash exercises/quality-gates/run_pipeline.sh
```

> **Check:** seven `PASSED` gates (accuracy 0.842, margin 0.037, rookie_accuracy 0.646) and
> `PROMOTED: f1-points-finish version 1 is now @champion (previous champion: none).`

The version numbers in this lesson are for a new codespace. If you registered more versions already, your numbers are higher. That is fine.

## Step 3: measure the champion per subgroup

Run the fairness report:

```bash
python exercises/fairness/fairness_report.py
```

> **Check:** `TODO 1: subgroup_frame() is not written yet.` The three measurements are yours to write.

Open [`fairness_report.py`](fairness_report.py) and complete **TODO 1 to TODO 3**:

| TODO | Function | What it returns |
|---|---|---|
| 1 | `subgroup_frame` | A fairlearn `MetricFrame` with the four metrics for every group of one column, with bootstrap intervals |
| 2 | `between_group_gaps` | For every metric, the largest difference and the ratio between two groups |
| 3 | `drivers_per_group` | How many different drivers each group holds |

Delete the `raise` line in each function when you write it. Run the report again. It takes 10 to 40 seconds, depending on the machine: the bootstrap resamples the 480 entries 500 times for each subgroup column.

> **Check:** the report starts like this, and ends with five `WARNING` lines and `Logged them to run ... under fairness/`:
>
> ```
> Fairness report: f1-points-finish version 1 (models:/f1-points-finish@champion) on 2025, 480 entries. Overall accuracy 0.842.
>
> rookie         entries drivers    acc    sel    fpr    fnr
>   False            432      18  0.863  0.500  0.142  0.131
>   True              48       2  0.646  0.875  0.762  0.037
>   difference                    0.218  0.375  0.620  0.094
>
> team_tier      entries drivers    acc    sel    fpr    fnr
>   backmarker       144       6  0.979  0.000  0.000  1.000
>   midfield         192       8  0.714  0.599  0.366  0.227
>   top              144       6  0.875  0.993  1.000  0.008
> ...
> Rookie gaps (rookie minus everyone else; positive = worse for rookies):
>                    model  grid rule   limit
>   fpr_gap         +0.620     +0.568    0.55
>   fnr_gap         -0.094     +0.113     0.1
>   accuracy_gap    +0.218     +0.315       -
> ```

Read it the way a reviewer would:

- **The rookie errors are almost all one kind.** The model predicts points for 87.5 % of rookie entries; 56 % of them scored. Of the 21 rookie entries that did not score, it predicted points for 16 (false positive rate 0.762, against 0.142 for everyone else). It hardly ever misses a rookie who scores (0.037). Lesson 04's single number, rookie accuracy 0.646, hid this.
- **The model is more over-optimistic than the grid rule.** The rule's false positive gap is 0.568; the model's is 0.620.
- **Two drivers.** All 48 rookie entries come from two people. The bootstrap interval of the rookie false positive rate is 0.56 to 0.93, and it is still too narrow: it resamples entries, but entries of the same driver are not independent.
- **Not every gap is unfair.** `team_tier` shows the largest differences: backmarker false negative rate 1.000, top false positive rate 1.000. They rest on 3 and 17 entries, and they say that backmarker cars almost never score and top cars almost always do. A gap that follows the task, not a person, is not a fairness finding. That is why `fairness.yaml` gates only the rookie comparison.

Open `fairness_report/fairness_by_group.png` (click it in the file explorer), then the MLflow UI: experiment **f1-points-finish**, the run of version 1, **Artifacts → fairness**. The report is stored next to the model it describes. `fairness_report/fairness_table.md` also holds the **rookie x team_tier** table: both rookie cells say `suppressed: 1 driver`.

## Step 4: make a prediction log

Monitoring (Day 3) needs a record of what the model was asked and what it answered. [`logged_api.py`](logged_api.py) is the lesson 05 API, unchanged, plus one middleware: every successful `POST /v1/predict` becomes one JSON line in `logs/predictions.jsonl`, with an **allowlist** of fields: time, model name and version, the seven inputs, the predictions and the probabilities. No client address, no headers, nothing else.

Start it in the second terminal (stop the lesson 05 API or container first if it still runs: it uses the same port):

```bash
uvicorn logged_api:app --app-dir exercises/fairness --port 8000
```

> **Check:** `Serving f1-points-finish version 1 (models:/f1-points-finish@champion)`, then `Application startup complete.`

In a third terminal, send the 2025 season, one race of 20 cars per request:

```bash
python exercises/fairness/send_season.py
```

> **Check:** `Sent 24 races (480 entries) to http://127.0.0.1:8000; model version 1 answered.` `wc -l logs/predictions.jsonl` says `24`.

`races.csv` holds the driver names and codes, but [`send_season.py`](send_season.py) sends only the seven inputs: minimisation starts at the client. Now send `driver_code` too, as a careless client would:

```bash
python exercises/fairness/send_season.py --with-driver-code
```

> **Check:** `2025-01: HTTP 422, 20 problems, first: entries.0.driver_code: Extra inputs are not permitted`, then `24 races were refused.` The log still has 24 lines.

The lesson 05 contract (`extra="forbid"`) is a privacy control too: an identifier never reaches the model or the log. Leave the API running.

## Step 5: run the fairness and privacy gates

Run both gate files in one pytest session, so that they write one gate report:

```bash
pytest exercises/fairness/test_fairness.py exercises/fairness/test_privacy.py -rs --tb=line
```

> **Check:** `1 failed, 4 passed, 1 skipped, 3 warnings`. The failure is
> `AssertionError: rookie false positive rate gap +0.620 is above 0.55`. The skipped gate is TODO 4. The three warnings are the small rookie group: 2 drivers, an FPR on 21 entries, an FNR on 27.

Open [`test_fairness.py`](test_fairness.py) and complete **TODO 4**, the false negative rate gate. It works like `test_rookie_fpr_gap` above it. Delete the `pytest.skip(...)` line and run the gates again.

> **Check:** `1 failed, 5 passed, 3 warnings`. `fairness_gate_report.md` shows `rookie_fnr_gap | PASS | rookie_fnr 0.037, others_fnr 0.131, fnr_gap -0.094, grid_rule_fnr_gap +0.113`, and the verdict line says **BLOCKED**.

The champion that passed every lesson 04 gate, the contract tests and the load test is blocked by a fairness gate. Look at the honesty problem in that sentence: it fails by about one rookie entry (one entry moves the rookie FPR by 1/21 = 0.048), on a measurement of two drivers. The gate still has to decide. The warnings and the model card are where you say how sure you are.

## Step 6: the mitigation test

Can you close the gap without losing accuracy? [`mitigate.py`](mitigate.py) retrains the champion configuration and tries two kinds of mitigation:

- **Sample weighting** (`reweight-xW`): rookie rows count W times in training. The model still has the seven API inputs; `rookie` is used only during training.
- **Per-group threshold** (`rookie-threshold-T`): the same model, but a rookie needs a probability of at least T to be predicted to score; everyone else 0.5. `past-rookies-T` picks T the honest way, on 2021 to 2024 only.

```bash
python exercises/fairness/mitigate.py
```

> **Check:** MLflow prints a link for each run, then the table of the eight runs:
>
> ```
> run                       acc  rookie  fpr_gap  fnr_gap  lab04  fair  API v1
> baseline                0.842   0.646   +0.620   -0.094   pass  FAIL     yes
> reweight-x2             0.840   0.667   +0.616   -0.136   pass  FAIL     yes
> reweight-x5             0.838   0.688   +0.573   -0.150   pass  FAIL     yes
> reweight-x10            0.825   0.625   +0.473   +0.026   pass  pass     yes
> rookie-threshold-0.6    0.846   0.688   +0.477   -0.057   pass  pass      no
> rookie-threshold-0.7    0.838   0.604   +0.335   +0.202   pass  FAIL      no
> rookie-threshold-0.8    0.829   0.521   +0.097   +0.535   FAIL  FAIL      no
> past-rookies-0.3        0.835   0.583   +0.811   -0.131   FAIL  FAIL      no
> ```

In the MLflow UI, open the experiment **f1-points-finish-fairness**. The parent run `mitigation-test-forest` holds `mitigation.csv` and the chart `mitigation_tradeoff.png`. Select the eight child runs, click **Compare** and plot `accuracy` against `fpr_gap`.

What the numbers say:

- **Sample weighting barely helps.** Rookie and non-rookie rows with the same inputs look the same to the model, so giving rookies more weight changes little until the weight is 10. Then the gap passes, accuracy drops from 0.842 to 0.825, just above the lesson 04 margin (0.804 + 0.02), and the false negative gap turns positive.
- **A per-group threshold trades one error for the other.** Raising the rookie threshold cuts false positives and adds false negatives: at 0.7 the FPR gap is fine and the FNR gap fails; at 0.8 rookie accuracy is 0.521, as bad as the grid rule.
- **`rookie-threshold-0.6` looks perfect, and it is not a valid result.** 0.6 was picked by looking at 2025, the holdout: that is tuning on the test set. Picked honestly on past seasons (`past-rookies`), the threshold is 0.3, and the gap gets worse (+0.811): the rookies of 2022 and 2023 scored much more often than the two of 2025. Rookie cohorts of two drivers do not resemble each other.

Is a per-group threshold even allowed? It treats two entries with the same inputs differently because of group membership, and it needs `rookie` at prediction time.

- For a **protected characteristic** (sex, nationality, religion, ...), using it at decision time is direct differential treatment. Many legal systems restrict or forbid it, and SDAIA's AI Ethics Principles ask that sensitive attributes are not used as model data unless the use is justified (Principle 1, Prepare Input Data, item 2). This course gives no legal advice: ask your legal and compliance team before you build one.
- For `rookie`, experience is relevant to racing and not protected, so the method can be legitimate. It still breaks two of our own rules: API v1 does not accept `rookie` (the **API v1** column says `no`: it needs a `/v2/` contract), and `fairness.yaml` lists `rookie` as reporting-only. Changing either is a decision for the model owner and the approver, written in the model card, not a code change.
- Sample weighting uses the group only in training. That is usually easier to justify, and it changes nothing at the API. Here it costs accuracy.

## Step 7: choose a fairer model

The cheapest mitigation can be a different model. Put the lesson 04 candidates, the champion and one new candidate through both gate sets, without promoting any:

```bash
FAIRNESS_GATES=exercises/fairness/solution/test_fairness.py bash exercises/fairness/compare_candidates.sh
```

(Use your own `test_fairness.py` instead of the solution if you completed TODO 4.) It takes about 45 seconds and registers six new versions as `@candidate`.

> **Check:**
>
> ```
> candidate      version  lesson 04  fairness  fpr_gap  fnr_gap  failed gates
> candidate            2       pass   BLOCKED   +0.620   -0.094  rookie_fpr_gap
> logreg               3       pass      pass   +0.487   -0.007  -
> stumps               4    BLOCKED   BLOCKED   +0.577   +0.035  rookie_accuracy, rookie_fpr_gap
> weak                 5    BLOCKED   BLOCKED   +0.332   +0.104  accuracy, roc_auc, beats_grid_rule, rookie_fnr_gap
> leaky                6    BLOCKED      pass   +0.000   +0.000  no_forbidden_features
> big_leaves           7       pass      pass   +0.525   -0.146  -
> ```

Three lessons in one table:

- **`logreg` and `big_leaves` pass everything.** `big_leaves` (50 trees, leaves of at least 50 entries) is as accurate as the champion (0.842) and is right for 34 of 48 rookie entries instead of 31.
- **`weak` shows the trade-off again.** It is the least over-optimistic model for rookies (FPR gap +0.332), and it writes off scoring rookies a little more often than allowed (FNR gap +0.104, limit 0.10). It also fails three quality gates.
- **`leaky` is perfectly "fair"**: it reads the result of the race, so it makes no errors for anyone. It is still blocked, by lesson 04's forbidden-features gate. A fairness gate is not a quality gate: you need both gate sets.

Choosing among models after looking at 2025 is a mild form of tuning on the holdout too. The lesson 04 gates were written before; the choice between two passing models is a judgement that belongs in the model card. Promote the logistic regression, the lesson 04 reference model, with the lesson 04 pipeline:

```bash
GATES=exercises/quality-gates/solution/test_gates.py bash exercises/quality-gates/run_pipeline.sh exercises/quality-gates/examples/logreg.yaml
```

> **Check:** seven `PASSED` gates (accuracy 0.835, margin 0.031, rookie_accuracy 0.646), then `PROMOTED: f1-points-finish version 8 is now @champion (previous champion: 1).`

Tell the running API to load the new champion, make a new prediction log and run the report and the gates again:

```bash
curl -s -X POST http://127.0.0.1:8000/admin/reload
rm logs/predictions.jsonl && python exercises/fairness/send_season.py
python exercises/fairness/fairness_report.py
pytest exercises/fairness/test_fairness.py exercises/fairness/test_privacy.py -rs --tb=line
```

> **Check:** `"previous_version":"1","model_version":"8"`; `model version 8 answered`; the report shows rookie FPR 0.619 against 0.132 (`fpr_gap +0.487`, `fnr_gap -0.007`); the gates end with `6 passed, 3 warnings`.

The warnings stay: the logistic regression is measured on the same two rookies.

## Step 8: write the model card

A model card is a short document that travels with a model: what it is for, what it is not for, how it was evaluated, and how it performs for the groups that matter (Mitchell et al., 2019). [`model_card.py`](model_card.py) builds it from what MLflow already knows, so nobody types a number by hand:

| Model card section | Where `model_card.py` gets it |
|---|---|
| Model details | Registry: version, aliases, `promoted_by`, previous champion. Run: model type, hyperparameters, inputs, data version, git commit |
| Intended use, out-of-scope uses, factors | `model_card.yaml`: written by people |
| Metrics, evaluation data, training data | Fixed text, the run's parameters and `races.csv` |
| Quantitative analyses | The run's metrics, the fairness report in the run (`fairness/fairness_report.json`), the lesson 04 gate tags and the fairness gate report |
| Ethical considerations, caveats | `model_card.yaml`, plus the warnings from the fairness report |
| Accountability and sign-off | `model_card.yaml`: owner, approver, decision. The course adds this section for the release dossier. |

Generate it:

```bash
python exercises/fairness/model_card.py
```

> **Check:** `Wrote fairness_report/model_card.md for f1-points-finish version 8.`, `Fairness and privacy gates: passed. Placeholders still to fill in model_card.yaml: 7.`, and `Logged it to run ... under model_card/ and tagged version 8 with model_card.`

Open `fairness_report/model_card.md` (right-click → **Open Preview**). It starts with **Release status** and a warning: **Not ready for review**, with the list of placeholders. Open [`model_card.yaml`](model_card.yaml) and fill in every `<...>`: an owner, an approver (a colleague in your group), a contact, the decision and its date, what your team decided about the rookie gap, and one more caveat. Generate the card again.

> **Check:** `Placeholders still to fill in model_card.yaml: 0.` The **Release status** table shows your decision and approver. In the MLflow UI, version 8 has the tags `model_card` and `fairness_gates = passed`, and its run has the card under **Artifacts → model_card**.

Read the card as a reviewer who was not in the room. Can you tell from it why the forest was replaced, and what the rookie numbers are worth?

## Step 9: test the privacy controls

The privacy gate passed on your log. Make sure it can fail. [`examples/prediction_log_v0.jsonl`](examples/prediction_log_v0.jsonl) is a log from an imaginary first version of the API that logged the whole request and a dashboard label:

```bash
PREDICTION_LOG=exercises/fairness/examples/prediction_log_v0.jsonl FAIRNESS_GATE_REPORT=fairness_report/privacy_v0.json \
  pytest exercises/fairness/test_privacy.py -rs --tb=line
```

> **Check:** `1 failed, 2 passed`, with `AssertionError: 6 personal identifiers in exercises/fairness/examples/prediction_log_v0.jsonl, first: line 1: field request.driver_code; line 1: dashboard_label = 'Amara Okafor (P1)'; line 2: field request.driver_code`.

The gate found both kinds of leak: a field called `driver_code`, and a driver's name hidden inside a free-text value. `FAIRNESS_GATE_REPORT` sends this experiment's report to another file, so that `fairness_gate_report.json` still describes version 8.

Now look at the terminal where `logged_api.py` runs. Every request shows a line like `INFO: 127.0.0.1:49687 - "POST /v1/predict HTTP/1.1" 200 OK`. That is a second log, written by uvicorn, and it holds the client's IP address, which in many legal systems is personal data. Our allowlist does not cover it. `uvicorn --no-access-log` turns it off; a real service decides what to keep, for how long, and who may read it.

## Privacy controls in this course

| Control | Where | What it prevents |
|---|---|---|
| Identifiers are not model inputs | `test_no_personal_identifiers_in_model_inputs`, and the lesson 04 feature list | Decisions based on who a person is |
| Subgroup columns are reporting-only | `test_no_reporting_only_columns_in_model_inputs` | Using data collected to measure fairness for something else |
| The API refuses unknown fields | lesson 05 contract, `extra="forbid"` | Identifiers reaching the model or the log by accident |
| The client sends only the inputs | `send_season.py` | Personal data leaving the place where it is needed |
| The prediction log keeps an allowlist | `logged_api.py` | A monitoring log that becomes a register of drivers |
| The log is tested | `test_no_personal_data_in_prediction_log` | A future change that starts logging names unnoticed |
| Small groups are suppressed in reports | `min_drivers_per_published_cell` in `fairness.yaml` | A "group" result that is one person's scorecard |
| Synthetic data | `f1/data/` | Training on real people at all |

What the course does **not** do, and a real system must decide: how long the prediction log is kept, who may read it, the web server's access log, access control on the MLflow server, and a privacy impact assessment.

## The controls and the sources

The course maps its controls to public frameworks. This table cites only what the sources say in the passages named.

| Control in this lesson | SDAIA AI Ethics Principles (2025) | NIST AI RMF 1.0 |
|---|---|---|
| Fairness metrics and thresholds written in `fairness.yaml` before the candidates | Principle 1 Fairness, Plan and Design, items 3 and 4: choose the fairness metrics and define the allowed threshold at this stage (p. 13) | MEASURE 2.11: fairness and bias are evaluated and results documented |
| The fairness gate on the champion, and the written justification when it fails | Principle 1, Build and Validate, item 2: the champion model's fairness metrics should be within the defined threshold; the assessment should be justified if the champion does not pass (p. 14) | MEASURE 2.11 |
| Group size and warnings | Principle 1, Prepare Input Data, item 1: the sample size has a significant impact on accuracy and fairness (p. 14) | Section 3.7: statistical biases often stem from non-representative samples |
| No identifiers or subgroup columns as inputs | Principle 1, Prepare Input Data, item 2 (sensitive attributes not in the model data, or justified); Principle 2 Privacy and Security, Plan and Design, items 1 and 2: decisions not based on personally identifying information, personal information limited to what is necessary (pp. 14, 16) | MEASURE 2.10: privacy risk examined and documented; section 3.6: data minimising methods |
| Prediction log allowlist and test | Principle 2, Plan and Design, item 2 (p. 16) | MEASURE 2.10 |
| Small-cell suppression | Principle 2, Build and Validate, item 3: test that the combination of available data does not reveal sensitive data or break anonymity (p. 18) | Section 3.6: aggregation of certain model outputs |
| The model card | Principle 6 Transparency and Explainability: document data sets and processes (Prepare Input Data, item 1); performance and measured impact documented and accessible (Deploy and Monitor, item 1) (p. 25). Annexure A: AI Fairness Position Statement (p. 33) | MAP 1.1 intended purposes and context documented; MAP 3.3 application scope documented; MEASURE 2.9 model explained, validated and documented |
| Owner, approver and decision in the card | Principle 7 Accountability and Responsibility, Build and Validate, items 2 to 4: assign ownership; owners review and sign off before production (p. 27) | GOVERN 2.1: roles and responsibilities documented |
| Monitoring the fairness metrics later | Principle 1, Deploy and Monitor, item 2: monitor the predefined fairness metrics in production (p. 15) | |

Sources:

- M. Mitchell, S. Wu, A. Zaldivar, P. Barnes, L. Vasserman, B. Hutchinson, E. Spitzer, I. D. Raji, T. Gebru. *Model Cards for Model Reporting.* FAT* 2019. [arXiv:1810.03993](https://arxiv.org/abs/1810.03993). Section 4 lists the card sections; section 4.4 the fairness definitions; section 4.7 asks for disaggregated results with confidence intervals.
- Saudi Data and Artificial Intelligence Authority (SDAIA). *AI Ethics Principles*, version 1, May 2025, document SDAIA-P114E. [sdaia.gov.sa](https://sdaia.gov.sa/en/SDAIA/about/Documents/ai-principles.pdf).
- NIST. *Artificial Intelligence Risk Management Framework (AI RMF 1.0)*, NIST AI 100-1, January 2023. [doi.org/10.6028/NIST.AI.100-1](https://doi.org/10.6028/NIST.AI.100-1).
- Fairlearn, `MetricFrame`: [fairlearn.org, user guide: Assessment](https://fairlearn.org/main/user_guide/assessment/index.html).

## Step 10: run the fairness check on GitHub Actions

The workflow [`fairness.yml`](../../.github/workflows/fairness.yml) runs the lesson in one job:

| Step | What it does |
|---|---|
| Start the MLflow server | An MLflow server for this run only |
| 1. Promote the candidate | The lesson 04 pipeline with the reference gates |
| 2. Make a prediction log | `logged_api.py` and `send_season.py`; checks that `--with-driver-code` is refused |
| 3. Fairness report | The reference `solution/fairness_report.py` |
| 4. Fairness and privacy gates | **Your** `test_fairness.py`, and `test_privacy.py`. The job fails here when a gate fails. |
| 5. Model card | `model_card.py`, also when the gates failed: a blocked release needs a card too |

It writes the gate report and the fairness table to the run page and keeps the card, the chart, the table, the gate reports and the prediction log as the artifact **fairness-dossier**.

1. **Commit your work** (the TODOs and your `model_card.yaml`):

   ```bash
   git add exercises/fairness
   git commit -m "Fairness report, FNR gate and model card"
   ```

2. **Push to your fork** (**Source Control → Sync Changes**, or **Publish Branch**). If this is a new fork, enable its workflows in the **Actions** tab, as in lesson 04 step 5.
3. **Make `08-fairness` the default branch of your fork** (**Settings → General → Default branch**), so that GitHub shows the **Run workflow** button.
4. Your push changed `exercises/fairness/`, so it started the workflow **fairness** with the default candidate, the forest.

> **Check:** the job **Fairness, privacy and model card** fails at step **4. Fairness and privacy gates** with `AssertionError: rookie false positive rate gap +0.620 is above 0.55`. Step **5. Model card** still runs. The run page shows the gate table with the verdict **BLOCKED**, and the artifact **fairness-dossier** holds `fairness_report/model_card.md`. We measured about two minutes.

Now the fairer model. Open **Actions → fairness → Run workflow**, set the candidate to `exercises/quality-gates/examples/logreg.yaml` and click **Run workflow**.

> **Check:** the job succeeds. The gate table says **passed** (`fpr_gap +0.487`). With the TODO 4 gate of the starter file it shows `rookie_fnr_gap` as **NOT CHECKED**; with yours, as PASS with `fnr_gap -0.007`.

## Discuss

1. The champion fails the FPR gate by about one rookie entry, measured on two drivers. Would you block the release, approve it with conditions, or change the threshold? Who should decide, and where is that decision written?
2. We gate on the error that harms rookies (positive gaps) and do not block a model that is better for rookies. Is that fair to everyone else? When would you gate on the absolute gap instead?
3. A per-group threshold fixed the numbers on 2025 but needs `rookie` at prediction time. For a protected characteristic, which of the three mitigations would you even consider, and whom would you ask?
4. The leaky model passed every fairness gate. What does a fairness gate prove, and what does it not prove?
5. The prediction log is needed to monitor the model, and it is a privacy risk. What would you keep, for how long, and who may read it?

## Stretch

- Gate the champion before it becomes champion: in a copy of `release.yml`, run the fairness report and the fairness gates on `@candidate` (set `MODEL_URI=models:/f1-points-finish@candidate`) in the job **Quality gates**, before **Promote to champion**.
- Intersectional analysis: change the report to measure `circuit_type` x `team_tier`. Which cells does the suppression rule hide, and which ones are just small?
- Try fairlearn's own mitigation, `fairlearn.postprocessing.ThresholdOptimizer` with `constraints="false_negative_rate_parity"` or `"equalized_odds"`, and `sensitive_features=train["rookie"]`. Compare it with `mitigate.py`. What does it need at prediction time?
- Bootstrap by driver, not by entry: resample whole drivers and compute the interval of the rookie FPR gap. How wide is it with two rookies?
- Write the "AI Fairness Position Statement" that SDAIA's Annexure A describes (p. 33) for this model: the fairness criteria, the rationale, in non-technical language, in half a page.

## Stuck?

There are reference solutions in [`solution/fairness_report.py`](solution/fairness_report.py) and [`solution/test_fairness.py`](solution/test_fairness.py). Try the TODOs first. To run them:

```bash
python exercises/fairness/solution/fairness_report.py
pytest exercises/fairness/solution/test_fairness.py exercises/fairness/test_privacy.py -rs --tb=line
```

Run one `test_fairness.py` at a time. pytest cannot collect `test_fairness.py` and `solution/test_fairness.py` in one run, because the two files have the same name.
