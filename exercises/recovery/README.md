# Lesson 07: recover the service and train a controlled challenger

In lesson 06 you watched the service: input drift, training-serving skew and labels that arrive late. You saw that the champion gets fewer races right in 2026. In this lesson somebody tries to fix that over a weekend, and the service goes down. You collect the evidence, decide with a runbook, recover the service and prove it with the lesson 05 checks. Then you do properly what the weekend hotfix tried to do: you train a challenger on newer data, in time order, compare it with the champion on the same held-out races, and let it become champion only through the lesson 04 gates and an approval.

Time: about 150 minutes.

| Part | What you do | What it shows |
|---|---|---|
| Know your normal | Collect the evidence while the service is healthy | You can only see what changed if you know what it looked like before |
| Incident and triage | Your trainer breaks the service. You answer the five runbook questions. | A diagnosis comes from evidence: health, version, logs, inputs, delayed labels |
| AI-assisted diagnosis | Give the evidence to an AI assistant and check its answer | An assistant is a fast second opinion. A person decides and is named. |
| Recovery | Roll back, then prove the recovery with the lesson 05 checks | The incident ends when the checks pass, not when you typed the fix |
| Controlled challenger | Train a challenger on 2021-2025 and early 2026, compare it on later 2026 races | A fair comparison uses the same held-out races, in time order, and is honest about how few they are |
| Gated promotion | Promote through `promote.py`, or watch it refuse | The only way to `@champion` is through the gates and a person |
| GitHub Actions | Run the comparison in your fork | CI shows when a challenger wins and when it is blocked |

Run every command from the repository root, `/workspaces/mlops-training`.

## What you will have at the end

- A service that went down, and a recovery that you proved with measurements.
- An incident report in `exercises/recovery/incident_report.md`, with a timeline, the evidence, the root cause and actions.
- A challenger trained with your code (TODO 1 to TODO 3 in `train_challenger.py`), and a comparison with the champion on 2026 races that neither model saw.
- A promotion through the lesson 04 path, and proposals that the same path refused.
- A GitHub Actions run where the challenger wins, and a run where it is blocked.

## The files

| File | What it does |
|---|---|
| [`RUNBOOK.md`](RUNBOOK.md) | What to do in an incident: triage, decision table, recovery checks, the AI assistant, the incident report template |
| [`triage.py`](triage.py) | Collects the evidence for the five runbook questions. It changes nothing. |
| [`incident.py`](incident.py) | Breaks the running service. **Your trainer runs it. Do not read it before step 4.** |
| [`train_challenger.py`](train_challenger.py) | Trains a challenger and registers it as `@challenger`. **You complete TODO 1 to TODO 3.** |
| [`challenger.yaml`](challenger.yaml) | The challenger you propose: the same model as the champion, newer training data |
| [`comparison.yaml`](comparison.yaml) | The test window (2026 rounds 5 to 8) and the rules a challenger must meet |
| [`compare.py`](compare.py) | Measures champion and challenger on the test window, runs the challenger gates, writes the lesson 04 gate report |
| [`examples/`](examples/) | Three other proposals: `all_history.yaml`, `heavy_2026.yaml`, `peek.yaml` |
| [`.github/workflows/challenger.yml`](../../.github/workflows/challenger.yml) | Runs the comparison on GitHub Actions |

## Step 1: start the service

Open a terminal and start the MLflow server, as in lessons 02 to 05:

```bash
mlflow server \
  --host 127.0.0.1 --port 5000 \
  --backend-store-uri sqlite:///mlflow.db \
  --artifacts-destination ./mlartifacts \
  --allowed-hosts "*.app.github.dev,localhost:*,127.0.0.1:*" \
  --cors-allowed-origins "https://*.app.github.dev"
```

Leave it running. Open a second terminal (**Terminal → New Terminal**). Make the lesson 04 champion:

```bash
GATES=exercises/quality-gates/solution/test_gates.py bash exercises/quality-gates/run_pipeline.sh
```

> **Check:** seven `PASSED` gates (accuracy 0.842, roc_auc 0.907, rookie_accuracy 0.646), then `PROMOTED: f1-points-finish version 1 is now @champion (previous champion: none).`

Build the lesson 05 image and run the API as a service, in the background, with a restart policy:

```bash
docker build -t f1-api:1 exercises/serve-model
docker run -d --name f1-api --network host --restart on-failure f1-api:1
```

`-d` runs the container in the background. `--restart on-failure` makes Docker start the API again whenever it stops with an error, the way a production platform does. The lesson 05 command used `--rm` instead; the two cannot go together.

> **Check:** `docker logs f1-api` shows `Serving f1-points-finish version 1 (models:/f1-points-finish@champion)`, and `curl -s http://127.0.0.1:8000/health` answers `"status":"ok"` and `"model_version":"1"`.

The version numbers in this lesson are for a new codespace. If you registered more versions already, your numbers are higher. That is fine.

> **Not in a codespace?** On Docker Desktop (Mac or Windows) use `docker run -d --name f1-api -p 8000:8000 -e MLFLOW_TRACKING_URI=http://host.docker.internal:5000 --restart on-failure f1-api:1`, and add `host.docker.internal:*` to `--allowed-hosts`, as in lesson 05.

## Step 2: know your normal

Open [`RUNBOOK.md`](RUNBOOK.md) and read part 1. Then collect the evidence while everything is healthy:

```bash
python exercises/recovery/triage.py all
```

> **Check:** the four parts say:
>
> - **Health:** `UP: model f1-points-finish version 1 (alias champion)`, and the container is `running, 0 restarts`.
> - **Registry:** version 1 has `@candidate @champion`, `promoted by <you>`, `gates: 7 passed`, `inputs: 7, the API v1 inputs`.
> - **Inputs:** `160 of 160 entries valid`. One input `MOVED`: `team_dnf_rate`, mean 0.109 in the history and 0.207 in 2026 (shift +0.98). `did not finish: history 10.9%, 2026 27.5%`. `6 changed from the provisional result`.
> - **Delayed labels:** accuracy 0.842 on the 2025 holdout, and `2026 overall: accuracy 0.706 on 160 entries with final labels`. Per race it is between 0.60 and 0.80.

This is the normal of the service, and it is already not good: the champion gets 0.706 of the 2026 entries right, against 0.842 in 2025. The inputs keep the contract, but the cars break down more than twice as often. That is the lesson 06 finding. Keep this output: you compare with it in step 4.

## Step 3: the incident

Your trainer runs the incident, or tells you to run it. **Do not open `incident.py` before you have written your diagnosis.**

```bash
python exercises/recovery/incident.py
```

> **Check:** after about five seconds, a pager message:
>
> ```
> {
>   "pager": "f1-api health check FAILING",
>   "since": "...",
>   "report": "Race engineers: the dashboard shows no predictions for the next race.",
>   "next": "Follow exercises/recovery/RUNBOOK.md. Collect evidence before you change anything."
> }
> ```

Write down the time. Copy the incident report template from part 5 of the runbook into `exercises/recovery/incident_report.md`. From now on, note every observation and action with its time.

## Step 4: triage

Follow part 1 of the runbook. Answer the five questions in order and write the answers in the **Evidence** table of your report. Do not change anything yet.

```bash
curl -s http://127.0.0.1:8000/health
docker ps -a --filter name=f1-api
docker logs --timestamps f1-api 2>&1 | tail -40
python exercises/recovery/triage.py all --out incident_evidence.md
```

> **Check:** the evidence we collected looked like this. Yours has other times and another restart count.
>
> - **Health:** `DOWN: no answer: Connection refused`. The container is `restarting`, with a growing number of restarts and `last exit code 3`.
> - **Logs:** `Serving f1-points-finish version 1`, then `"POST /admin/reload HTTP/1.1" 500`, then, again at every restart, `RuntimeError: f1-points-finish version 2 needs the inputs [..., 'laps_completed', ...], but API v1 accepts [...]` and `Application startup failed. Exiting.`
> - **Registry:** version 2 has `@champion`. Run `hotfix-2026-dnf`, `proposed by weekend-hotfix`, `promoted by hotfix-script`, `previous champion 1`, `gates: not run`. `inputs: 8, NOT the API v1 inputs: extra ['laps_completed'], after-race ['laps_completed']`. Run metric `accuracy_2026 0.931`.
> - **Inputs:** the same as in step 2: `160 of 160 entries valid`, only `team_dnf_rate` moved.
> - **Delayed labels:** `The API does not answer, so nobody can measure what it serves.`

The API is down, so measure the two versions offline:

```bash
python exercises/recovery/triage.py labels --model-uri models:/f1-points-finish/2
python exercises/recovery/triage.py labels --model-uri models:/f1-points-finish/1
```

> **Check:** version 2: `Cannot evaluate models:/f1-points-finish/2: it needs the inputs [...]. ... Missing: ['laps_completed'].` Version 1: accuracy 0.842 on 2025 and 0.706 on 2026, the same as in step 2.

Open the MLflow UI too (**Models → f1-points-finish → Version 2**, then its run). The run has a note, and its parameters say which data it was trained on.

Now write your diagnosis in the report: what happened, in what order, and why the service is down. Answer three questions: Why did `/admin/reload` not take the service down at once? What did the restart change? What does `accuracy_2026 0.931` measure, and why is it not evidence that version 2 is good?

## Step 5: ask an AI assistant

Follow part 4 of the runbook. Open GitHub Copilot Chat in the codespace, or another assistant you are allowed to use. Give it `incident_evidence.md` and the prompt from the runbook.

Compare its three hypotheses with your diagnosis. For each line it quotes, find the line in `incident_evidence.md`. In your report, under **Decision**, write what it suggested, what you kept and what you rejected.

> **Check:** you have the assistant's hypotheses in your report, and each one is marked as confirmed or rejected by a line of evidence. If you have no assistant, ask a colleague to read your evidence instead, without your diagnosis.

## Step 6: recover and prove it

Find your row in the decision table (part 2 of the runbook). The evidence points to one row: the service broke right after `@champion` moved, the inputs are fine, and version 1, the previous champion, is servable and was healthy an hour ago. Roll back:

```bash
python exercises/quality-gates/promote.py --rollback --approver "<your name>"
```

> **Check:** `ROLLED BACK: f1-points-finish version 1 is @champion again (version 2 removed by <your name>).`

You did not restart anything. The container is still crash-looping, and Docker starts it again with a growing delay. At its next start, the API loads `@champion`, which is version 1 again. Wait, and repeat `curl -s http://127.0.0.1:8000/health` (or run `docker restart f1-api` to not wait).

> **Check:** within a minute (we measured 27 seconds) `/health` answers `"status":"ok"` and `"model_version":"1"`, and `docker ps` shows the container `Up`.

The service answers. Now prove that it is the service you had, with the lesson 05 checks (part 3 of the runbook):

```bash
pytest exercises/serve-model/solution/test_api.py
bash exercises/serve-model/load_test.sh
python exercises/recovery/triage.py labels
```

> **Check:** `8 passed`; `SLO MET` (in a container on a laptop we measured 247 requests and a p95 latency of 81 ms); `Model version 1` and `2026 overall: accuracy 0.706`, the same as in step 2.

Write down the time: the incident ends now. Complete your report: impact, timeline, root cause, contributing factors and actions. Think about **prevent** (who may move `@champion`?), **detect** (what alert would have fired on Sunday, before the restart?) and **respond**.

The service is back, but it is back to a model that gets 0.706 of the 2026 entries right. The hotfix author saw a real problem. The rest of this lesson solves it properly.

## Step 7: train a controlled challenger

A **challenger** is a model that competes with the champion on the same evidence before it may replace it. **Controlled** means that you change one thing and keep the rest. [`challenger.yaml`](challenger.yaml) keeps the champion's model, hyperparameters and seed, and changes only the training data:

| | Champion (version 1) | Challenger |
|---|---|---|
| Model | forest, 300 trees, depth 8, min_samples_leaf 5, seed 42 | the same |
| History | 2021-2024 | 2023-2025: the last three seasons |
| 2026 | none | rounds 1-4, with their **final** labels |
| Test | 2025 | 2026 rounds 5-8, for both models |

**Why this split.** There are eight 2026 races with final labels, and no split of eight races is good. These are the trade-offs:

- **Time order.** In production the model predicts a race after the races it learned from. A random split would put race 8 in training and race 2 in the test, and the test would be easier than real life.
- **Final labels only.** The provisional result is known on the evening of the race. The final one comes 2 to 26 hours later, after the stewards. 6 of the 160 provisional labels changed. A model must learn from, and be judged on, the final ones.
- **Four and four.** More training races teach the challenger more about 2026. More test races make the comparison less noisy. With four test races there are 80 entries, and one more right call moves the accuracy by 0.0125.
- **Both models are held out.** The champion never saw 2026. The challenger never sees rounds 5 to 8. So both are measured on races they did not see.
- **Decide before you look.** Each proposal you try on rounds 5 to 8 is one more chance to win by luck. Write the proposal down first, then measure it.

Open [`train_challenger.py`](train_challenger.py) and run it as it is:

```bash
python exercises/recovery/train_challenger.py
```

> **Check:** `STOPPED at TODO 1: labelled_2026() is not written yet. Nothing was trained.`

Complete **TODO 1 to TODO 3**:

| TODO | Function | What it does |
|---|---|---|
| 1 | `labelled_2026` | Joins the 2026 pre-race inputs with the **final** labels, on `race_id` and `driver_code` |
| 2 | `training_data` | The history seasons from `races.csv`, plus the 2026 rounds that the file lists, nothing else |
| 3 | `sample_weights` | One weight per entry: `weight_2026` for a 2026 entry, 1 for a history entry |

Delete the `raise NotImplementedError(...)` line in each function when you write it. Run the script again.

> **Check:**
>
> ```
> challenger-2023-2025+2026r1-4-w1: trained on 1520 entries (80 from 2026, final labels), history 2023-2025, 2026 rounds 1-4
> On the test window (2026 rounds 5-8, 80 entries): accuracy 0.738, roc_auc 0.756
> Registered f1-points-finish version 3 as @challenger (run ...)
> ```

## Step 8: compare champion and challenger

[`compare.py`](compare.py) loads `@champion` and `@challenger`, lets both predict the same 80 entries of rounds 5 to 8, and compares them against the rules in [`comparison.yaml`](comparison.yaml):

| Rule | Value | Why |
|---|---:|---|
| Accuracy gain over the champion | at least +0.05 | 4 more right calls in 80. Less is within the noise of four races. |
| ROC AUC | may fall by at most 0.02 | A challenger that is right more often but ranks the cars worse gives worse probabilities on the dashboard |
| Accuracy per rookie and per team_tier group | may fall by at most 0.10 | The lesson 04 rookie gate, extended to the team tiers: Cobalt Racing, a backmarker, leads 2026 |
| Margin over the grid rule | at least 0.02 | The lesson 04 gate, on the test window |

The challenger gates also check what lesson 04 and 05 checked: no after-race input, exactly the seven API inputs (so the API can serve it), predictions of 0 or 1, and a new one, `time_split`: no 2026 training race in the test window. The lesson 04 absolute thresholds (accuracy 0.82, ROC AUC 0.88) are not used: they were set for 2025, and in 2026 no model reaches them.

```bash
python exercises/recovery/compare.py
```

> **Check:**
>
> ```
> Champion v1 vs challenger v3 on 2026 rounds 5-8 (80 entries, final labels)
>
>   Measure                               Champion v1  Challenger v3  Change                  Rule
>   accuracy                                    0.688          0.738  +0.050  gain at least +0.050
>   roc_auc                                     0.763          0.756  -0.008     drop at most 0.02
>   margin over grid rule (0.675)               0.012          0.062                 at least 0.02
>   accuracy rookie (8)                         1.000          1.000  +0.000      drop at most 0.1
>   accuracy top (24)                           0.583          0.750  +0.167      drop at most 0.1
>   accuracy midfield (32)                      0.719          0.719  +0.000      drop at most 0.1
>   accuracy backmarker (24)                    0.750          0.750  +0.000      drop at most 0.1
>   predicted points share (actual 0.50)         0.61           0.59
>
>   Accuracy per race  2026-05  2026-06  2026-07  2026-08
>   Champion v1           0.60     0.65     0.75     0.75
>   Challenger v3         0.70     0.75     0.85     0.65
>
> Accuracy gain +0.050, 90 % bootstrap interval [-0.025, +0.125] over the 80 entries of 4 races.
> ```
>
> Then nine `PASSED` gates and `RECOMMENDATION: PROMOTE the challenger (version 3). It passed all 9 gates.`, and `@candidate -> version 3`.
>
> We measured these numbers on Linux, like the codespace and GitHub Actions. On a Mac the accuracies are the same, but roc_auc can differ in the last digit (there the champion reads 0.762): a forest trained on another kind of computer can differ by a hair, and one pair of cars ordered differently moves ROC AUC by 0.0006.

Read it honestly before you celebrate:

- The gain is **exactly** the minimum: 4 more right calls in 80. All of it comes from the top teams (+0.167 on 24 entries); midfield and backmarkers did not change.
- The challenger is better in three races and worse in one (2026-08).
- The 90 % interval of the gain goes from -0.025 to +0.125. It includes zero: on these 80 entries we cannot exclude that the challenger is no better at all.
- The rookie group has 8 entries. Both models got all 8 right. That says almost nothing.
- The champion beats the grid rule by only 0.012 in 2026. It would fail its own lesson 04 margin gate.
- We also measured each history window on rounds 2, 3 and 4, one race at a time, before the test window: all of 2021-2025 got 0.683 right, the last three seasons 0.667. The earlier races do not favour the recent window.

So the rules say PROMOTE, and the evidence says: probably not worse, perhaps better, from four races. Whether to promote is a decision for a person. Step 9 is that decision.

## Step 9: promote through the gated path

`compare.py` wrote `gate_report.json` in the lesson 04 format, for version 3, and moved `@candidate` to version 3. So the promotion takes the same path as every promotion since lesson 04: `promote.py` reads the report, refuses if a gate failed, and names the person who approved.

Decide with your group, using the evidence of step 8 and the trade-offs below. We promote: the challenger is not worse on any rule, the champion is known to be weak in 2026, and you proved in step 6 that a rollback takes seconds. We write down when we look again: after rounds 9 to 12 have final labels.

```bash
python exercises/quality-gates/promote.py --approver "<your name>"
curl -s -X POST http://127.0.0.1:8000/admin/reload
```

> **Check:** nine `PASSED` gates, then `PROMOTED: f1-points-finish version 3 is now @champion (previous champion: 1). Approved by <your name>.` The reload answers `"previous_version":"1","model_version":"3"`.

A new champion is a release, so run the release checks:

```bash
pytest exercises/serve-model/solution/test_api.py
bash exercises/serve-model/load_test.sh
```

> **Check:** `8 passed`, and `SLO MET` (we measured 266 requests and a p95 latency of 71 ms). The challenger has the same size as the champion, so it is as fast.

`triage.py labels` now shows version 3 with 0.885 on 2025 and 0.738 on 2026. Neither number is a fair test: version 3 trained on 2025 and on rounds 1 to 4. Only the test window of step 8 is fair. Remember that when you read monitoring numbers after a retrain.

Now watch the same path block a proposal. Train the obvious retrain, all history plus rounds 1 to 4, and compare it with the new champion:

```bash
python exercises/recovery/train_challenger.py --challenger exercises/recovery/examples/all_history.yaml
python exercises/recovery/compare.py
python exercises/quality-gates/promote.py --approver "<your name>"
```

> **Check:** the training says `accuracy 0.713, roc_auc 0.749`. `compare.py` says `RECOMMENDATION: KEEP the champion (version 3). The challenger (version 4) failed 2 of 9 gates.`: `accuracy gain -0.025 is below +0.050` and `accuracy for top (24) falls by 0.125, more than 0.1`. `promote.py` says `REFUSED: the candidate cannot be promoted.` with the same two gates.

Try the other two examples the same way:

| Proposal | What it changes | We measured (against version 3) |
|---|---|---|
| [`examples/heavy_2026.yaml`](examples/heavy_2026.yaml) | Each 2026 entry counts five times | accuracy 0.750 (+0.012), roc_auc 0.712 (-0.044): KEEP, `accuracy_gain` and `roc_auc_not_worse` failed |
| [`examples/peek.yaml`](examples/peek.yaml) | Trains on all eight 2026 races | accuracy 0.800 (+0.062), roc_auc 0.901: KEEP, `time_split` failed: `trained on 2026 rounds [5, 6, 7, 8]: the test window starts at round 5` |

`peek.yaml` is the best model on paper. It has seen the answers.

## Step 10: run the comparison on GitHub Actions

The workflow [`challenger.yml`](../../.github/workflows/challenger.yml) does steps 1, 7 and 8 in one job, with its own MLflow store:

| Step | What it does |
|---|---|
| 1. Build the champion | The lesson 04 pipeline with the reference gates. The champion is version 1. |
| 2. Train the challenger | The training script on the challenger file. Both are inputs of the workflow. |
| 3. Compare champion and challenger | `compare.py`. It writes the comparison and the gate table to the run page. The job fails unless the recommendation is PROMOTE. |

It keeps `comparison_report.md`, the gate report and the MLflow store as the artifact `comparison`. It does not promote: promotion stays in the lesson 04 path, with an approval.

The training script is an input, `trainer`. It defaults to the reference solution, so the workflow works before you finish the TODOs. Set it to `exercises/recovery/train_challenger.py` to run yours.

1. **Commit your work** (your TODOs and your incident report) and push it to your fork, as in lesson 05 step 9. The push starts the workflow **challenger**.
2. **Make `07-recovery` the default branch of your fork** (**Settings → General → Default branch**), so that GitHub shows the **Run workflow** button.
3. Open **Actions → challenger → Run workflow**. Keep the challenger `exercises/recovery/challenger.yaml`, set the trainer to `exercises/recovery/train_challenger.py`, and click **Run workflow**.

> **Check:** the job **Champion vs challenger** succeeds in about a minute. The run page shows the comparison table (accuracy 0.688 against 0.738, gain +0.050; roc_auc 0.763 against 0.756), the per-race table, `Challenger gates: f1-points-finish version 2: passed`, and `RECOMMENDATION: PROMOTE the challenger (version 2). It passed all 9 gates.` On GitHub the challenger is version 2: there is no incident there.

Now a blocked run: **Run workflow** again, with the challenger `exercises/recovery/examples/all_history.yaml`.

> **Check:** step **3. Compare champion and challenger** fails with `Process completed with exit code 1`. The run page shows accuracy 0.688 against 0.713 (gain +0.025, interval [+0.000, +0.050]), the gate `accuracy_gain` as FAIL with `accuracy gain +0.025 is below +0.050`, `Challenger gates: f1-points-finish version 2: BLOCKED` and `RECOMMENDATION: KEEP the champion (version 1). The challenger (version 2) failed 1 of 9 gates.`

Here the all-history proposal is compared with version 1, not with your version 3, so it fails one gate, not two.

## Trade-offs: what to do when the model gets worse

| Option | What it changes | Cost | Risk | Choose it when |
|---|---|---|---|---|
| **Roll back** | Which version serves: one alias | Seconds. A crash-looping container came back in 27 seconds. | You go back to the previous model, with its weaknesses (here: 0.706 on 2026). Needs a previous version that is known to be good. | A release broke the service or its answers |
| **Recalibrate** | The probabilities, not the order of the cars (ROC AUC stays the same) | Small: a calibration step on a few races of final labels, then a release | With few labels the calibration itself is noisy. It barely changes which cars are predicted to score. | People use the probabilities and they are off: in rounds 5 to 8 the champion predicts points for 0.61 of the entries, and 0.50 scored |
| **Retrain a challenger** | What the model learned | Hours: wait for final labels, train, compare, gates, approval. The training itself takes seconds. | A small test window (here 80 entries): a "win" can be luck, and each extra proposal makes luck more likely | The inputs are valid, the world changed, and accuracy on final labels fell over several races |
| **Wait for labels** | Nothing, yet | The users keep a weaker model for longer. Tell them, with a date when you look again. | If the drop is real, you act late | The drop is in one or two races, or only on provisional labels |

None of them is free, and the fastest one does not fix the cause. In this lesson you used two in a row: a rollback to stop the outage, then a challenger for the slow decline.

## Discuss

1. The hotfix author saw a real problem and tried to fix it. What should they have done instead? Which of these would have stopped version 2: the lesson 04 gates, the API contract, an approval, registry permissions? Which one did stop it, and when?
2. You rolled back to a model that you knew gets 0.706 right in 2026. Was that the right call, or should you have fixed forward? When is fixing forward the better choice?
3. The challenger met the rules with exactly the minimum gain, and the interval of its gain includes zero. Would you have approved it? What would make you roll it back, and when will you know? This lesson tried four proposals on the same four races: what does that do to the meaning of "the challenger won"?
4. An AI assistant read your evidence and suggested causes and commands. Who is accountable for a command that it suggested and you ran? What evidence should never be pasted into an assistant?

## Stretch

- **Rolling origin.** Change `train_challenger.py` into a small backtest: for each round k from 2 to 8, train on the rounds before k and test on round k. Does the recent window still win when every race is a test race once?
- **Recalibrate.** Fit a logistic regression on the champion's probabilities for rounds 1 to 4 (Platt scaling) and measure the Brier score (`sklearn.metrics.brier_score_loss`) on rounds 5 to 8, before and after. What happens to ROC AUC, and why?
- **Prevent the incident.** Write a check that fails when `@champion` points to a version without the tag `gates`, and run it before the nightly restart. Where else could it run?
- **The other incident.** In lesson 06 a new feed sent `quali_gap_ms` in seconds. Which runbook question would show it, which row of the decision table applies, and why is a rollback wrong there?
- **Draft the report with an assistant.** Give it `incident_evidence.md` and your timeline, ask for a draft report, then check every sentence against the evidence. Count the sentences you had to correct.

## Stuck?

There is a reference solution in [`solution/train_challenger.py`](solution/train_challenger.py). Try the TODOs first. To run it:

```bash
python exercises/recovery/solution/train_challenger.py
```

Each run registers a new version and moves `@challenger` to it. To compare a proposal again, train it again.
