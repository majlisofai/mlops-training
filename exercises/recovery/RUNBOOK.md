# Runbook: the f1-points-finish service has an incident

Use this runbook when the f1-points-finish API is down, gives wrong answers, or its predictions get worse. It has five parts: triage, decide, recover and verify, ask an AI assistant, and write the incident report.

**Three rules for the whole incident**

1. **Evidence before action.** Collect the evidence of part 1 before you change anything. A rollback, a restart or a retrain destroys evidence.
2. **Write the time down.** Note the time (UTC) of every observation and every action. The incident report needs a timeline.
3. **Change one thing at a time, and check after each change.** Then you know which change fixed the service.

Run every command from the repository root. The commands use the codespace defaults: the API at `http://127.0.0.1:8000`, the container `f1-api`, the MLflow server at `http://127.0.0.1:5000`.

## 1. Triage: collect the evidence

Answer the five questions in order. [`triage.py`](triage.py) collects the evidence and changes nothing. `python exercises/recovery/triage.py all --out incident_evidence.md` runs all five and keeps a copy.

| # | Question | Commands | Healthy looks like |
|---|---|---|---|
| 1 | **Health.** Is the API up? Which model version does it serve? | `curl -s http://127.0.0.1:8000/health`<br>`docker ps -a --filter name=f1-api`<br>`triage.py health` | `"status":"ok"` and the version that the registry says is `@champion`. The container is `Up`, with 0 restarts. |
| 2 | **Version.** Which version is `@champion` in the registry? Who moved it, when, and did the gates run? Can the API serve it? | `triage.py registry`, or the MLflow UI: **Models → f1-points-finish** | `@champion` has `promoted_by` a person, `gates: 7 passed` (or the challenger gates), and `inputs: 7, the API v1 inputs` |
| 3 | **Logs.** What did the service say, and since when? | `docker logs --timestamps f1-api 2>&1 \| tail -40`<br>`triage.py health` | `Serving f1-points-finish version N`, no `ERROR`, no `startup failed` |
| 4 | **Inputs.** Do the clients send valid inputs, in the units of the training data? | `triage.py inputs` | All entries keep the contract. No input moved by half a standard deviation or more. A unit change shows as a median that is 1000 times smaller or larger. |
| 5 | **Performance on delayed labels.** How right is the model on the races whose final labels have arrived? | `triage.py labels` (what the API serves)<br>`triage.py labels --model-uri models:/f1-points-finish/<version>` (one version, offline) | Accuracy near the accuracy of the release, measured on races the model did not train on |

Final labels arrive 2 to 26 hours after a race, when the stewards' decisions are in. Never judge a model on the provisional results alone, and never on a race it was trained on.

## 2. Decide: roll back, fix the input, or retrain

Find the row that matches your evidence. If two rows match, deal with the one that stops the service first.

| Evidence | Likely cause | Action | Do not |
|---|---|---|---|
| The service broke, or answers changed, right after `@champion` moved. The previous champion was healthy. | A bad release | **Roll back**: `promote.py --rollback --approver "<you>"`, then `POST /admin/reload` or restart the container | Retrain in a hurry. You would ship a second untested model. |
| Some inputs break the contract (422 answers), or an input moved in a way that looks like a unit or format change. No release happened. | A fault in a client or a feed | **Fix the input**: the contract rejects what it can (422). Convert at the boundary only when the conversion is certain. Tell the owner of the feed. | Retrain on the broken inputs. Roll back: the model is not the problem. |
| All inputs are valid, but some moved for a real reason. Accuracy on final labels fell over several races. No release explains it. | The world changed: the model is out of date | **Retrain a challenger**, compare it with the champion on the same held-out races, promote it only through the gates and an approval. Keep serving the champion meanwhile, and tell the users how good it is now. | Promote a retrained model without a comparison. |
| Accuracy fell on one or two races, or only on the provisional labels. | Noise, or labels that are not final yet | **Wait for the labels** and the next races. Write down when you will look again. | Act on two races. |
| The container stops for lack of memory, or MLflow does not answer. The model and the inputs are fine. | The platform | **Fix the platform**: memory, disk, the MLflow server | Change the model. |

A rollback needs a previous champion that is known to be good. `promote.py --rollback` goes to the version in the `previous_champion` tag of the current champion. Check that version with `triage.py registry` before you roll back.

## 3. Recover and verify

1. Write down the time and the action you chose, and why.
2. Do the action. For a rollback:

   ```bash
   python exercises/quality-gates/promote.py --rollback --approver "<your name>"
   ```

   A running API picks the new champion up with `curl -s -X POST http://127.0.0.1:8000/admin/reload`. A container that crash-loops picks it up at its next restart, within a minute. `docker restart f1-api` does it at once.
3. Prove that the service has recovered, with the lesson 05 checks:

   | Check | Command | Recovered when |
   |---|---|---|
   | Health | `curl -s http://127.0.0.1:8000/health` | `"status":"ok"` and the expected version |
   | Contract | `pytest exercises/serve-model/solution/test_api.py` | `8 passed` |
   | Service-level objective | `bash exercises/serve-model/load_test.sh` | `SLO MET` |
   | Delayed labels | `triage.py labels` | The accuracy that the version had before the incident |

4. Write down the time when every check passed. The incident ends then, not at the time of the rollback.

## 4. Ask an AI assistant, safely

An AI assistant (for example GitHub Copilot Chat in the codespace) reads evidence fast and suggests causes you did not think of. It also states wrong things with confidence. Use it as a second opinion, not as the incident lead.

1. Give it the evidence, not your conclusion: the file `incident_evidence.md` from `triage.py all --out`. Remove tokens, passwords and personal data first.
2. Ask for hypotheses, each with the evidence for it and against it:

   ```text
   You are helping with an incident on a model-serving API. Below is the evidence that our
   triage script collected. List the three most likely causes. For each cause, quote the lines
   of evidence that support it and the lines that speak against it, and say which one command
   would confirm or rule it out without changing anything. Do not suggest fixes yet.
   ```

3. Check every claim against the evidence yourself. A line that the assistant quotes must be in the file.
4. Choose the action with the decision table in part 2. The assistant may help you write the commands; a person runs them and is named as the approver.
5. Never let an assistant run a command that changes the registry or the service without a person reading it first.

## 5. Incident report

Copy the template below into `exercises/recovery/incident_report.md` and fill it in. Write facts with times and evidence, not blame. The report is for the next person on call.

```markdown
# Incident report: <one line: what users saw>

- Status: resolved | monitoring
- Incident lead: <name>        Scribe: <name>
- Severity: <1 = service down for everyone, 2 = degraded, 3 = minor>

## Summary
<Three sentences: what happened, what you did, how you know it is fixed.>

## Impact
- Start (UTC): <first sign in the evidence, not when you noticed>
- End (UTC): <when every check in part 3 passed>
- Duration: <minutes>
- Who and what was affected: <for example: no predictions for the dashboard>

## Timeline (UTC)
| Time | What happened or what we did | Evidence |
|---|---|---|
| | | |

## Evidence
| Question | What we found | Command |
|---|---|---|
| 1. Health | | |
| 2. Version | | |
| 3. Logs | | |
| 4. Inputs | | |
| 5. Delayed labels | | |

## Root cause
<The first cause that, if it had not happened, would have prevented the incident.>

## Contributing factors
<What made it possible, or worse, or slower to find.>

## Decision
<The row of the decision table, the action, and why. What the AI assistant suggested, and what you kept or rejected.>

## Recovery and proof
<Commands and their measured results: health, contract tests, SLO, delayed labels.>

## What went well, what did not

## Actions
| Action | Type (prevent / detect / respond) | Owner | Due |
|---|---|---|---|
| | | | |
```
