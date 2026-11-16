# Scenario check: answer key

For the trainer. The numbers on the cards are the ones we measured in lesson 09 (cards 1, 2, 4, 6, 7) and lesson 05 (card 3). Accept other reasonable next actions; the decision and the evidence must match.

| Card | Decision | The evidence that decides | Controls |
|---|---|---|---|
| 1. Reload endpoint is public | **Do not ship**, and treat it as a security incident | Gate 6 Access tests FAIL; the image is not the accepted one (gate 8 names `lab09-unsecured:1`) | NIST MEASURE 2.7; SDAIA Privacy & Security |
| 2. Model file hash changed | **Do not ship.** Do not sign again | Gate 2 Integrity FAIL: `model/model.skops: CHANGED` | NIST MEASURE 2.7; SDAIA Privacy & Security |
| 3. p95 doubled | **Ship is allowed** (the human decides), with the regression recorded | Gate 7 SLO MET, 150 ms against 500 ms | NIST MEASURE 2.3 |
| 4. Data md5 changed | **Do not ship** this release. Retrain, gate and sign a new version | Gate 1 FAIL (md5 differs from the training data) and gate 2 FAIL | NIST MAP 2.3, MEASURE 2.7 |
| 5. Evidence missing | **Do not ship** | Gates 9 and 10 EVIDENCE MISSING | NIST MEASURE 2.9, MEASURE 2.11 |
| 6. Key in the repository | **Do not ship** until the key is rotated | Gate 6 FAIL: `a key is in a file that git tracks` | NIST MEASURE 2.7; SDAIA Privacy & Security |
| 7. Vulnerability that "does not matter" | **Do not ship as it is.** Remove pip from the image (the lesson 09 Dockerfile does) | Gate 8 FAIL, 6 open findings in `pip 25.0.1` | NIST MEASURE 2.7, MANAGE 3.1 |
| 8. Approved by the author | The decision stands technically, but **the process failed**: get a second, independent approver | The warning in the decision block | NIST GOVERN 2.1, MEASURE 1.3 |

## Card 1: the reload endpoint is public

- **Decision:** do not ship. The service that runs is not the release that was accepted.
- **Evidence:** the access tests fail exactly where lesson 05 left the API open: `/v1/predict` and `/admin/reload` answer 200 without a key. The image name and its 6 open findings show that the lesson 05 image was deployed, not `lab09-api`.
- **Next action:** replace the container with the accepted image, with keys from `keys.py`. Rotate the keys if they were ever shared with the wrong image. Read the container log: who called `/admin/reload`, and which version was served after each call (each answer names its version). Tell the service owner and the security owner. Add a check that the running image is the accepted one (compare the image ID in the dossier with the running container).

## Card 2: the model file hash changed

- **Decision:** do not ship, and do not run `manifest.py sign --force`. Signing again would turn the alarm off without knowing why it rang.
- **Evidence:** the integrity gate. All other gates passing is expected: the tampered file is a real, working model. The clues are in the numbers: ROC AUC 0.907 → 0.905 and p95 75 ms → 33 ms. A model that got faster without a release is a smaller model. We measured exactly this with `tamper.py --target model`: 30 of the 300 trees.
- **Next action:** treat it as a security incident. Find out who could write to the artifact store. Restore the model from a trusted copy (or retrain and register a new version), verify against the signed manifest, and reload. The security owner leads.

## Card 3: p95 doubled after promotion

- **Decision:** the suite recommends a human decision, and shipping is allowed: the SLO is met with room to spare (150 ms of 500 ms).
- **Evidence:** gate 7. In lesson 05 we measured the same good model between 71 and 190 ms in different runs on a 2-CPU container, so a single run of 75 ms against 150 ms may be noise.
- **Next action:** run the load test again two or three times before you decide. If the slowdown is real, record it in the dossier reason, and ask the service owner whether the error budget and the capacity still fit. A model that costs twice the CPU for the same accuracy is a cost decision, not only an SLO decision.

## Card 4: the data md5 changed

- **Decision:** do not ship this release.
- **Evidence:** gate 1 still finds 0 failure cases: the new value is valid. But the md5 is no longer the one in the training run (`data.version` tag), and gate 2 shows the file changed after the sign-off. The gates now describe data that the model was not trained on.
- **Next action:** if the correction is legitimate, the data owner commits it with a reason, then the pipeline trains, gates and promotes a **new** version, and the release is signed and accepted again. The old version keeps its manifest. If nobody can explain the change, restore the file (`git checkout f1/data/races.csv`) and investigate.

## Card 5: everything passed, but EVIDENCE MISSING

- **Decision:** do not ship. Missing evidence is not a pass.
- **Evidence:** gates 9 and 10. A model card from an earlier release describes another version: `accept.py` refuses a card for another version, and so should the approver.
- **Next action:** run `subgroup_check.py` (or attach the lesson 08 fairness report) and write the model card for this version. Both take minutes. The product owner's deadline does not change which evidence the release needs.

## Card 6: a key in the repository

- **Decision:** do not ship until the key is rotated.
- **Evidence:** the access test `test_keys_are_not_committed` (we measured this failure by putting a key into `slo.yaml`).
- **Next action:** rotate both keys with `keys.py --rotate` and restart the API with the new ones. Deleting the line in a new commit is not enough: the key stays in the git history and in every fork and clone. Tell the security owner. Then remove the line, and give the load test the key from the environment (`API_KEY`), as `load_test.sh` does.

## Card 7: the vulnerability that "does not matter"

- **Decision:** do not ship the image as it is.
- **Evidence:** gate 8. "Never runs pip" may be true, but pip is still in the image, and anyone who gets a shell in the container can use it.
- **Next action:** remove what the service does not need: the lesson 09 Dockerfile runs `pip uninstall --yes pip` after installing the packages, and the same audit then finds 0 open findings in 56 packages. Accepting a finding is a valid answer only when no fix exists: then write the reason, a named owner and an expiry date in `release.yaml`, so that the acceptance expires and somebody looks again.

## Card 8: approved by the author

- **Decision:** the release passed every gate, so "ship" is not wrong on the evidence. But nobody independent looked at it.
- **Evidence:** the warning `the approver is also the model owner: no separation of duties`.
- **Next action:** add a second approver who did not build the model (for example the service or security owner) to `approvers`, and in GitHub add them as a required reviewer of the environment `production`, with **Prevent self-review** on. NIST AI RMF MEASURE 1.3 asks for assessors who did not serve as front-line developers; GOVERN 2.1 asks for documented roles, which `owners` in `release.yaml` are.
