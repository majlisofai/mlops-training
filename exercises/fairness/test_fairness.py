"""Fairness gates for f1-points-finish: rookie entries against everyone else, on 2025.

Your task: complete TODO 4. The first two gates are done for you.
The fixtures (criteria, holdout, predictions, gaps, grid_gaps, warn) come from conftest.py.
A gap is the rookie rate minus the rate of everyone else: positive means worse for rookies.
See exercises/fairness/README.md for the steps.

A gate that is still a TODO is skipped. A skipped gate checks nothing.

Usage:
    pytest exercises/fairness/test_fairness.py exercises/fairness/test_privacy.py
"""

import pytest


def test_rookie_group_is_large_enough(gaps, criteria, record_property, warn):
    """The rookie group has enough entries to measure; warn when the numbers are shaky."""
    f = criteria["fairness"]
    group = gaps["group"]
    record_property("rookie_entries", group["entries"])
    record_property("rookie_drivers", gaps["drivers"])
    if gaps["drivers"] < f["warn_group_drivers"]:
        warn(f"{group['entries']} rookie entries come from only {gaps['drivers']} drivers")
    if group["non_scorers"] < f["warn_rate_base"]:
        warn(f"the rookie FPR rests on {group['non_scorers']} entries that did not score")
    if group["scorers"] < f["warn_rate_base"]:
        warn(f"the rookie FNR rests on {group['scorers']} entries that scored")
    assert group["entries"] >= f["min_group_entries"], (
        f"only {group['entries']} rookie entries, fewer than {f['min_group_entries']}: the gaps cannot be measured"
    )


def test_rookie_fpr_gap(gaps, grid_gaps, criteria, record_property):
    """Rookies are not predicted to score when they do not, much more often than everyone else."""
    gap = gaps["fpr_gap"]
    limit = criteria["fairness"]["max_fpr_gap"]
    record_property("rookie_fpr", f"{gaps['group']['false_positive_rate']:.3f}")
    record_property("others_fpr", f"{gaps['rest']['false_positive_rate']:.3f}")
    record_property("fpr_gap", f"{gap:+.3f}")
    record_property("grid_rule_fpr_gap", f"{grid_gaps['fpr_gap']:+.3f}")
    assert gap <= limit, f"rookie false positive rate gap {gap:+.3f} is above {limit}"


def test_rookie_fnr_gap(gaps, grid_gaps, criteria, record_property):
    """Rookies who score are not written off much more often than everyone else."""
    # TODO 4: write this gate like test_rookie_fpr_gap, for the false negative rate:
    #         the gap is gaps["fnr_gap"], the limit is criteria["fairness"]["max_fnr_gap"].
    #         Record the rookie FNR (gaps["group"]["false_negative_rate"]), everyone else's FNR
    #         (gaps["rest"]["false_negative_rate"]), the gap, and the grid rule's gap
    #         (grid_gaps["fnr_gap"]). Assert that the gap is at most the limit.
    #         Then delete the pytest.skip line.
    pytest.skip("TODO 4: the gate 'rookie FNR gap' is not written yet")
