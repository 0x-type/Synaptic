"""Verdict rules enforced in code (not only in prompts): once a claim has a verdict, later actors can only
LOWER it. An auditor, a referee, a follow-up or a lab meeting can never upgrade a family to "new" or a hypothesis
to "supported". The only upward steps are taken by code gates: the held-out confirmation run and the novelty gate.

Families:   artifact (0) < not_robust (1) < known_phenomenon (2) < new_signal (3)
Hypotheses: refuted/dropped (0) < not_shown = inconclusive (1) < known (2) < supported (3) < supported_new (4)
            pre-result states: reserve, open, testing, awaiting_confirmation
"refuted" means a test CONTRADICTED the claim (significant, opposite direction). A test that is simply not
significant is "not_shown": absence of evidence is not evidence of absence.
"""
from __future__ import annotations

FAMILY_RANK = {"artifact": 0, "not_robust": 1, "known_phenomenon": 2, "new_signal": 3}
HYP_RANK = {"refuted": 0, "dropped": 0, "inconclusive": 1, "not_shown": 1, "known": 2, "supported": 3, "supported_new": 4}
HYP_PRE = {"reserve", "open", "testing", "awaiting_confirmation", "", None}
CODE_KILL_TAGS = ("[independent check]", "[robustness suite]", "[screened automatically]", "[field test]")
CODE_ACTORS = {"engine", "confirmation", "novelty_gate"}


class VerdictRefused(Exception):
    pass


def family_change(current: str | None, new: str | None, actor: str, note: str = "") -> tuple[bool, str]:
    """May `actor` move a family from `current` to `new`? Returns (allowed, reason)."""
    if new == current or new is None:
        return True, "unchanged"
    if new not in FAMILY_RANK:
        return False, f"unknown outcome {new}"
    if actor == "operator":
        return True, "human operator"
    if current and any(t in (note or "") for t in CODE_KILL_TAGS):
        return False, f"closed by a code check ({current}); no agent can reopen it"
    if not current:
        if new == "new_signal" and actor not in ("deep_dive", "engine"):
            return False, "only a deep dive (through the novelty gate) can make the first new_signal claim"
        return True, "first verdict"
    if FAMILY_RANK[new] > FAMILY_RANK.get(current, 3):
        return False, f"verdicts can only go down ({current} -> {new} refused for {actor})"
    return True, "downgrade"


def hypothesis_change(current: str | None, new: str | None, actor: str) -> tuple[bool, str]:
    """May `actor` move a hypothesis from `current` to `new`?"""
    if new == current or new is None:
        return True, "unchanged"
    if actor == "operator":
        return True, "human operator"
    if new in HYP_PRE:
        if current in HYP_PRE or current is None:
            return True, "still before a result"
        return False, f"a result ({current}) cannot be reopened by {actor}"
    if new not in HYP_RANK:
        return False, f"unknown status {new}"
    if new == "supported":
        if actor != "confirmation":
            return False, "only the held-out confirmation run can set 'supported'"
        # a hypothesis the explorers could not test (its claim lives in held-out data) gets its ONE pre-registered
        # confirmation too
        return (current in ("awaiting_confirmation", "inconclusive", "not_shown")), "confirmation"
    if new == "supported_new":
        if actor != "novelty_gate" or current != "supported":
            return False, "only the novelty gate can raise a CONFIRMED result to supported_new"
        return True, "novelty gate"
    if current in HYP_PRE or current is None:
        return True, "first result"
    if current in ("supported", "supported_new") and actor == "agent":
        return False, ("a result confirmed on held-out data is lowered only by a referee, the auditor or the PI, "
                       "not by an exploring agent that cannot see that data")
    if HYP_RANK[new] > HYP_RANK.get(current, 4):
        return False, f"verdicts can only go down ({current} -> {new} refused for {actor})"
    return True, "downgrade"
