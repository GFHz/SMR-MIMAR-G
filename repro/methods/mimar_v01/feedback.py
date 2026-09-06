"""Feedback outcomes and state transitions."""
from .interest_profile import update_active_interests

ACCEPT = "ACCEPT"
REJECT = "REJECT"


def apply_feedback(active, memory, route, item, outcome):
    if outcome == ACCEPT:
        memory.advance()
        return update_active_interests(active, item["genres"], accepted=True)
    if outcome == REJECT:
        memory.advance(route=route, rejected_item_id=item["id"])
        return update_active_interests(active, item["genres"], accepted=False)
    raise ValueError("Feedback must be ACCEPT or REJECT")
