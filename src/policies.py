"""FCFS and greedy seating rules from the project report."""

from __future__ import annotations

import numpy as np

from env import TABLES


def fcfs(env) -> int:
    """Seat at the first free table that fits, walking the room from Table 1.

    The report's FCFS host seats a party whenever any table fits, and does not
    save a tighter table. Waiting is only used when nothing fits.
    """
    mask = env.action_mask()
    for i in range(5):
        if mask[i + 2]:
            return i + 2
    return 0


def greedy(env) -> int:
    """Best-fit seating, with a hard rule against wasting the large tables.

    Reservations, near-fits, and late service are seated. A party of one or two
    is turned away only when every open table is at least three seats too big
    and the peak is still ahead. That is the report's fit-based heuristic.
    """
    mask = env.action_mask()
    party = env.queue[0]
    candidates = [i for i in range(5) if mask[i + 2]]
    if not candidates:
        return 0
    best = min(candidates, key=lambda i: (TABLES[i] - party.size, TABLES[i], i))
    slack = TABLES[best] - party.size
    if party.reservation or slack <= 1 or env.phase_name() == "Late":
        return best + 2
    if party.size <= 2 and slack >= 3:
        return 1
    return best + 2


def masked_argmax(q: np.ndarray, mask: np.ndarray) -> int:
    scored = np.where(mask, q, -1e9)
    return int(np.argmax(scored))
